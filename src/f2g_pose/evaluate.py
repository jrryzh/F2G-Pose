import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from .data import build_manifest, PoseDataset, collate_records, manifest_text
from .io import write_json
from .metrics import compute_criterion,summarize


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def environment(device=None):
    import importlib.metadata as md
    import platform
    import torch
    packages={}
    for name in ('torch','torchvision','numpy','scipy','timm','cutoop','gradio','pointnet2-ops'):
        try:packages[name]=md.version(name)
        except md.PackageNotFoundError:packages[name]=None
    flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
               cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
               float32_matmul_precision=torch.get_float32_matmul_precision(),
               flash_sdp=torch.backends.cuda.flash_sdp_enabled(),memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled())
    for key in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
        value=os.environ.get(key)
        flags[key]=value if value in (None,'0','1') else 'set_other'
    return dict(packages=packages,python=platform.python_version(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,backend_flags=flags)


def manifest_is_full(path, count, digest, split):
    """A custom manifest needs matching full-run provenance to claim full coverage."""
    provenance=Path(path).with_name('run.json')
    if not provenance.is_file():return False
    source=json.loads(provenance.read_text())
    return (source.get('protocol')=='full' and source.get('limit') is None and
            source.get('split')==split and source.get('shard_count',1)==1 and
            source.get('candidate_count')==count and source.get('candidate_manifest_sha256')==digest)


def evaluate(args):
    import torch
    from cutoop.obj_meta import ObjectMetaData
    from .estimator import PoseEstimator
    if not 0<=args.shard_index<args.shard_count:raise ValueError('Shard index must be in [0, shard-count).')
    if args.manifest and args.limit:raise ValueError('Use a limited manifest instead of combining --manifest and --limit.')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='building_manifest'))
    start=time.perf_counter()
    if args.manifest:
        records=[json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
        excluded=[json.loads(line) for line in Path(args.metadata_exclusions).read_text().splitlines()] if args.metadata_exclusions else []
        frames=len({r['frame'] for r in records})
    else:
        records,excluded,frames=build_manifest(args.data_root,args.split,args.protocol,args.seed,args.limit)
    source_count=len(records)
    source_digest=hashlib.sha256(manifest_text(records).encode()).hexdigest()
    source_is_full_test=(args.protocol=='full' and args.limit is None and
                        (not args.manifest or manifest_is_full(args.manifest,source_count,source_digest,args.split)))
    begin=source_count*args.shard_index//args.shard_count
    end=source_count*(args.shard_index+1)//args.shard_count
    records=records[begin:end]
    text=manifest_text(records);(out/'candidate_manifest.jsonl').write_text(text)
    (out/'metadata_excluded.jsonl').write_text(manifest_text(excluded))
    (out/'excluded.jsonl').write_text('')
    meta_path=Path(args.object_meta) if args.object_meta else Path(args.data_root).parent/'Meta'/('real_obj_meta.json' if args.split=='rope' else 'obj_meta.json')
    objmeta=ObjectMetaData.load_json(str(meta_path))
    config=dict(vars(args));config.update(checkpoint_sha256=sha256(args.checkpoint),object_meta_sha256=sha256(meta_path),
                                        candidate_manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),candidate_count=len(records),frame_count=frames,
                                        source_manifest_sha256=source_digest,source_candidate_count=source_count,shard_begin=begin,shard_end=end,
                                        source_is_full_test=source_is_full_test,
                                        implementation_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                                        model_config_sha256=sha256(args.config) if args.config else None,
                                        environment=environment(args.device),randomness='fixed per relative frame/object/mask/seed; historical RNG state not recoverable')
    write_json(out/'run.json',config)
    print(f'{args.split} {args.protocol}: {frames} frames, {len(records)} candidate objects',flush=True)
    estimator=PoseEstimator(args.checkpoint,config=args.config,radio_repository=args.radio_repository,radio_checkpoint=args.radio_checkpoint,device=args.device)
    dataset=PoseDataset(args.data_root,records,seed=args.seed,image_size=estimator.config['image_size'])
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,
                                       collate_fn=collate_records,pin_memory=False)
    (out/'parts').mkdir()
    pending=[];evaluated=[];chunks=[];rejected=0
    def collect(entry):
        number,future,part,rows=entry
        iou,deg,cm=future.result()
        part.update(iou=iou,degrees=deg,shift_cm=cm)
        np.savez_compressed(out/'parts'/f'{number:06d}.npz',**part)
        evaluated.extend(rows);chunks.append(part)
        with (out/'sample_manifest.jsonl').open('a') as f:f.write(manifest_text(rows))
        write_json(out/'status.json',dict(status='running',evaluated=len(evaluated),excluded=rejected,elapsed_s=time.perf_counter()-start))
    with ProcessPoolExecutor(max_workers=args.metric_workers,mp_context=mp.get_context('spawn')) as pool:
        for index,items in enumerate(loader):
            valid=[]
            for item in items:
                if 'error' in item:
                    rejected+=1
                    with (out/'excluded.jsonl').open('a') as f:f.write(manifest_text([dict(**item['record'],reason=item['error'])]))
                elif item['record']['object_id'] not in objmeta.instance_dict:
                    rejected+=1
                    with (out/'excluded.jsonl').open('a') as f:f.write(manifest_text([dict(**item['record'],reason='object absent from object metadata')]))
                else:valid.append(item)
            if not valid:continue
            inputs=[torch.from_numpy(np.stack([getattr(i['observation'],key) for i in valid])).to(estimator.device) for key in ('points','rgb','rows','columns')]
            with torch.inference_mode():
                with torch.autocast('cuda',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16,enabled=args.precision!='fp32'):
                    ret=estimator.model(*inputs,return_shape=False)
                from .geometry import compute_rotation_matrix_from_ortho6d
                # Historical get_dm converted all pose/size heads to FP32 before decoding.
                rotation=compute_rotation_matrix_from_ortho6d(ret[2].float()).cpu().numpy()
                translation=ret[3].float().cpu().numpy()+np.stack([i['observation'].center for i in valid])
                size=ret[4].float().cpu().numpy()
            if not all(np.isfinite(v).all() for v in (rotation,translation,size)):raise RuntimeError('Nonfinite model predictions; evaluation stopped.')
            pred=np.tile(np.eye(4,dtype=np.float32),(len(valid),1,1));pred[:,:3,:3]=rotation;pred[:,:3,3]=translation
            labels=np.array([i['record']['class_label'] for i in valid])
            gt=np.stack([i['gt_affine'] for i in valid]);gt_size=np.stack([i['gt_size'] for i in valid])
            syms=[]
            for i in valid:
                s=objmeta.instance_dict[i['record']['object_id']].tag.symmetry
                syms.append(dict(any=bool(s.any),x=s.x,y=s.y,z=s.z))
            part=dict(pred_affine=pred,pred_size=size,gt_affine=gt,gt_size=gt_size,labels=labels)
            payload=dict(gt_affine=gt,gt_size=gt_size,gt_sym_labels=syms,gt_class_labels=labels,pred_affine=pred,pred_size=size)
            pending.append((index,pool.submit(compute_criterion,payload),part,[i['record'] for i in valid]))
            if len(pending)>=args.metric_workers*2:collect(pending.pop(0))
            if index%20==0:print(f'batch={index} completed={len(evaluated)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        for entry in pending:collect(entry)
    if not chunks:raise ValueError('No valid instances were evaluated.')
    arrays={k:np.concatenate([p[k] for p in chunks]) for k in ('labels','iou','degrees','shift_cm')}
    metrics=summarize(arrays['labels'],arrays['iou'],arrays['degrees'],arrays['shift_cm'])
    metrics.update(is_full_test=source_is_full_test and args.shard_count==1,limit=args.limit,
                   excluded_count=rejected,metadata_excluded_count=len(excluded),protocol=args.protocol,split=args.split,
                   sample_manifest_sha256=sha256(out/'sample_manifest.jsonl'),elapsed_s=time.perf_counter()-start)
    write_json(out/'metrics.json',metrics)
    import csv
    with (out/'per_category.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['label','count','AUC25','AUC50','AUC75','VUS5_2','VUS5_5','VUS10_2','VUS10_5','rotation_deg','translation_cm'])
        for label,m in metrics['class_metrics'].items():writer.writerow([label,m['count'],*[100*v for v in m['iou_auc']],*[100*v for v in m['pose_auc']],m['deg_mean'],m['sht_mean']])
    write_json(out/'status.json',dict(status='complete',evaluated=len(evaluated),excluded=rejected,elapsed_s=time.perf_counter()-start))
    print(json.dumps(metrics['class_means']),flush=True)
