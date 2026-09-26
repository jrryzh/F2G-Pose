"""SOPE training with explicit data, frozen RADIO, AMP and resumable checkpoints."""
from pathlib import Path
import math
import os
import random
import numpy as np
from .io import write_json
from .data import build_manifest,PoseDataset,collate_records,manifest_text


def train(args):
    import torch
    import yaml
    from easydict import EasyDict
    from cutoop.obj_meta import ObjectMetaData
    from .model import F2GPose
    from .estimator import PoseEstimator
    distributed=int(os.environ.get('WORLD_SIZE','1'))>1
    rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if args.batch_size<2:raise ValueError('Training BatchNorm requires per-GPU batch size >=2.')
    if args.accumulation<1:raise ValueError('Accumulation must be positive.')
    device=torch.device(f'cuda:{local_rank}' if distributed else args.device)
    torch.cuda.set_device(device)
    if distributed:torch.distributed.init_process_group(backend='nccl')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    config=yaml.safe_load(Path(args.config).read_text())
    out=Path(args.output)
    if rank==0:out.mkdir(parents=True,exist_ok=bool(args.resume))
    if distributed:torch.distributed.barrier()
    if args.checkpoint:
        model=PoseEstimator(args.checkpoint,config=config,radio_repository=args.radio_repository,
                            radio_checkpoint=args.radio_checkpoint,device=str(device)).model
    else:
        mc=EasyDict(config['model']);mc.radio_repository=args.radio_repository;mc.radio_checkpoint=args.radio_checkpoint
        model=F2GPose(mc).to(device)
    model.base_model.radio_encoder.requires_grad_(False)
    if distributed:model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),**config['optimizer']['kwargs'])
    scaler=torch.amp.GradScaler('cuda')
    start_epoch=0;start_batch=0;step=0
    if args.resume:
        # Local training checkpoints contain Python/NumPy RNG state. Never load untrusted files.
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        result=model.load_state_dict(saved['state_dict'],strict=False)
        if result.unexpected_keys or any(not k.startswith('base_model.radio_encoder.') for k in result.missing_keys):raise ValueError('Resume state is incompatible.')
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch'];start_batch=saved['next_batch'];step=saved['global_step']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if 'python_rng' in saved:random.setstate(saved['python_rng'])
        if 'numpy_rng' in saved:
            rng=saved['numpy_rng']
            np.random.set_state((rng[0],np.asarray(rng[1],dtype=np.uint32),rng[2],rng[3],rng[4]))
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True) if distributed else model
    if rank==0 and not (out/'train_manifest.jsonl').exists():
        rows,excluded,_=build_manifest(args.data_root,'train',protocol='historical',seed=args.seed,limit=args.limit)
        (out/'train_manifest.jsonl').write_text(manifest_text(rows))
        (out/'excluded_metadata.jsonl').write_text(manifest_text(excluded))
    if distributed:torch.distributed.barrier()
    import json
    rows=[json.loads(line) for line in (out/'train_manifest.jsonl').read_text().splitlines()]
    if len(rows)<args.batch_size:raise ValueError('Manifest is smaller than training batch size.')
    meta_path=args.object_meta or str(Path(args.data_root).parent/'Meta/obj_meta.json')
    metadata=ObjectMetaData.load_json(meta_path)
    dataset=PoseDataset(args.data_root,rows,train=True,image_size=config['image_size'],seed=args.seed,shapes_root=args.shapes_root)
    epochs=args.epochs or config['max_epoch']
    def save(epoch,next_batch):
        if rank!=0:return
        state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('base_model.radio_encoder.')}
        numpy_rng=np.random.get_state()
        payload=dict(state_dict=state,config=config,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
                     next_batch=next_batch,global_step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                     python_rng=random.getstate(),numpy_rng=(numpy_rng[0],numpy_rng[1].tolist(),numpy_rng[2],numpy_rng[3],numpy_rng[4]))
        target=out/'checkpoint-last.pt';tmp=out/'checkpoint-last.tmp';torch.save(payload,tmp);tmp.replace(target)
    if rank==0:write_json(out/'run.json',dict(arguments=vars(args),config=config,world_size=int(os.environ.get('WORLD_SIZE','1')),
                                             effective_batch=args.batch_size*args.accumulation*int(os.environ.get('WORLD_SIZE','1')),
                                             training_protocol='historical losses, DZI/mask augmentation; 8 object draws per SOPE frame; explicit manifest'))
    stop=False
    for epoch in range(start_epoch,epochs):
        dataset.seed=args.seed+epoch
        sampler=torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True,seed=args.seed) if distributed else None
        if sampler:sampler.set_epoch(epoch)
        generator=torch.Generator().manual_seed(args.seed+epoch)
        loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,shuffle=sampler is None,
                                           sampler=sampler,generator=generator,collate_fn=collate_records,drop_last=True)
        schedule=config['scheduler']['kwargs'];lr=config['optimizer']['kwargs']['lr']*max(schedule['lr_decay']**(epoch//schedule['decay_step']),schedule['lowest_decay'])
        for group in optimizer.param_groups:group['lr']=lr
        bnc=config['bnmscheduler']['kwargs'];momentum=max(bnc['bn_momentum']*bnc['bn_decay']**(epoch//bnc['decay_step']),bnc['lowest_decay'])
        model.train();model.base_model.radio_encoder.eval()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.momentum=momentum
        optimizer.zero_grad(set_to_none=True)
        for index,items in enumerate(loader):
            if epoch==start_epoch and index<start_batch:continue
            invalid=[x for x in items if 'error' in x]
            if invalid:
                # Fail explicitly instead of replacing a sample or letting DDP ranks diverge.
                raise ValueError(f'Invalid training sample: {invalid[0]}')
            inputs=[torch.from_numpy(np.stack([getattr(i['observation'],key) for i in items])).to(device) for key in ('points','rgb','rows','columns')]
            complete=torch.from_numpy(np.stack([i['complete'] for i in items])).to(device)
            gt_R=torch.from_numpy(np.stack([i['gt_affine'][:3,:3] for i in items])).to(device)
            gt_6d=torch.cat([gt_R[:,:,0],gt_R[:,:,1]],dim=1)
            gt_t=torch.from_numpy(np.stack([i['gt_affine'][:3,3]-i['observation'].center for i in items])).to(device)
            gt_s=torch.from_numpy(np.stack([i['gt_size'] for i in items])).to(device)
            symcodes={'none':0,'any':1,'half':2,'quarter':3}
            syms={axis:torch.tensor([symcodes[getattr(metadata.instance_dict[i['record']['object_id']].tag.symmetry,axis)] for i in items],device=device) for axis in 'xyz'}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ret=wrapped(*inputs)
                losses=model.get_loss(ret,complete,gt_6d,gt_t,gt_s,syms,epoch=epoch,step=step)
                group_start=(index//args.accumulation)*args.accumulation
                group_size=min(args.accumulation,len(loader)-group_start)
                loss=sum(losses)/group_size
            if not torch.isfinite(loss) or any(not torch.isfinite(component) for component in losses):
                raise RuntimeError('Nonfinite training loss.')
            scaler.scale(loss).backward()
            update=(index+1)%args.accumulation==0 or index+1==len(loader)
            if update:
                scaler.unscale_(optimizer)
                fusion_grad=sum(p.grad.float().abs().sum().item() for p in model.base_model.radio_fusion_net.parameters() if p.grad is not None)
                if fusion_grad==0 or not math.isfinite(fusion_grad):raise RuntimeError('Fusion gradient missing or nonfinite.')
                if any(p.grad is not None for p in model.base_model.radio_encoder.parameters()):raise RuntimeError('RADIO unexpectedly received gradients.')
                fusion_before=[p.detach().clone() for p in model.base_model.radio_fusion_net.parameters()] if args.max_steps else None
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(grad_norm):raise RuntimeError('Nonfinite training gradient.')
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);step+=1
                fusion_delta=(sum((p.detach()-old).float().abs().sum().item() for p,old in zip(model.base_model.radio_fusion_net.parameters(),fusion_before))
                              if fusion_before is not None else None)
                if fusion_delta is not None and (fusion_delta<=0 or not math.isfinite(fusion_delta)):
                    raise RuntimeError('Fusion parameters did not update or became nonfinite.')
                if rank==0:
                    entry=dict(epoch=epoch,batch=index,global_step=step,losses=[float(l.detach()) for l in losses],fusion_gradient_l1=fusion_grad,
                               fusion_parameter_delta_l1=fusion_delta,gradient_norm=float(grad_norm),lr=lr)
                    with (out/'training.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                    print(entry,flush=True)
                if args.max_steps and step>=args.max_steps:
                    save(epoch,index+1);stop=True;break
        if stop:break
        save(epoch+1,0)
    if rank==0:write_json(out/'status.json',dict(status='complete',global_step=step,epoch=epoch,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
    if distributed:torch.distributed.destroy_process_group()
