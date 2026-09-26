"""Explicit frame/object manifests for SOPE and ROPE; invalid samples never retry."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import random
import numpy as np
from .preprocess import prepare


def image_files(prefix):
    prefix = Path(prefix)
    def field(name, suffix):
        parts = list(prefix.parts)
        if 'color' in parts: parts[parts.index('color')] = name
        return str(Path(*parts)) + suffix
    return dict(rgb=str(prefix)+'color.png', depth=field('depth','depth.exr'),
                mask=field('mask','mask.exr'), meta=field('meta','meta.json'))


@lru_cache(maxsize=8)
def load_frame(prefix):
    from cutoop.data_loader import Dataset
    files = image_files(prefix)
    return (Dataset.load_color(files['rgb']), Dataset.load_depth(files['depth']),
            Dataset.load_mask(files['mask']), Dataset.load_meta(files['meta']))


def frame_prefixes(root, split):
    from cutoop.data_loader import Dataset
    root=Path(root)
    pattern = str(root/'*') if split=='rope' else str(root/'*'/('train' if split=='train' else 'test'))
    return Dataset.glob_prefix(pattern)


def build_manifest(root, split, protocol='full', seed=0, limit=None):
    root=Path(root); frames=frame_prefixes(root,split)
    if protocol=='historical' and split=='sope': frames=frames[:5000]
    if protocol=='historical' and split=='rope':
        groups={}
        for frame in frames: groups.setdefault(str(Path(frame).parent),[]).append(frame)
        frames=[f for group in sorted(groups) for f in sorted(groups[group])[::50]]
    from cutoop.data_loader import Dataset
    records, excluded=[],[]
    rng=random.Random(seed)
    for frame in frames:
        rel=str(Path(frame).relative_to(root))
        try: meta=Dataset.load_meta(image_files(frame)['meta'])
        except (OSError,ValueError,KeyError) as exc:
            excluded.append(dict(frame=rel,reason='metadata: '+str(exc))); continue
        valid=[obj for obj in meta.objects if obj.is_valid]
        for obj in meta.objects:
            if not obj.is_valid:
                excluded.append(dict(frame=rel,object_id=str(obj.meta.oid),mask_id=int(obj.mask_id),reason='metadata is_valid=false'))
        if protocol=='historical' and split=='train' and valid:
            valid=[valid[i%len(valid)] if i < 8-8%len(valid) else rng.sample(valid,1)[0] for i in range(8)]
        if protocol=='historical' and split=='sope' and valid:
            valid=rng.sample(valid,1)
        for obj in valid:
            records.append(dict(frame=rel,object_id=str(obj.meta.oid),mask_id=int(obj.mask_id),
                                class_label=int(obj.meta.class_label),class_name=obj.meta.class_name))
        if limit and len(records)>=limit:
            records=records[:limit];break
    return records,excluded,len(frames)


class PoseDataset:
    def __init__(self, root, records, *, image_size=336, seed=0, train=False, shapes_root=None):
        self.root=Path(root);self.records=records;self.image_size=image_size
        self.seed=seed;self.train=train;self.shapes_root=Path(shapes_root) if shapes_root else None
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        record=self.records[index]
        try:
            rgb,depth,mask,meta=load_frame(str(self.root/record['frame']))
            obj=next(o for o in meta.objects if str(o.meta.oid)==record['object_id'] and o.mask_id==record['mask_id'])
            intr=meta.camera.intrinsics
            factor=rgb.shape[0]/intr.height
            if not np.isclose(rgb.shape[1]/intr.width,factor):raise ValueError('Dataset intrinsics/image aspect mismatch')
            K=np.array([[intr.fx*factor,0,intr.cx*factor],[0,intr.fy*factor,intr.cy*factor],[0,0,1]],np.float32)
            sid=f'{self.seed}:{record["frame"]}:{record["object_id"]}:{record["mask_id"]}'
            if self.train: sid+=f':{index}'
            seed=int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8],'little')
            if self.train: np.random.seed(seed % (2**32))
            obs=prepare(rgb,depth,K,(mask==obj.mask_id).astype(np.uint8),1.,seed=seed,
                        image_size=self.image_size,augment=self.train,
                        rng=np.random if self.train else None)
            from scipy.spatial.transform import Rotation
            q=obj.quaternion_wxyz
            R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix().astype(np.float32)
            t=np.array(obj.translation,np.float32)
            T=np.eye(4,dtype=np.float32);T[:3,:3]=R;T[:3,3]=t
            # Symmetry belongs to object metadata, provided by the benchmark metadata file.
            item=dict(record=record,observation=obs,gt_affine=T,gt_size=np.array(obj.meta.bbox_side_len,np.float32),K=K)
            if self.train:
                if self.shapes_root is None:raise ValueError('Training requires --shapes-root containing <oid>/Aligned.npy')
                canonical=np.load(self.shapes_root/record['object_id']/'Aligned.npy',allow_pickle=False)*obj.meta.scale
                item['complete']=(canonical@R.T+t-obs.center).astype(np.float32)
            return item
        except (ValueError,OSError,KeyError,StopIteration) as exc:
            return dict(record=record,error=f'{type(exc).__name__}: {exc}')


def collate_records(items):return items


def manifest_text(records):
    return ''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in records)
