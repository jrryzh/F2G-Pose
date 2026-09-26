#!/usr/bin/env python3
"""Create a complete inference example from a locally licensed Omni6DPose sample."""
import argparse
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
from pathlib import Path
import numpy as np
from PIL import Image
from f2g_pose.data import build_manifest,PoseDataset,load_frame
from f2g_pose.io import write_json
p=argparse.ArgumentParser()
p.add_argument('--data-root',required=True);p.add_argument('--split',choices=['sope','rope'],default='sope');p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
rows,_,_=build_manifest(a.data_root,a.split,'historical',limit=32)
dataset=PoseDataset(a.data_root,rows)
for i,row in enumerate(rows):
 item=dataset[i]
 if 'error' not in item:break
else:raise ValueError('No valid sample in first 32 candidates.')
rgb,depth,mask,_=load_frame(str(Path(a.data_root)/row['frame']))
Image.fromarray(rgb).save(out/'rgb.png');Image.fromarray((mask==row['mask_id']).astype(np.uint8)*255).save(out/'mask.png')
np.save(out/'depth.npy',depth.astype(np.float32));np.savetxt(out/'K.txt',item['K'])
write_json(out/'input.json',dict(sample=row,depth_scale=1.,channel_order='RGB',alignment='dataset RGB/depth/mask aligned',mask_source='dataset annotation'))
print(f'Example written to {out}; use --depth-scale 1.')
