#!/usr/bin/env python3
"""Export a trusted local training checkpoint without optimizer/RNG/RADIO weights."""
import argparse
from pathlib import Path
import torch
import yaml
from f2g_pose.evaluate import sha256
from f2g_pose.io import write_json

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
p.add_argument('--trusted-local-checkpoint',action='store_true',required=True,help='Confirm the input pickle is a trusted local training artifact.')
a=p.parse_args()
src=Path(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
config=yaml.safe_load(Path(a.config).read_text())
checkpoint=torch.load(src,map_location='cpu',weights_only=False)
state=checkpoint.get('base_model',checkpoint.get('state_dict'))
if state is None:raise ValueError('Expected base_model or state_dict in checkpoint.')
state={k.removeprefix('module.'):v.detach().cpu() for k,v in state.items()}
state={k:v for k,v in state.items() if not k.startswith('base_model.radio_encoder.')}
source=dict(checkpoint=src.name,epoch=int(checkpoint.get('epoch',-1)),sha256=sha256(src))
target=out/'f2g-pose.pt'
torch.save(dict(format_version=1,state_dict=state,config=config,source=source,radio_included=False),target)
(out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(out/'SHA256SUMS').write_text(f'{sha256(target)}  {target.name}\n')
write_json(out/'provenance.json',dict(source=source,radio_included=False,validation='Run fixed-sample parity before publication.'))
print(target)
