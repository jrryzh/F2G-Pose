#!/usr/bin/env python3
"""Merge complete evaluation shards, validating exact coverage and provenance."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from f2g_pose.data import manifest_text
from f2g_pose.evaluate import sha256,manifest_is_full,environment
from f2g_pose.io import write_json
from f2g_pose.metrics import summarize


def merge(shards,expected_manifest,output):
    directories=[Path(p).resolve() for p in shards]
    entries=[(json.loads((d/'run.json').read_text()),d) for d in directories]
    entries.sort(key=lambda e:e[0]['shard_index'])
    reference=entries[0][0]
    expected=[json.loads(line) for line in Path(expected_manifest).read_text().splitlines()]
    digest=hashlib.sha256(manifest_text(expected).encode()).hexdigest()
    if len(entries)!=reference['shard_count']:raise ValueError('Missing evaluation shards.')
    if [e[0]['shard_index'] for e in entries]!=list(range(len(entries))):raise ValueError('Duplicate or missing shard indices.')
    fields=('checkpoint_sha256','object_meta_sha256','source_manifest_sha256','source_candidate_count','split','protocol','seed','shard_count','model_config_sha256','implementation_sha256')
    all_candidates=[];evaluated=[];excluded=[];arrays=[];part_sources=[]
    for run,d in entries:
        if any(run[k]!=reference[k] for k in fields):raise ValueError(f'Incompatible shard provenance: {d}')
        if run.get('precision','fp32')!=reference.get('precision','fp32'):raise ValueError('Mixed precision settings between shards.')
        if json.loads((d/'status.json').read_text())['status']!='complete':raise ValueError(f'Incomplete shard: {d}')
        candidates=[json.loads(l) for l in (d/'candidate_manifest.jsonl').read_text().splitlines()]
        if candidates!=expected[run['shard_begin']:run['shard_end']]:raise ValueError('Shard candidates do not match source manifest slice.')
        good=[json.loads(l) for l in (d/'sample_manifest.jsonl').read_text().splitlines()]
        bad=[json.loads(l) for l in (d/'excluded.jsonl').read_text().splitlines()]
        encode=lambda rows:Counter(json.dumps(r,sort_keys=True) for r in rows)
        clean_bad=[{k:v for k,v in r.items() if k!='reason'} for r in bad]
        if encode(good+clean_bad)!=encode(candidates):raise ValueError('Candidate coverage contains omissions or duplicates.')
        all_candidates.extend(candidates);evaluated.extend(good);excluded.extend(bad)
        count=0
        for p in sorted((d/'parts').glob('*.npz')):
            with np.load(p,allow_pickle=False) as data:
                a={k:data[k] for k in ('labels','iou','degrees','shift_cm')}
            count+=len(a['labels']);arrays.append(a);part_sources.append(str(p))
        if count!=len(good):raise ValueError('Prediction/manifest counts differ.')
    if all_candidates!=expected or reference['source_manifest_sha256']!=digest:raise ValueError('Combined source manifest mismatch.')
    values={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    metrics=summarize(values['labels'],values['iou'],values['degrees'],values['shift_cm'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'candidate_manifest.jsonl').write_text(manifest_text(expected))
    (out/'sample_manifest.jsonl').write_text(manifest_text(evaluated))
    (out/'excluded.jsonl').write_text(manifest_text(excluded))
    metadata=(entries[0][1]/'metadata_excluded.jsonl').read_text()
    (out/'metadata_excluded.jsonl').write_text(metadata)
    full_source=manifest_is_full(expected_manifest,len(expected),digest,reference['split'])
    metrics.update(split=reference['split'],protocol=reference['protocol'],is_full_test=reference['protocol']=='full' and reference['limit'] is None and full_source,
                   candidate_count=len(expected),excluded_count=len(excluded),metadata_excluded_count=len(metadata.splitlines()),
                   sample_manifest_sha256=sha256(out/'sample_manifest.jsonl'),source_manifest_sha256=digest)
    write_json(out/'metrics.json',metrics)
    write_json(out/'run.json',dict(shards=[str(d) for _,d in entries],source_runs_sha256={str(d):sha256(d/'run.json') for _,d in entries},
                                 expected_manifest_sha256=digest,reference=reference,merge_environment=environment()))
    write_json(out/'prediction_parts.json',part_sources)
    with (out/'per_category.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['label','count','AUC25','AUC50','AUC75','VUS5_2','VUS5_5','VUS10_2','VUS10_5','rotation_deg','translation_cm'])
        for label,m in metrics['class_metrics'].items():writer.writerow([label,m['count'],*[100*v for v in m['iou_auc']],*[100*v for v in m['pose_auc']],m['deg_mean'],m['sht_mean']])
    write_json(out/'status.json',dict(status='complete',evaluated=len(evaluated),excluded=len(excluded),candidate_count=len(expected),shards=len(entries),coverage_verified=True))
    print(json.dumps(metrics['class_means']))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards',nargs='+',required=True);p.add_argument('--expected-manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();merge(a.shards,a.expected_manifest,a.output)
