import importlib.util
import json
from pathlib import Path
import hashlib
import numpy as np
import pytest
from f2g_pose.data import manifest_text

spec=importlib.util.spec_from_file_location('merge_evaluations',Path(__file__).parents[1]/'scripts/merge_evaluations.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(tmp):
    records=[dict(frame=f'{i}/',object_id=str(i),mask_id=1,class_label=i%2,class_name=str(i%2)) for i in range(4)]
    source=tmp/'source.jsonl';source.write_text(manifest_text(records))
    digest=hashlib.sha256(manifest_text(records).encode()).hexdigest()
    (tmp/'run.json').write_text(json.dumps(dict(protocol='full',limit=None,split='sope',candidate_count=4,candidate_manifest_sha256=digest)))
    dirs=[]
    for shard in range(2):
        out=tmp/f'shard{shard}';out.mkdir();(out/'parts').mkdir();dirs.append(out)
        candidates=records[2*shard:2*shard+2]
        good=candidates if shard==0 else candidates[:1]
        bad=[] if shard==0 else [dict(candidates[1],reason='insufficient valid depth')]
        for name,rows in [('candidate_manifest',candidates),('sample_manifest',good),('excluded',bad),('metadata_excluded',[])]:
            (out/f'{name}.jsonl').write_text(manifest_text(rows))
        run=dict(checkpoint_sha256='same',object_meta_sha256='same',source_manifest_sha256=digest,source_candidate_count=4,split='sope',protocol='full',seed=0,shard_count=2,shard_index=shard,shard_begin=shard*2,shard_end=shard*2+2,model_config_sha256=None,implementation_sha256={},limit=None)
        (out/'run.json').write_text(json.dumps(run));(out/'status.json').write_text('{"status":"complete"}')
        n=len(good);np.savez(out/'parts/000000.npz',labels=[r['class_label'] for r in good],iou=np.full(n,.7),degrees=np.full(n,2.),shift_cm=np.full(n,1.))
    return dirs,source


def test_merge_exact_coverage_and_exclusions(tmp_path):
    dirs,source=fixture(tmp_path)
    metrics=module.merge(dirs,source,tmp_path/'merged')
    assert metrics['excluded_count']==1
    assert metrics['is_full_test']
    assert sum(m['count'] for m in metrics['class_metrics'].values())==3
    state=json.loads((tmp_path/'merged/status.json').read_text())
    assert state['coverage_verified'] and state['candidate_count']==4


def test_merge_rejects_missing_instances(tmp_path):
    dirs,source=fixture(tmp_path)
    (dirs[1]/'excluded.jsonl').write_text('')
    with pytest.raises(ValueError,match='coverage'):module.merge(dirs,source,tmp_path/'merged')
    assert not (tmp_path/'merged').exists()


def test_custom_subset_cannot_claim_full_test(tmp_path):
    dirs,source=fixture(tmp_path)
    (tmp_path/'run.json').unlink()
    result=module.merge(dirs,source,tmp_path/'merged')
    assert result['is_full_test'] is False
