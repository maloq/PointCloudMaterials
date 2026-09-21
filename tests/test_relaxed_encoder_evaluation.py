import numpy as np
import json
import time
from types import SimpleNamespace
import pytest
from sklearn.metrics import average_precision_score

from src.research.relaxed_encoder.availability import requested_assay_frames, matched_assay_mask
from src.research.relaxed_encoder.evaluation import population_audit
from src.research.relaxed_encoder.evaluation_metrics import weighted_ap, source_draws, bootstrap_values
from src.research.crystallization_information.runtime import score_hazard
from src.research.relaxed_encoder.selection import selected_runs


def test_role_grids_do_not_count_intentionally_unused_frames_as_timeouts():
    plan=dict(config=dict(frames=[0,1,2],frames_by_role=dict(train=[0,2],test=[0,1,2])),
              sources=[dict(id=1,split='train'),dict(id=2,split='test')])
    requested=requested_assay_frames(plan)
    assert matched_assay_mask(np.array([1,1,2,2]),np.array([0,1,0,1]),requested).tolist()==[True,False,True,True]


def test_event_audit_deduplicates_local_onsets_but_retains_source_count():
    pop=dict(source=np.array([1,1,1,2]),role=np.array(['test']*4),
             rows=np.array([[1,0,3],[1,1,3],[1,0,4],[2,0,3]]))
    result=population_audit(pop,np.array([64,68,64,64]),np.array([80,80,78,100]))['test']
    assert result==dict(windows=4,sources=2,positive_windows_12ps=3,
                        distinct_local_onsets_12ps=2,event_sources_12ps=1)


def test_weighted_ap_ties_and_zero_weight_sources_match_sklearn():
    y=np.array([1,0,1,0,0,1],bool);score=np.array([.9,.9,.3,.2,.2,.1])
    for w in [np.ones(6),np.array([0,0,3,2,2,1.]),np.array([1,4,2,6,1,3.])]:
        np.testing.assert_allclose(weighted_ap(y,score,w),average_precision_score(y,score,sample_weight=w))


def test_source_bootstrap_preserves_strata_and_point_metrics():
    sources=np.repeat([1,2,3,4],3);temps=np.repeat([400,400,500,500],3)
    counts,inverse=source_draws(sources,temps,100,19)
    np.testing.assert_array_equal(counts[:,:2].sum(1),2)
    np.testing.assert_array_equal(counts[:,2:].sum(1),2)
    assert np.all(inverse[:3]==inverse[0])
    rng=np.random.default_rng(17);logits=rng.normal(size=(12,5)).astype(np.float32)
    pop=dict(source=sources,temperature=temps,event=np.array([0,5,4]*4),
             delay=np.array([.75,20,11]*4,dtype=np.float32))
    ids=np.arange(6,12);cal=np.arange(6)
    m=score_hazard(pop,ids,logits[ids],cal,logits[cal])
    ones,ix=source_draws(sources[ids],temps[ids],1,1);ones[:]=1
    v=bootstrap_values(pop,ids,logits[ids],m,ones,ix)
    np.testing.assert_allclose(v['event_nll'][0],m['event_nll'])
    np.testing.assert_allclose(v['ap_12ps'][0],m['classification']['12.0']['average_precision'])
    np.testing.assert_allclose(v['recall_12ps'][0],m['classification']['12.0']['recall'])
    if m['timing']['12.0']['detected_timing_mae_ps'] is not None:
        np.testing.assert_allclose(v['detected_timing_mae_ps'][0],m['timing']['12.0']['detected_timing_mae_ps'])


def test_stopped_encoder_cannot_resume_and_evaluation_does_not_wait_for_it(tmp_path, monkeypatch):
    from src.research.relaxed_encoder import queue, report
    root=tmp_path/'technical';(root/'queue').mkdir(parents=True);(root/'assay').mkdir()
    (root/'assay/ready.json').write_text('{}')
    plan=dict(identity='release',config=dict(output=str(tmp_path),runs=[dict(name='done'),dict(name='stopped')]))
    record=dict(plan_identity='release',excluded=dict(stopped=dict(reason='User stopped training')))
    (root/'evaluation-exclusions.json').write_text(json.dumps(record))
    (root/'queue/fit-done.json').write_text(json.dumps(dict(state='complete')))
    (root/'queue/fit-stopped.json').write_text(json.dumps(dict(state='checkpointed')))
    with pytest.raises(ValueError,match='explicitly stopped'):
        queue.execute(plan,'stopped','fit')
    calls=[]
    def run(command,**kwargs):
        name=command[command.index('--name')+1];phase=command[command.index('--phase')+1]
        assert phase!='fit' and name!='stopped'
        calls.append((phase,name));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue,'deadline_for_job',lambda:time.time()+1000)
    monkeypatch.setattr(queue.subprocess,'run',run)
    monkeypatch.setattr(queue,'fatal_failures',lambda p:[])
    monkeypatch.setattr(report,'report',lambda p:None)
    queue.worker(plan,'evaluation-test',tmp_path/'config.json',evaluation_only=True)
    assert ('extract','done') in calls and ('probe','done') in calls
    assert ('probe','geometry_cold') in calls
    assert json.loads((root/'gpu-evaluation-test.json').read_text())['state']=='finished'
    assert json.loads((root/'queue/fit-stopped.json').read_text())['state']=='checkpointed'


def test_exclusions_are_bound_to_the_frozen_plan(tmp_path):
    root=tmp_path/'technical';root.mkdir()
    plan=dict(identity='release',config=dict(output=str(tmp_path),runs=[dict(name='done')]))
    receipt=root/'evaluation-exclusions.json'
    receipt.write_text(json.dumps(dict(plan_identity='different',excluded={})))
    with pytest.raises(ValueError,match='another plan'):selected_runs(plan)
    receipt.write_text(json.dumps(dict(plan_identity='release',excluded=dict(typo=dict(reason='Stop')))))
    with pytest.raises(ValueError,match='Unknown excluded'):selected_runs(plan)
