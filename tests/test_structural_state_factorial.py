import json

import numpy as np
import pytest
import torch

from src.research.local_predictability.metrics import source_weights, hazard_loss, weighted_scores
from src.research.structural_state.factorial_report import onset_scores, mechanism_effects, collect


def test_onset_replay_includes_censoring_ties_and_natural_source_weights(tmp_path):
    indices=np.arange(6)
    source=np.array([1,1,1,2,2,3])
    event=np.array([0,5,3,5,4,5])
    logits=np.array([[20,-20,1,2,3],[-20,-20,-20,-20,-20],[0,0,0,0,0],
                     [0,0,0,0,0],[-4,-3,-2,-1,0],[1,1,1,1,1]],dtype=np.float64)
    risks=1-np.cumprod(1-1/(1+np.exp(-logits)),axis=1)
    weights=source_weights(source)
    nll=hazard_loss(torch.from_numpy(logits),torch.from_numpy(event)).numpy()
    metrics=dict(nll=float(weights@nll),horizons={'12.0':weighted_scores(event<5,risks[:,-1],source)})
    path=tmp_path/'predictions.npz'
    np.savez(path,indices=indices,source=source,event=event,risks=risks,logits=logits)
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    result=onset_scores(path,indices,source,event,np.vstack([weights,weights]))
    np.testing.assert_array_equal(result['nll'],[weights@nll]*2)
    with pytest.raises(AssertionError):onset_scores(path,indices[::-1],source,event,weights[None])
    metrics['nll']+=1
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(AssertionError):onset_scores(path,indices,source,event,weights[None])


def test_mechanism_rule_rejects_forecast_gain_that_loses_present_information():
    physical=[];neighbors=[];heads=[];onset={}
    for arm,factor in [('reference',1.),('candidate',.9)]:
        for family in ('relaxed_radial','relaxed_angular','relaxed_l6','current_order','future_order_12'):
            physical.append(dict(arm=arm,representation='exported',target=family,readout='ridge',
                                 population='PTM_other',mse=factor))
        for family in ('relaxed_angular','relaxed_l6'):
            neighbors.append(dict(arm=arm,representation='exported',target=family,population='noncrystalline',mse=factor))
        for block in ('radial','l2','l4'):
            heads.append(dict(arm=arm,checkpoint='last',block=block,population='PTM_other',mse=factor))
        onset[arm,'exported','mlp']={'average_precision':np.array([1-factor]),'brier':np.array([factor])}
    contrasts=[['test','reference','candidate']]
    row=mechanism_effects(1,contrasts,physical,neighbors,heads,onset)[0]
    assert row['future_rule_pass'] and row['neighbor_rule_pass']
    heads[-1]['mse']=1.03
    row=mechanism_effects(1,contrasts,physical,neighbors,heads,onset)[0]
    assert row['future12_error_change_percent']<0 and row['onset_ap_delta']>0
    assert not row['retention_pass'] and not row['future_rule_pass'] and not row['neighbor_rule_pass']


def test_pending_campaign_exports_honest_status_and_documentation(tmp_path):
    config=dict(seeds=[dict(seed=1,output=str(tmp_path/'not-started'))])
    assert collect(config,tmp_path) is False
    status=json.loads((tmp_path/'technical/status.json').read_text())
    assert status['state']=='partial' and status['completed_seeds']==[] and status['pending_seeds']==[1]
    assert (tmp_path/'tables/METRICS.md').exists()


def test_initial_audit_accepts_reduction_noise_but_rejects_changed_encoder(tmp_path):
    from src.research.structural_state.factorial_report import initial_audit
    saved=dict(encoder_config={'width':2},auxiliary_targets={'lag_ps':9},
        calibration={'current_order':{'ridge':1}},model={
            'encoder.weight':torch.ones(2,2),'encoder.pooled_mean':torch.zeros(2),
            'encoder.pooled_scale':torch.ones(2),'heads.current_order.weight':torch.eye(2),
            'heads.current_order.bias':torch.zeros(2)})
    reference=tmp_path/'reference.pt';candidate=tmp_path/'candidate.pt'
    torch.save(saved,reference);torch.save(saved,candidate)
    z=np.ones((128,2),dtype=np.float32);noisy=z.copy();noisy[0,0]+=5e-5
    assert initial_audit(reference,candidate,z,noisy)['export_max_abs']>0
    saved['model']['encoder.weight'][0,0]+=1e-6
    torch.save(saved,candidate)
    with pytest.raises(AssertionError):initial_audit(reference,candidate,z,noisy)
    saved['model']['encoder.weight'].fill_(1);torch.save(saved,candidate)
    with pytest.raises(ValueError,match='CUDA noise'):initial_audit(reference,candidate,z,z+.001)
    saved['model']['heads.current_order.weight'][0,0]+=.01;torch.save(saved,candidate)
    with pytest.raises(ValueError,match='head predictions'):initial_audit(reference,candidate,z,noisy)
