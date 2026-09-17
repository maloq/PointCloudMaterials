import numpy as np
import torch
from src.research.local_predictability.metrics import hazard_loss,cumulative_risk,source_weights,threshold_at_fpr,weighted_scores,alarm_episodes
from src.research.local_predictability.baselines import history_features


def test_hazard_likelihood_and_cumulative_probabilities():
    probability=torch.tensor([[.2,.3,.4]]).repeat(4,1)
    logits=torch.logit(probability)
    actual=hazard_loss(logits,torch.tensor([0,1,2,3]))
    expected=-torch.log(torch.tensor([.2,.8*.3,.8*.7*.4,.8*.7*.6]))
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(cumulative_risk(logits)[0],torch.tensor([.2,.44,.664]))


def test_source_weighting_ignores_duplicate_windows_and_threshold_includes_ties():
    source=np.array([0,1,1,1]);w=source_weights(source)
    assert w[0]==.5 and np.isclose(w[1:].sum(),.5)
    actual=np.array([0,0,0,1],bool);score=np.array([.5,.5,.2,.9],np.float32)
    threshold=threshold_at_fpr(actual,score,source,.05)
    result=weighted_scores(actual,score,source,threshold)
    assert result['false_positive_rate']==0
    assert result['recall']==1


def test_history_is_causal_and_repeated_control_keeps_current():
    packet=np.arange(2*100*3,dtype=np.float32).reshape(2,100,3)
    anchors=np.array([30,40]);original=history_features(packet,anchors,12)
    changed=packet.copy();changed[:,41:]=1e10
    np.testing.assert_array_equal(original,history_features(changed,anchors,12))
    repeated=history_features(packet,anchors,12,repeat=True)
    np.testing.assert_array_equal(repeated[:,:,:3],packet[:,anchors])
    np.testing.assert_allclose(repeated[:,:,9:15],0,atol=1e-6)


def test_alarm_collapse_and_missed_event_kept():
    anchors=np.arange(20);risk=np.zeros(20);risk[1:4]=1;risk[15:]=1
    result=alarm_episodes(anchors,risk,.5,19,5)
    assert result['alarms']==[1,15] and result['false_alarms']==1
    assert result['detected'] and result['lead_ps']==3
    missed=alarm_episodes(anchors,np.zeros(20),.5,19,5)
    assert not missed['detected'] and missed['lead_ps'] is None


def test_small_hazard_fit_exports_real_predictions(tmp_path):
    import json,os
    from pathlib import Path
    from src.research.local_predictability.baselines import hazard_fit,score_hazard
    rng=np.random.default_rng(5);n=120
    arrays=dict(x=rng.normal(size=(n,4)).astype(np.float32),y=np.tile([0,2,6],40),
        split=np.repeat(['train','selection','calibration','test'],30),source=np.repeat(np.arange(12),10),
        temperature=np.full(n,400),center=np.arange(n),anchor=np.full(n,100))
    plan=json.loads(Path('configs/local_predictability/two_gpu_16h.json').read_text())
    plan['descriptors']['linear_hazard']['regularization_grid']=[1]
    plan['descriptors']['mlp_hazard'].update(maximum_updates=10,validation_every=5,batch_size=16)
    config=dict(seed=20260919,training_deadline_utc='2099-01-01T00:00:00+00:00')
    for kind in ['linear','mlp']:
        probability,logits,masks=hazard_fit(arrays,kind,plan,torch.device(os.environ.get('PCM_TEST_DEVICE','cpu')),config,tmp_path/kind)
        assert probability.shape==(n,6) and np.isfinite(probability).all()
        assert (np.diff(probability,axis=1)>=-1e-6).all()
        scores=score_hazard(arrays,probability,logits,masks,plan)
        assert len(scores['population'])==18
        assert (tmp_path/kind/'model.pt').exists()
