import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.research.crystallization_paths.model import Forecaster,event_nll,cdf_from_logits,sampled_onset_cdf
from src.research.crystallization_paths.data import ResidentPaths,STATE_DIM
from src.research.crystallization_paths.metrics import coarse_logits,dense_brier,path_scores
from src.research.crystallization_paths.queue import select_context,tasks
from src.research.crystallization_transfer.refinement import variants


def spec(method):
    s=variants(dict(screen_epochs=6,encoder_lrs={'finetune':[],'scratch':[]},head_lrs=[]))[2]
    return dict(s,method=method,head_width=16,heads=4,depth=1,training_event_cdf=np.linspace(.001,.4,128).tolist())


def example(method,device='cpu'):
    torch.manual_seed(71);model=Forecaster(spec(method)).to(device)
    g=torch.randn(3,21,4,device=device);g[:,:,:3]*=5;g[:,::7,:3]=0
    g[:,:,3]=torch.tensor([-12.]*7+[-3.]*7+[0.]*7,device=device)
    observed=dict(features=torch.randn(3,21,128,device=device),geometry=g,condition=torch.randn(3,7,device=device))
    x,w=model.context.inputs(observed['features'],g);model.context.normalization.calibrate(x,w)
    event=torch.tensor([0,47,128],device=device)
    target=dict(state=torch.randn(3,32,STATE_DIM,device=device),event=event,
        occurred=(torch.arange(128,device=device)[None]>=event[:,None]).reshape(3,32,4).float())
    return model,observed,target


@pytest.mark.parametrize('method',['direct','ar_mse','ar_gaussian','mixture','diffusion'])
def test_finite_gradients_and_open_loop_forecasts(method):
    model,observed,target=example(method);model.train()
    loss=model.loss(observed,target,.5);assert loss.shape==(3,) and torch.isfinite(loss).all()
    loss.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();paths,cdf=model.forecast(observed,samples=3,diffusion_steps=4)
    assert paths.shape==(3,1 if method in ('direct','ar_mse') else 3,32,STATE_DIM)
    assert cdf.shape==(3,128) and torch.isfinite(paths).all()
    assert ((cdf>=0)&(cdf<=1)).all() and (cdf.diff(dim=-1)>=-1e-6).all()


def test_event_censoring_and_conversion_are_exact():
    logits=torch.full((3,32,4),-3.,dtype=torch.float64);events=torch.tensor([0,47,128])
    nll=event_nll(logits,events)
    p=torch.sigmoid(logits[0,0,0])
    expected=torch.tensor([-torch.log(p),-47*torch.log1p(-p)-torch.log(p),-128*torch.log1p(-p)])
    torch.testing.assert_close(nll.double(),expected,rtol=1e-6,atol=1e-6)
    cdf=cdf_from_logits(logits).numpy();lags=[1,4,12,32,64,128]
    recovered=cdf_from_logits_6(coarse_logits(cdf,lags))
    np.testing.assert_allclose(recovered,cdf[:,np.array(lags)-1],atol=1e-7)
    truth=np.arange(128)[None]>=events.numpy()[:,None]
    np.testing.assert_array_equal(dense_brier(truth.astype(float),events.numpy()),0)


def cdf_from_logits_6(x):
    return 1-np.cumprod(1/(1+np.exp(x)),axis=-1)


def test_absorbing_projection_does_not_allow_backwards_event_time():
    x=torch.full((1,2,32,4),-1.);x[0,0,2,1]=1
    cdf=sampled_onset_cdf(x)
    assert torch.count_nonzero(cdf[:,:9])==0
    torch.testing.assert_close(cdf[:,9:],torch.full_like(cdf[:,9:],.5))


def test_ar_teacher_forcing_is_causal_and_forbidden_at_evaluation():
    model,observed,target=example('ar_mse');model.train();context=model.encode(observed)
    changed=target['state'].clone();changed[:,10:]+=100
    first=model.recurrent(context,target['state'],1)[0];second=model.recurrent(context,changed,1)[0]
    torch.testing.assert_close(first[:,:11],second[:,:11],atol=0,rtol=0)
    assert not torch.allclose(first[:,11:],second[:,11:])
    model.eval()
    with pytest.raises(ValueError,match='forbidden'):model.recurrent(context,target['state'],1)


def test_ddim_oracle_recovers_entire_clean_path():
    model,observed,target=example('diffusion');model.eval()
    clean=torch.cat((target['state'],target['occurred']*2-1),-1).repeat_interleave(2,0)
    def oracle(x,t,context):
        a=model.alpha_bar[t,None,None]
        return (x-a.sqrt()*clean)/(1-a).sqrt()
    model.denoise=oracle
    paths,cdf=model.forecast(observed,samples=2,diffusion_steps=8)
    torch.testing.assert_close(paths,target['state'][:,None].expand_as(paths),atol=1e-4,rtol=1e-4)
    expected=(2*target['occurred'].flatten(1)+torch.tensor(model.spec['training_event_cdf']))/3
    torch.testing.assert_close(cdf,expected)


def test_state_forecast_scores_are_not_best_of_sample():
    target=torch.zeros(1,32,STATE_DIM);paths=torch.stack((target,target+2),1)
    score=path_scores(paths,target)
    torch.testing.assert_close(score[:,:,:4],torch.ones(1,32,4))
    torch.testing.assert_close(score[:,:,4:],torch.full((1,32,4),.5))


def test_timeline_input_and_target_separation():
    data=ResidentPaths.__new__(ResidentPaths);data.device=torch.device('cpu');data.spec={'radius_A':25}
    data.rows=torch.tensor([[0,64,1,400],[1,664,2,500]])
    data.offsets=torch.tensor([-16,-4,0]);data.future=torch.arange(1,33)*4
    data.features=torch.randn(2,167,16,7,128);data.geometry=torch.randn(2,167,16,7,3)
    data.states=torch.randn(2,199,16,STATE_DIM);data.onsets=torch.full((2,16),801);data.onsets[0,1]=65
    data.mean=torch.zeros(STATE_DIM);data.scale=torch.ones(STATE_DIM)
    observed=data.observed([0,1]);target=data.targets([0,1])
    torch.testing.assert_close(observed['features'][0,-7:],data.features[0,16,1])
    torch.testing.assert_close(target['state'][1,-1],data.states[1,198,2])
    assert target['event'].tolist()==[0,128]
    data.states+=1000;data.onsets[:]=50
    for key,value in observed.items():torch.testing.assert_close(data.observed([0,1])[key],value,atol=0,rtol=0)


def test_resident_producer_float_temperature_and_training_only_moments(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from src.research.crystallization_paths import data as module
    arrays={}
    for sid,value in [(0,0.),(1,1000.)]:
        arrays[sid]=dict(features=np.full((167*16*7,128),value,np.float32),
            mapping=np.arange(167*16*7).reshape(167,16,7),relative=np.zeros((167,16,7,3),np.float32),
            packet=np.full((16,801,128),value,np.float32),order=np.full((16,801,8),value,np.float32),
            labels=np.zeros((16,801),np.uint8),onset=np.full(16,801))
    corpus=SimpleNamespace(arrays=arrays,groups={0:np.array([0])},splits={'train':[0]},
        rows=[(0,0,0,400.0),(1,0,0,500.0)])
    monkeypatch.setattr(module,'Corpus',lambda plan:corpus)
    monkeypatch.setattr(module.np,'load',lambda path:np.zeros((199,16,128),np.float32))
    plan={'sources':[{'id':0},{'id':1}],'anchors':[64],'config':{'future_cache':str(tmp_path)}}
    resident=ResidentPaths(plan,{'history_ps':12,'radius_A':25},device='cpu')
    assert resident.rows.dtype==torch.long
    torch.testing.assert_close(resident.mean,torch.zeros(STATE_DIM),atol=0,rtol=0)
    assert resident.observed([0,1])['features'].shape==(2,21,128)
    assert resident.targets([0,1])['state'].shape==(2,32,STATE_DIM)


def test_context_choice_ignores_test_score_and_unfinished_runs(tmp_path):
    root=tmp_path/'technical/runs';root.mkdir(parents=True)
    for name,selection,test,state in [('a',.8,9.,'complete'),('b',.9,.1,'complete'),('c',.1,.01,'running')]:
        folder=root/name;folder.mkdir();s=spec('direct');s['name']=name
        (folder/'spec.json').write_text(json.dumps(s))
        (folder/'status.json').write_text(json.dumps(dict(state=state,best_selection_nll=selection,test_event_nll=test)))
    ref=select_context({'reference_output':str(tmp_path)})
    assert ref['source']=='a'
    queue=tasks({'epochs':[12,24]},ref)
    assert len(queue)==10 and {x['training']['epochs'] for x in queue}=={12,24}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA integration')
@pytest.mark.parametrize('method',['direct','ar_mse','ar_gaussian','mixture','diffusion'])
def test_cuda_update_and_checkpoint_roundtrip(method,tmp_path):
    model,observed,target=example(method,'cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
    model.train();loss=model.loss(observed,target,.5).mean();loss.backward();optimizer.step()
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    restored=Forecaster(model.spec).cuda();restored.load_state_dict(torch.load(path,weights_only=True));model.eval();restored.eval()
    torch.manual_seed(6);a=model.forecast(observed,2,4)
    torch.manual_seed(6);b=restored.forecast(observed,2,4)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
