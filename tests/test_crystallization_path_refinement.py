import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.research.crystallization_paths.refinement import screens,expanded_tasks
from src.research.crystallization_paths.refined_model import RefinedForecaster,project_onset,clean_and_noise
from src.research.crystallization_paths.runtime import teacher_probability
from src.research.crystallization_paths.data import STATE_DIM,ResidentPaths
from src.research.crystallization_transfer.refinement import variants


def recipe():
    ref=variants(dict(screen_epochs=6,encoder_lrs={'finetune':[],'scratch':[]},head_lrs=[]))[2]
    config={'refinement':{'screen_epochs':12,'long_epochs':36}}
    return config,{'spec':ref}


def example(method,**extra):
    config,reference=recipe();s=next(x for x in screens(config,reference) if x['method']==method)
    s.update(head_width=16,heads=4,**extra);s['training_event_cdf']=np.linspace(.001,.4,128).tolist()
    model=RefinedForecaster(s)
    g=torch.zeros(3,21,4);g[:,:,3]=torch.tensor([-12.]*7+[-3.]*7+[0.]*7)
    observed=dict(features=torch.randn(3,21,128),geometry=g,condition=torch.randn(3,7))
    if s['motion_input']:observed['motion']=torch.randn(3,43)
    x,w=model.context.inputs(observed['features'],g);model.context.normalization.calibrate(x,w)
    event=torch.tensor([0,47,128])
    target=dict(state=torch.randn(3,32,STATE_DIM),present=torch.randn(3,STATE_DIM),event=event,
        occurred=(torch.arange(128)[None]>=event[:,None]).reshape(3,32,4).float())
    return model,observed,target


@pytest.mark.parametrize('method,extra',[
    ('direct',{}),('direct',{'residual_anchor':True,'present_weight':.25}),('direct',{'motion_input':True}),
    ('ar_mse',{'residual_anchor':True}),('ar_gaussian',{}),('ar_gaussian',{'gaussian_rank':8}),
    ('mixture',{}),('mixture',{'mixture_style':'stratified'}),
    ('mixture',{'mixture_style':'stratified','mixture_boundaries':[128]}),
    ('diffusion',{}),('diffusion',{'diffusion_prediction':'x0'})])
def test_gradients_and_free_rollout(method,extra):
    torch.manual_seed(90);model,observed,target=example(method,**extra);model.train()
    loss=model.loss(observed,target,1. if method=='ar_gaussian' else .5)
    assert loss.shape==(3,) and torch.isfinite(loss).all()
    loss.mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();path,cdf=model.forecast(observed,3,4)
    assert path.shape==(3,1 if method in ('direct','ar_mse') else 3,32,STATE_DIM)
    assert torch.isfinite(path).all() and ((cdf>=0)&(cdf<=1)).all() and (cdf.diff(dim=-1)>=-1e-6).all()


@pytest.mark.parametrize('kind',['v','x0'])
def test_stable_diffusion_recovers_oracle_at_zero_terminal_snr(kind):
    model,observed,target=example('diffusion',diffusion_prediction=kind);model.eval()
    assert model.alpha_bar[-1]==0
    clean=torch.cat((target['state'],2*target['occurred']-1),-1).repeat_interleave(2,0)
    def oracle(x,t,context):
        if kind=='x0':return clean
        a=model.alpha_bar[t,None,None]
        return (a.sqrt()*x-clean)/(1-a).sqrt()
    model.denoise=oracle
    paths,cdf=model.forecast(observed,2,8)
    torch.testing.assert_close(paths,target['state'][:,None].expand_as(paths),rtol=1e-5,atol=1e-5)
    expected=(2*target['occurred'].flatten(1)+torch.tensor(model.spec['training_event_cdf']))/3
    torch.testing.assert_close(cdf,expected)


def test_velocity_parameterization_does_not_amplify_terminal_error():
    x=torch.randn(2,32,269);pred=torch.randn_like(x)
    clean,_=clean_and_noise(x,pred,torch.tensor(0.),'v')
    altered,_=clean_and_noise(x,pred+.01,torch.tensor(0.),'v')
    torch.testing.assert_close(altered-clean,torch.full_like(x,-.01),rtol=1e-4,atol=1e-6)


def test_step_projection_minimizes_global_curve_error():
    x=torch.randn(5,3,32,4);flat=x.flatten(-2)
    templates=2*(torch.arange(128)[None]>=torch.arange(129)[:,None]).float()-1
    error=(flat[:,:,None]-templates[None,None]).square().sum(-1)
    selected=error.argmin(-1)
    expected=(torch.arange(128)>=selected[...,None]).float().mean(1)
    torch.testing.assert_close(project_onset(x),expected)
    assert torch.count_nonzero(project_onset(torch.zeros(1,2,32,4)))==0
    isolated=torch.full((1,1,32,4),-1.);isolated[0,0,0,0]=1
    assert torch.count_nonzero(project_onset(isolated))==0


def test_stratified_mixture_is_a_normalized_event_distribution():
    model,obs,target=example('mixture',mixture_style='stratified');model.eval();context=model.encode(obs)
    _,_,mass=model.stratified(context,model.anchor(obs,context));p=mass.exp()
    torch.testing.assert_close(p.sum(-1),torch.ones(3,4))
    assert torch.count_nonzero(p[:,~model.event_support])==0
    categories=torch.bucketize(torch.tensor([0,11,12,63,64,127,128]),model.boundaries,right=True)
    assert categories.tolist()==[0,0,1,1,2,2,3]


def test_likelihood_teacher_forcing_is_budget_independent():
    model,_,_=example('ar_gaussian');s=model.spec
    assert teacher_probability(s,500,100,1000)==1
    s=dict(s,teacher_mode='scheduled',teacher_epochs=2.)
    assert teacher_probability(s,100,100,1000)==teacher_probability(s,100,100,10000)==.5
    assert teacher_probability(s,200,100,1000)==0


def test_low_rank_covariance_has_correlated_innovations():
    torch.manual_seed(8);mean=torch.zeros(3);factor=torch.tensor([[1.],[1.],[0.]])
    distribution=torch.distributions.LowRankMultivariateNormal(mean,factor,torch.ones(3)*.01)
    samples=distribution.sample((10000,));cov=torch.cov(samples.T)
    torch.testing.assert_close(cov,distribution.covariance_matrix,atol=.04,rtol=.05)


def test_motion_features_only_read_observed_current_frame():
    data=ResidentPaths.__new__(ResidentPaths);data.device=torch.device('cpu');data.spec={'radius_A':25,'motion_input':True}
    data.rows=torch.tensor([[0,64,1,400]]);data.offsets=torch.tensor([-16,-4,0])
    data.features=torch.randn(1,167,16,7,128);data.geometry=torch.randn(1,167,16,7,3)
    data.states=torch.randn(1,199,16,STATE_DIM);data.mean=torch.zeros(STATE_DIM);data.scale=torch.ones(STATE_DIM)
    observed=data.observed([0]);data.states[:,17:]+=1000
    for key,value in observed.items():torch.testing.assert_close(data.observed([0])[key],value,rtol=0,atol=0)
    assert observed['motion'].shape==(1,43)


def test_promotions_use_validation_and_physical_gate_only(tmp_path):
    config,ref=recipe();(tmp_path/'reference-selection.json').write_text(json.dumps(ref))
    items=screens(config,ref);assert len(items)==30
    assert expanded_tasks(config,tmp_path)==items
    for i,s in enumerate(items):
        folder=tmp_path/'runs'/s['name'];folder.mkdir(parents=True)
        bad='control' in s['name'] or 'v128' in s['name'] or 'likelihood' in s['name']
        status=dict(state='complete',best_selection_brier=.001 if bad else .1+i/10000,
            best_selection_physical_mse=10 if bad else 1.,test_event_nll=-10000 if bad else 10000)
        (folder/'status.json').write_text(json.dumps(status))
    tasks=expanded_tasks(config,tmp_path);assert len(tasks)==35
    promotion=json.loads((tmp_path/'promotions.json').read_text())
    assert all(s['selection_physical_mse']==1. for s in promotion['selection'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA integration')
@pytest.mark.parametrize('method,extra',[('ar_gaussian',{'gaussian_rank':8}),('mixture',{'mixture_style':'stratified'}),('diffusion',{})])
def test_cuda_updates_and_checkpoint_roundtrip(method,extra,tmp_path):
    model,obs,target=example(method,**extra);model.cuda();obs={k:v.cuda() for k,v in obs.items()};target={k:v.cuda() for k,v in target.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    loss=model.loss(obs,target,1.).mean();loss.backward();opt.step()
    assert torch.isfinite(loss)
    checkpoint=tmp_path/'model.pt';torch.save(model.state_dict(),checkpoint)
    other=RefinedForecaster(model.spec).cuda();other.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval();other.eval()
    torch.manual_seed(19);a=model.forecast(obs,3,8)
    torch.manual_seed(19);b=other.forecast(obs,3,8)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
