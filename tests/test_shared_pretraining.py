import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.structural_pretraining.batches import collate,move
from src.training_methods.shared_pretraining.runtime import (CausalModel,CausalObjective,cached_update,learning_rate,atomic_checkpoint)
from src.training_methods.shared_pretraining.data import CausalRelease
from src.training_methods.shared_pretraining.analysis import source_mean,bootstrap_gain
from src.training_methods.structural_pretraining.train import target_batch

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sample(seed,frames):
    rng=np.random.default_rng(seed);x=rng.normal(size=(frames,12,3)).astype(np.float32);x[:,0]=0
    return dict(positions=x,weights=np.ones((frames,12),np.float32),center=0,species=1,log_scale=0.,
        times=np.arange(1-frames,1,dtype=np.float32)*.75,physical=rng.normal(size=85).astype(np.float32),
        tda=rng.normal(size=144).astype(np.float32),tda_valid=True)


def setup():
    torch.manual_seed(17);model=CausalModel('gatr').to(DEVICE)
    norm={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in [('physical',85),('tda',144)]}
    obj=CausalObjective(dict(normalization=norm,forecast_normalization=dict(mean=[0.]*229,std=[1.]*229)),'lejepa').to(DEVICE)
    samples=[sample(i,3 if i<4 else 1) for i in range(8)]
    chunks=[collate(samples[a:b],'gatr') for a,b in [(0,3),(3,4),(4,7),(7,8)]]
    rng=np.random.default_rng(1);extra=dict(future_physical=rng.normal(size=(4,3,85)).astype(np.float32),future_tda=rng.normal(size=(4,3,144)).astype(np.float32))
    return model,obj,samples,chunks,extra


def test_causal_cache_matches_full_batch_with_uneven_chunks():
    a,oa,samples,chunks,extra=setup();b,ob,*_=setup()
    aa=torch.optim.SGD(a.parameters(),lr=.02);bb=torch.optim.SGD(b.parameters(),lr=.02)
    full=[move(collate(samples[:4],'gatr'),DEVICE),move(collate(samples[4:],'gatr'),DEVICE)]
    z=torch.cat([a.encoder(v) for v in full]);target=target_batch(full,DEVICE)
    target.update({k:torch.tensor(v,device=DEVICE) for k,v in extra.items()})
    loss,_=oa(a,z,target,True,torch.full((4,),.75,device=DEVICE));loss.backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in a.future.parameters())
    torch.nn.utils.clip_grad_norm_(a.parameters(),5.);aa.step()
    cached_update(b,ob,chunks,bb,True,[.75]*4,extra)
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters(),strict=True):
        torch.testing.assert_close(p,q,atol=2e-6,rtol=3e-4,msg=name)


def test_warmup_cosine_checkpoint_continuation(tmp_path):
    a,oa,_,chunks,extra=setup();optim=torch.optim.AdamW(a.parameters(),lr=.02)
    identity={'config':{'schedule':dict(peak=.02,warmup_fraction=.2,minimum_ratio=.01)}}
    def advance(model,obj,opt,step):
        for g in opt.param_groups:g['lr']=learning_rate(step,5,**identity['config']['schedule'])
        cached_update(model,obj,chunks,opt,True,[.75]*4,extra)
    for step in [1,2,3]:advance(a,oa,optim,step)
    path=tmp_path/'last.pt';atomic_checkpoint(path,a,oa,optim,3,1.,identity,5)
    advance(a,oa,optim,4);advance(a,oa,optim,5)
    b,ob,*_=setup();other=torch.optim.AdamW(b.parameters(),lr=.02)
    saved=torch.load(path,map_location=DEVICE,weights_only=False)
    b.load_state_dict(saved['model']);ob.load_state_dict(saved['objective']);other.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'].cpu())
    if DEVICE.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    assert saved['scheduler']['next_update']==4
    for step in [4,5]:advance(b,ob,other,step)
    for p,q in zip(a.parameters(),b.parameters(),strict=True):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert int(oa.sigreg.global_step)==int(ob.sigreg.global_step)
    assert learning_rate(1,10,.02)==pytest.approx(.02)
    assert learning_rate(10,10,.02)==pytest.approx(.0002)


def test_heldout_roles_never_enter_causal_sampling(tmp_path):
    shards=[]
    for i,split in enumerate(['train','selection','calibration','test']):
        (tmp_path/'shards'/str(i)).mkdir(parents=True)
        shards.append(dict(task=dict(id=str(i),split=split),material='Al',potential='al-lee2003-meam',static=False,anchors=3))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(state='complete',shards=shards)))
    release=CausalRelease(tmp_path)
    assert list(release.groups.values())==[[0,1,2]]
    assert release.selection==[3,4,5]
    assert release.splits['test']==[9,10,11]


def test_source_balanced_metrics_do_not_weight_large_sources_more():
    value=np.array([[1.],[3.],[10.]]);sources=np.array([1,1,2])
    np.testing.assert_allclose(source_mean(value,sources),[6.])
    base=np.ones((3,3,7));result=bootstrap_gain(base,base*.5,sources,13)
    np.testing.assert_allclose(result['mean_error_reduction'],.5)
    assert result['sources']==2
