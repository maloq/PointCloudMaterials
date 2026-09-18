import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.data.structural_pretraining.batches import Release
from src.models.encoders.mixed_gatr import GroupBatchNorm, MixedSnapshotGATr
from src.training_methods.shared_pretraining.mixed import (
    MixedObjective, backtracking, calibration_indices, prepare, quotas)
from src.training_methods.shared_pretraining.runtime import cached_update
from src.training_methods.structural_pretraining.train import target_batch

KEYS=[('Al','meam',False),('Mg','eam',False)]
NORM={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in [('physical',85),('tda',144)]}


def test_group_moments_match_training_and_do_not_mix_material_offsets():
    torch.manual_seed(17)
    x=torch.cat((torch.randn(16,8)*.003+1.5,torch.randn(24,8)*.3-5))
    groups=torch.tensor([0]*16+[1]*24)
    norm=GroupBatchNorm(2,8,affine=True)
    train=norm(x,groups)
    changed=x.clone();changed[16:]*=100
    torch.testing.assert_close(norm(changed,groups)[:16],train[:16],atol=0,rtol=0)
    norm.eval();norm.fit(x,groups)
    torch.testing.assert_close(norm(x,groups),train,atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(norm(x[:1],groups[:1]),norm(x,groups)[:1],atol=0,rtol=0)
    with pytest.raises(ValueError,match='normalization group'):norm(x,groups+1)


def test_heads_calibrate_in_order_and_survive_checkpoint_without_batch_dependence():
    torch.manual_seed(3);model=MixedSnapshotGATr(KEYS)
    x=torch.cat((torch.randn(32,128)*.01+1,torch.randn(32,128)*.2-3))
    groups=torch.tensor([0]*32+[1]*32)
    trained=model.heads(x,groups)
    parameters={k:v.clone() for k,v in model.named_parameters()}
    model.eval();model.calibrate(x,groups)
    calibrated=model.heads(x,groups)
    for k in trained:torch.testing.assert_close(trained[k],calibrated[k],atol=1e-4,rtol=2e-4)
    for k,p in model.named_parameters():torch.testing.assert_close(p,parameters[k],rtol=0,atol=0)
    restored=MixedSnapshotGATr(KEYS).eval();restored.load_state_dict(model.state_dict())
    for k,value in restored.heads(x[:1],groups[:1]).items():
        torch.testing.assert_close(value,calibrated[k][:1],atol=1e-5,rtol=1e-4)


def test_backtracking_constant_velocity_irregular_times_and_reversal_gradients():
    torch.manual_seed(29);z=torch.randn(7,128);v=torch.randn_like(z)
    delta=torch.rand(7,2)+.03
    assert backtracking(z-v*delta[:,:1],z,z+v*delta[:,1:],delta)<1e-11
    before=z.clone().requires_grad_();now=(z+v).requires_grad_();after=z.clone().requires_grad_()
    equal=torch.ones(7,2)*.75
    value=backtracking(before,now,after,equal)
    torch.testing.assert_close(value,(after-2*now+before).square().sum(-1).mean())
    value.backward()
    assert all(t.grad.norm()>0 and torch.isfinite(t.grad).all() for t in (before,now,after))


def test_batch_quotas_and_training_only_calibration():
    counts=quotas([142364,15016,29820,29820,37500],2048,128)
    assert counts.sum()==2048 and counts.min()>=128
    release=SimpleNamespace(group_keys=KEYS,groups={KEYS[0]:list(range(30)),KEYS[1]:list(range(30,60))})
    indices=calibration_indices(release,12,8)
    assert indices==calibration_indices(release,12,8)
    assert len(indices)==len(set(indices))==16 and all(i<60 for i in indices)
    assert sum(i<30 for i in indices)==8


def make_release(tmp_path):
    shards=[]
    for name,material,split,static,offset in [('al','Al','train',False,0.),('mg','Mg','train',False,1.),
            ('heldout','Al','selection',False,1000.),('static','Al','train',True,10000.)]:
        record=dict(task=dict(id=name,split=split),material=material,potential='test',static=static,anchors=4,scale=17.)
        shards.append(record)
        if static:continue # Static data must not even be opened.
        folder=tmp_path/'shards'/name;folder.mkdir(parents=True)
        arrays=dict(views=np.tile(np.arange(5),(4,1)),offsets=np.arange(6)*3,
            positions=np.tile([[0.,0,0],[1.,0,0],[0,1.,0]],(5,1)).astype(np.float32),
            atom_ids=np.tile([7,8,9],5),center_ids=np.full(5,7),times=np.array([-.6,-.3,0.,.1]),
            physical=np.full((5,85),offset,dtype=np.float32),tda=np.full((5,144),offset,dtype=np.float32),
            tda_valid=np.array([False,False,True,True,True]))
        arrays['physical'][:2]=np.nan;arrays['tda'][:2]=np.nan
        for key,value in arrays.items():np.save(folder/f'{key}.npy',value)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(state='complete',identity={'id':'test'},shards=shards)))
    return Release(tmp_path,dynamic_only=True)


def test_mixed_sampling_filters_static_and_keeps_order_labels_and_true_time(tmp_path):
    release=make_release(tmp_path)
    assert len(release.rows)==12 and set(release.arrays)=={'al','mg','heldout'}
    assert release.manifest['normalization']['physical']['mean']==[.5]*85
    requested=[];original=release.observation
    def observed(index,which,history,mace):
        requested.append(which);return original(index,which,history,mace)
    release.observation=observed
    config=dict(architecture='gatr',history_frames=1,method='vicreg',seed=17,batch_size=8,
        minimum_group_size=2,microbatch_size=3)
    kinds=set()
    for step in range(5):
        requested.clear()
        batches,temporal,delta,indices,_,_,extra=prepare(release,step,config)
        kinds.add(temporal)
        assert len(indices)==len(set(indices))==8 and not set(indices)&set(release.selection)
        assert sorted(extra['domain'].tolist())==[0]*4+[1]*4
        if temporal:np.testing.assert_allclose(extra['triplet_dt'],np.tile([.3,.1],(8,1)))
        else:assert 'triplet_dt' not in extra
        assert all(b['positions'].shape[1]==1 for b in batches)
        labels=torch.cat([b['tda_valid'] for b in batches])
        assert labels[:16].all() and not labels[16:24].any()
        assert len(labels)==(3 if temporal else 2)*8
        assert set(requested)==({'anchor','future','previous'} if temporal else {'anchor','spatial'})
    assert kinds=={True,False}


class LinearEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(5,128)
    def forward(self,batch):return self.linear(batch['features'])


@pytest.mark.parametrize('temporal',[False,True])
def test_full_statistical_objective_matches_gradient_cache_and_ignores_context_labels(temporal):
    torch.manual_seed(17)
    model=MixedSnapshotGATr(KEYS);model.encoder=LinearEncoder()
    other=copy.deepcopy(model);objective=MixedObjective(NORM,KEYS,.1,.001)
    n=8;count=(3 if temporal else 2)*n
    batch=dict(features=torch.randn(count,5),physical=torch.randn(count,85),
        tda=torch.randn(count,144),tda_valid=torch.ones(count,dtype=torch.bool))
    # A context-only past frame has no descriptor target; fail if it is consumed.
    batch['physical'][2*n:]=float('nan');batch['tda'][2*n:]=float('nan');batch['tda_valid'][2*n:]=False
    domains=torch.tensor([0,1]*4);extra=dict(domain=domains)
    if temporal:extra['triplet_dt']=torch.ones(n,2)*.1
    chunks=[{k:v[a:a+5] for k,v in batch.items()} for a in range(0,count,5)]
    target=target_batch([batch],'cpu');target.update(extra)
    z=model.encoder(batch);loss,terms=objective(model,z,target,temporal,torch.ones(n))
    assert terms['labelled_views']==16
    if not temporal:
        assert terms['backtracking']==terms['backtracking_weighted']==0
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    opt=torch.optim.SGD(model.parameters(),lr=.01);opt.step()
    cached_update(other,objective,chunks,torch.optim.SGD(other.parameters(),lr=.01),temporal,[.1]*n,extra)
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters(),strict=True):
        torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5,msg=name)


def test_vicreg_cannot_use_between_material_variation_to_avoid_variance_penalty():
    from src.training_methods.structural_pretraining.objective import vicreg
    q=torch.cat((torch.full((16,64),-3.),torch.full((16,64),3.)))
    assert vicreg(q,q)[1]['variance']==0
    within=sum(vicreg(a,a)[1]['variance']/2 for a in q.chunk(2))
    assert within==pytest.approx(.99)


def test_parallel_preparation_preserves_sampler_order_and_cache_accounting(tmp_path):
    release=make_release(tmp_path)
    config=dict(architecture='gatr',history_frames=1,method='vicreg',seed=17,batch_size=8,
        minimum_group_size=2,microbatch_size=3)
    serial=prepare(release,2,config)
    # Eviction and duplicated concurrent requests must preserve exact LRU bytes.
    release.max_graph_bytes=1000
    parallel=prepare(release,2,dict(config,preparation_workers=4))
    assert serial[1:6]==parallel[1:6]
    for a,b in zip(serial[0],parallel[0],strict=True):
        for key in a:torch.testing.assert_close(a[key],b[key],equal_nan=True,rtol=0,atol=0)
    assert release.graph_bytes==sum(v['cache_bytes'] for v in release.graphs.values())


def test_objective_transition_preserves_weights_optimizer_rng_and_schedule(tmp_path):
    from src.models.encoders.mixed_gatr import MIXED_ARCHITECTURE_REVISION
    from src.training_methods.shared_pretraining.runtime import atomic_checkpoint
    from src.training_methods.shared_pretraining.initialization import continue_mixed_objective
    torch.manual_seed(83)
    model=MixedSnapshotGATr(KEYS);model.encoder=LinearEncoder()
    objective=MixedObjective(NORM,KEYS,.1,.001)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    data=dict(features=torch.randn(24,5),physical=torch.randn(24,85),tda=torch.randn(24,144),
              tda_valid=torch.ones(24,dtype=torch.bool))
    extra=dict(domain=torch.tensor([0,1]*4),triplet_dt=torch.ones(8,2)*.1)
    cached_update(model,objective,[data],optimizer,True,[.1]*8,extra)
    config=dict(seed=83,backtracking_weight=.001,schedule=dict(peak=.002,warmup_fraction=.1,minimum_ratio=.01))
    old=dict(protocol='shared_pretraining_mixed_v7',architecture_revision=MIXED_ARCHITECTURE_REVISION,
             data={'id':'immutable'},config=config)
    atomic_checkpoint(tmp_path/'last.pt',model,objective,optimizer,1,.3,old,100)
    saved=torch.load(tmp_path/'last.pt',map_location='cpu',weights_only=False)
    changed=dict(old,protocol='shared_pretraining_mixed_v8',config=dict(config,backtracking_weight=10.,continue_from='last.pt'))
    new=MixedSnapshotGATr(KEYS);new.encoder=LinearEncoder()
    updated=MixedObjective(NORM,KEYS,.1,10.)
    new_optimizer=torch.optim.AdamW(new.parameters(),lr=1.)
    receipt=continue_mixed_objective(new,updated,new_optimizer,saved,changed,100)
    assert receipt['parent_step']==1 and updated.backtracking_weight==10.
    for key,value in model.state_dict().items():torch.testing.assert_close(new.state_dict()[key],value,atol=0,rtol=0)
    assert new_optimizer.param_groups[0]['lr']==optimizer.param_groups[0]['lr']
    for key,state in optimizer.state_dict()['state'].items():
        for name,value in state.items():torch.testing.assert_close(new_optimizer.state_dict()['state'][key][name],value,atol=0,rtol=0)
    torch.testing.assert_close(torch.get_rng_state(),saved['torch_rng'],atol=0,rtol=0)
    with pytest.raises(ValueError,match='update budget'):
        continue_mixed_objective(new,updated,new_optimizer,saved,changed,101)
    with pytest.raises(ValueError,match='only backtracking'):
        continue_mixed_objective(new,updated,new_optimizer,saved,dict(changed,config=dict(changed['config'],seed=84)),100)
    updated.physical_std.mul_(2)
    with pytest.raises(ValueError,match='statistical buffer'):
        continue_mixed_objective(new,updated,new_optimizer,saved,changed,100)
