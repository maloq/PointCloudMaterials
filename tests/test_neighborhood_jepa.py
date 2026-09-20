import copy
import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.spatial import cKDTree
from src.data.structural_pretraining.batches import collate,move
from src.training_methods.neighborhood_jepa.model import NeighborhoodEncoder,NeighborhoodModel,IRREPS,EQ_DIM,split_tensors
from src.training_methods.neighborhood_jepa.objective import Objective
from src.training_methods.neighborhood_jepa.prepare import distributed_neighbors
from src.training_methods.neighborhood_jepa.variants import screens,promotions


def sample(x):
    pairs=cKDTree(x).query_pairs(5.,output_type='ndarray');edges=np.concatenate((pairs,pairs[:,::-1]),0).T
    return dict(positions=x[None].astype(np.float32),weights=np.ones((1,len(x)),np.float32),center=0,times=np.array([0.],np.float32),
        species=1,log_scale=0.,edges=edges,physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=True)


def cube():return np.vstack((np.zeros((1,3)),np.array([[i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]])*2)).astype(np.float32)


def spec(**changes):
    s=next(s for s in screens({'screen_epochs':3}) if s['name']=='spacetime-six');s.update(changes);return s


def test_neighbor_selection_rotates_and_keeps_identity():
    x=np.random.default_rng(1).normal(size=(40,3));r=o3.rand_matrix().numpy()
    a=distributed_neighbors(x,6,np.random.default_rng(4));b=distributed_neighbors(x@r.T,6,np.random.default_rng(4))
    np.testing.assert_array_equal(a,b);assert len(set(a))==6


def test_promotions_never_use_latent_or_test_loss():
    specs=screens({'screen_epochs':3});status={s['name']:{'selection_score':1+i,'test_score':-i} for i,s in enumerate(specs)}
    result=promotions(specs,status,8)
    assert len(result)==3 and result[0]['promoted_from']=='anchors-sigreg'
    assert result[1]['promoted_from']=='spatial-six'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuEquivariance GPU test')
@pytest.mark.parametrize('bf16',[False,True])
def test_cubic_features_and_rotation(bf16):
    torch.manual_seed(12);model=NeighborhoodEncoder('mace').cuda().eval();x=cube();r=o3.rand_matrix();d=IRREPS.D_from_matrix(r).cuda()
    a=move(collate([sample(x)],'mace'),'cuda');b=move(collate([sample(x@r.numpy().T)],'mace'),'cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):u=model(a);v=model(b)
    torch.testing.assert_close(u[:,:128],v[:,:128],atol=2e-3 if bf16 else 5e-5,rtol=2e-3 if bf16 else 5e-5)
    torch.testing.assert_close(u[:,128:]@d.T,v[:,128:],atol=2e-4 if bf16 else 2e-5,rtol=2e-3 if bf16 else 1e-4)
    blocks=split_tensors(u[:,128:]);assert blocks[0].abs().max()<1e-5 and blocks[1].abs().max()<1e-5
    assert blocks[2].norm()>1e-3 and blocks[3].norm()>1e-3


@pytest.mark.parametrize('history',[False,True])
def test_predictor_equivariance_and_query_permutation(history):
    torch.manual_seed(4);m=NeighborhoodModel('mace',history).eval();r=o3.rand_matrix();d=IRREPS.D_from_matrix(r)
    current=torch.randn(2,128+EQ_DIM);previous=torch.randn_like(current);position=torch.randn(2,6,3);lag=torch.randn(2,6);past=-torch.ones(2)
    a=m.predict(current,previous,position,lag,past)
    transform=lambda z:torch.cat((z[:,:128],z[:,128:]@d.T),-1)
    b=m.predict(transform(current),transform(previous),position@r.T,lag,past)
    torch.testing.assert_close(a[0],b[0],atol=1e-5,rtol=1e-5);torch.testing.assert_close(a[1]@d.T,b[1],atol=2e-5,rtol=1e-5)
    perm=torch.randperm(6);c=m.predict(current,previous,position[:,perm],lag[:,perm],past)
    for x,y in zip(a,c):torch.testing.assert_close(x[:,perm],y)


def test_joint_target_gradients_and_fixed_future_anchors():
    torch.manual_seed(10);s=spec();model=NeighborhoodModel('mace');manifest=dict(groups=[['Al','potential']],normalization={name:dict(mean=[0.]*n,std=[1.]*n) for name,n in [('physical',85),('tda',144)]})
    objective=Objective(manifest,s);b=8;z=torch.randn(b*3*7,128+EQ_DIM,requires_grad=True)
    target=dict(group=torch.zeros(b,dtype=torch.long),position=torch.randn(b,7,3),times=torch.tensor([[-.75,0,.75]]).expand(b,3),
        physical=torch.randn(b,2,85),tda=torch.randn(b,2,144),bonds=torch.randn(b,2,12,3))
    loss,terms=objective(model,z,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad.reshape(b,3,7,-1)[:,2,1:,128:].abs().sum()>0
    assert terms['future']>0


def test_spatial_prediction_has_no_future_prediction_loss():
    s=spec(prediction='spatial');model=NeighborhoodModel('mace');manifest=dict(groups=[['Al','potential']],normalization={name:dict(mean=[0.]*n,std=[1.]*n) for name,n in [('physical',85),('tda',144)]})
    objective=Objective(manifest,s);b=8;z=torch.randn(b*3*7,128+EQ_DIM,requires_grad=True)
    target=dict(group=torch.zeros(b,dtype=torch.long),position=torch.randn(b,7,3),times=torch.tensor([[-1.,0,1.]]).expand(b,3),physical=torch.randn(b,2,85),tda=torch.randn(b,2,144),bonds=torch.randn(b,2,12,3))
    loss,terms=objective(model,z,target);assert terms['future']==0


def test_validation_normalization_cannot_see_future_or_other_samples():
    manifest=dict(groups=[['Al','potential']],normalization={name:dict(mean=[0.]*n,std=[1.]*n) for name,n in [('physical',85),('tda',144)]})
    obj=Objective(manifest,spec()).eval();z=torch.randn(3,3,7,128+EQ_DIM);g=torch.zeros(3,dtype=torch.long)
    a=obj.normalized(z,g)[:,1,0];changed=z.clone();changed[:,2]*=999
    torch.testing.assert_close(a,obj.normalized(changed,g)[:,1,0])
    torch.testing.assert_close(a[:1],obj.normalized(z[:1],g[:1])[:,1,0])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuEquivariance GPU test')
def test_full_batch_and_joint_gradient_replay_agree():
    from src.training_methods.neighborhood_jepa.runtime import training_step
    torch.manual_seed(17);s=spec();a=NeighborhoodModel('mace').cuda();b=NeighborhoodModel('mace').cuda();b.load_state_dict(a.state_dict())
    manifest=dict(groups=[['Al','potential']],normalization={name:dict(mean=[0.]*n,std=[1.]*n) for name,n in [('physical',85),('tda',144)]})
    oa=Objective(manifest,s).cuda();ob=Objective(manifest,s).cuda();samples=[sample(cube()*(1+i*.001)) for i in range(4*3*7)]
    target=dict(group=torch.zeros(4,dtype=torch.long,device='cuda'),position=torch.randn(4,7,3,device='cuda'),times=torch.tensor([[-.75,0,.75]],device='cuda').expand(4,3),physical=torch.randn(4,2,85,device='cuda'),tda=torch.randn(4,2,144,device='cuda'),bonds=torch.randn(4,2,12,3,device='cuda'))
    torch.manual_seed(99);encoded=a.encoder(move(collate(samples,'mace'),'cuda'));loss,_=oa(a,encoded,target);loss.backward()
    torch.manual_seed(99);other,_=training_step(b,ob,[collate(samples[i:i+21],'mace') for i in range(0,len(samples),21)],target,'float32')
    assert abs(float(loss)-other)<2e-4
    ag=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None]);bg=torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
    relative=(ag-bg).norm()/ag.norm().clamp_min(1e-9)
    assert relative<.003,float(relative)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuEquivariance GPU test')
@pytest.mark.parametrize('bf16',[False,True])
def test_disordered_snapshot_rotation_and_atom_permutation(bf16):
    torch.manual_seed(33);model=NeighborhoodEncoder('mace').cuda().eval();rng=np.random.default_rng(8)
    x=np.vstack((np.zeros((1,3)),rng.uniform(-3,3,(79,3)))).astype(np.float32);r=o3.rand_matrix();d=IRREPS.D_from_matrix(r).cuda()
    perm=np.r_[0,1+rng.permutation(79)]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
        u=model(move(collate([sample(x)],'mace'),'cuda'));v=model(move(collate([sample((x@r.numpy().T)[perm])],'mace'),'cuda'))
    torch.testing.assert_close(u[:,:128],v[:,:128],atol=2e-3 if bf16 else 5e-5,rtol=2e-3 if bf16 else 5e-5)
    torch.testing.assert_close(u[:,128:]@d.T,v[:,128:],atol=2e-4 if bf16 else 2e-5,rtol=2e-3 if bf16 else 1e-4)
