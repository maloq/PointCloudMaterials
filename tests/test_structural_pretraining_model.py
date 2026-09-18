import copy
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from src.data.structural_pretraining.batches import collate,move
from src.models.encoders.structural import StructuralModel
from src.training_methods.structural_pretraining.objective import Objective,vicreg
from src.training_methods.structural_pretraining.train import cached_update,target_batch


DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sample(seed,t=1,n=12):
    rng=np.random.default_rng(seed);x=rng.uniform(-4,4,(t,n,3)).astype(np.float32);x[:,0]=0
    pairs=cKDTree(x[0]).query_pairs(5.,output_type='ndarray')
    return dict(positions=x,weights=np.ones((t,n),np.float32),center=0,
        times=np.arange(1-t,1,dtype=np.float32)*.1,species=seed%5,log_scale=.05,
        physical=rng.normal(size=85).astype(np.float32),tda=rng.normal(size=144).astype(np.float32),
        tda_valid=True,edges=np.concatenate((pairs,pairs[:,::-1]),axis=0).T)


def model(architecture,history=False):
    torch.manual_seed(18)
    return StructuralModel(architecture,backend='cueq' if DEVICE.type=='cuda' else 'e3nn',history=history).to(DEVICE)


def objective(method):
    n={k:dict(mean=[0.]*d,std=[1.]*d) for k,d in [('physical',85),('tda',144)]}
    return Objective(n,method).to(DEVICE)


@pytest.mark.parametrize('architecture',['mace','gatr'])
def test_symmetry_packing_and_species(architecture):
    net=model(architecture).eval();a=sample(2);b=sample(3,n=17)
    batch=move(collate([a,b],architecture),DEVICE)
    with torch.no_grad():
        y=net.encoder(batch)
        single=net.encoder(move(collate([a],architecture),DEVICE))
    torch.testing.assert_close(y[:1],single,rtol=3e-4,atol=3e-5)
    q=np.linalg.qr(np.random.default_rng(7).normal(size=(3,3)))[0].astype(np.float32)
    rotated=copy.deepcopy(a);rotated['positions']=rotated['positions']@q
    with torch.no_grad():yr=net.encoder(move(collate([rotated],architecture),DEVICE))
    torch.testing.assert_close(single,yr,rtol=3e-4,atol=3e-5)
    changed=copy.deepcopy(a);changed['species']=4;changed['log_scale']=.3
    with torch.no_grad():yc=net.encoder(move(collate([changed],architecture),DEVICE))
    assert not torch.allclose(single,yc,atol=1e-6)


def test_gatr_causality_and_use_of_history():
    net=model('gatr',history=True).eval();batch=move(collate([sample(2,t=3)],'gatr'),DEVICE)
    changed={k:v.clone() for k,v in batch.items()};changed['positions'][:,2,1:]+=2
    with torch.no_grad():
        a=net.encoder.atom_features(batch);b=net.encoder.atom_features(changed)
    torch.testing.assert_close(a[:,:2],b[:,:2],rtol=3e-5,atol=3e-6)
    batch['positions'].requires_grad_(True)
    net.encoder(batch).square().sum().backward()
    assert batch['positions'].grad[:,0].abs().max()>1e-10
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in net.encoder.temporal.parameters())


@pytest.mark.parametrize('architecture,method',[('gatr','vicreg'),('gatr','lejepa'),('mace','vicreg')])
def test_cached_gradient_matches_true_full_batch(architecture,method):
    # cuEquivariance kernel state must be constructed on its target device.
    a=model(architecture,history=method=='lejepa');b=model(architecture,history=method=='lejepa');b.load_state_dict(a.state_dict())
    oa=objective(method);ob=copy.deepcopy(oa)
    optim_a=torch.optim.SGD(a.parameters(),lr=.01);optim_b=torch.optim.SGD(b.parameters(),lr=.01)
    samples=[sample(i,t=3 if method=='lejepa' and i<4 else 1) for i in range(8)]
    full=[move(collate(samples[:4],architecture),DEVICE),move(collate(samples[4:],architecture),DEVICE)]
    chunks=[collate(samples[i:i+2],architecture) for i in range(0,8,2)]
    z=torch.cat([a.encoder(x) for x in full]);target=target_batch(full,DEVICE)
    loss,_=oa(a,z,target,True,torch.full((4,),.1,device=DEVICE));loss.backward()
    torch.nn.utils.clip_grad_norm_(a.parameters(),5.,error_if_nonfinite=True);optim_a.step()
    cached_update(b,ob,chunks,optim_b,True,[.1]*4)
    # Full-batch conditioning can amplify tiny FP32 scatter differences from
    # packing the same graphs into different microbatches. Bound the update's
    # whole gradient as well as the individual parameter differences.
    ga=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
    gb=torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
    assert (ga-gb).norm()/ga.norm()<.002
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters(),strict=True):
        torch.testing.assert_close(p,q,rtol=3e-4,atol=5e-6,msg=name)
    assert int(oa.sigreg.global_step)==int(ob.sigreg.global_step)


def test_between_material_means_do_not_pay_within_material_floor():
    mixed=torch.cat((torch.ones(64,4),-torch.ones(64,4)))
    _,pooled=vicreg(mixed,mixed)
    _,within=vicreg(mixed[:64],mixed[:64])
    assert pooled['variance']==0
    assert within['variance']>.98


def test_sigreg_detects_collapsed_representation():
    obj=objective('lejepa');torch.manual_seed(9)
    x=torch.randn(2048,64,device=DEVICE)
    assert obj.sigreg(x)<obj.sigreg(torch.zeros_like(x))*.1


def test_export_reload_matches():
    net=model('gatr',history=True);other=model('gatr',history=True);other.load_state_dict(net.state_dict())
    batch=move(collate([sample(1,t=3)],'gatr'),DEVICE)
    torch.testing.assert_close(net.encoder(batch),other.encoder(batch),rtol=0,atol=0)


@pytest.mark.skipif(DEVICE.type!='cuda',reason='BF16 training uses CUDA kernels')
@pytest.mark.parametrize('architecture',['mace','gatr'])
def test_bf16_cached_gradient_and_float32_statistics(architecture):
    from src.training_methods.shared_pretraining.runtime import cached_update as mixed_update
    a=model(architecture);b=model(architecture);b.load_state_dict(a.state_dict())
    oa=objective('vicreg');ob=copy.deepcopy(oa)
    samples=[sample(i) for i in range(8)]
    full=[move(collate(samples[:4],architecture),DEVICE),move(collate(samples[4:],architecture),DEVICE)]
    chunks=[collate(samples[i:i+2],architecture) for i in range(0,8,2)]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=torch.cat([a.encoder(x).float() for x in full])
        assert all(v.dtype==torch.float32 for v in a.heads(z).values())
        loss,terms=oa(a,z,target_batch(full,DEVICE),True,torch.full((4,),.75,device=DEVICE))
    assert all(terms[k].dtype==torch.float32 for k in ('loss','vicreg','variance','covariance'))
    loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),5.,error_if_nonfinite=True)
    optimizer=torch.optim.SGD(b.parameters(),lr=0.)
    mixed_update(b,ob,chunks,optimizer,True,[.75]*4,precision='bf16')
    reference=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
    actual=torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
    assert reference.norm()>0
    assert (reference-actual).norm()/reference.norm()<.05
    assert all(p.dtype==torch.float32 for p in b.parameters())
