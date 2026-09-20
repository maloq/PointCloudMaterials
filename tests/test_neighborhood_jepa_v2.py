"""V2 causal interface, scale, symmetry, query and real CUDA replay contracts."""
import copy
import itertools
import io
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from e3nn import o3
from test_neighborhood_jepa import sample
from src.data.structural_pretraining.batches import collate,move
from src.training_methods.neighborhood_jepa.model import NeighborhoodModel,IRREPS
from src.training_methods.neighborhood_jepa.v2.contracts import LAYOUT,RequiredViewPlan,variants
from src.training_methods.neighborhood_jepa.v2.model import Model,Encoder
from src.training_methods.neighborhood_jepa.v2.objective import Objective,SIGReg
from src.training_methods.neighborhood_jepa.v2.geometry import moments,scaled_error,blocks
from src.training_methods.neighborhood_jepa.v2.data import Data,pack


def fcc():
    shell = sorted(set(itertools.permutations((0.,1.,1.))))
    points = sorted(set(tuple(np.array(p)*s) for p in shell for s in itertools.product((-1,1),repeat=3)))
    return np.vstack((np.zeros((1,3)),np.array(points)*2)).astype(np.float32)


def fixture(arm='E',n=4,all_views=False):
    spec = variants({'seed':1})[ord(arm)-65]
    plan = RequiredViewPlan.from_spec(spec,all_views)
    manifest = dict(normalization={name:dict(mean=[0.]*d,std=[1.]*d) for name,d in [('physical',85),('tda',144)]})
    target = dict(group=torch.zeros(n,dtype=torch.long),temperature_K=torch.full((n,),450.),
        physical=torch.randn(n,2,85),tda=torch.randn(n,2,144),moments=torch.randn(n,len(plan.views),120),
        position=torch.randn(n,7,3),times=torch.tensor([[-.75,0,.75]]).expand(n,3))
    z = torch.randn(n*len(plan.views),LAYOUT.packed_dim,requires_grad=True)
    return spec,plan,manifest,target,z


@pytest.mark.parametrize('train',[True,False])
@pytest.mark.parametrize('n',[1,2,8])
def test_inputs_are_causal_and_batch_independent(train,n):
    spec,plan,_,target,z = fixture(n=n)
    model = Model().train(train)
    original = model.predictions(z,target,plan)
    altered = z.detach().reshape(n,len(plan.views),-1).clone()
    for slot,(time,_) in enumerate(plan.views):
        if time == 2: altered[:,slot] *= 100
    changed = copy.deepcopy(target)
    changed['physical'] *= 999
    changed['tda'] *= -111
    outputs = model.predictions(altered.flatten(0,1),changed,plan)
    for a,b in zip(original,outputs): torch.testing.assert_close(a,b)
    single = model.predictions(z[:len(plan.views)],{k:v[:1] for k,v in target.items()},plan)
    for a,b in zip(original,single): torch.testing.assert_close(a[:1],b,atol=2e-6,rtol=2e-6)
    gradient = torch.autograd.grad(original[0].sum()+original[1].sum(),z)[0].reshape(n,len(plan.views),-1)
    for slot,(time,_) in enumerate(plan.views):
        if time == 2: assert gradient[:,slot].count_nonzero()==0


def test_joint_target_gradients_and_family_weights():
    spec,plan,manifest,target,z = fixture()
    model = Model()
    objective = Objective(manifest,spec)
    loss,_ = objective(model,z,target)
    loss.backward()
    grad = z.grad.reshape(4,len(plan.views),-1)
    assert grad[:,plan.slot(2,1)].abs().sum()>0
    assert [len(RequiredViewPlan.from_spec(s).views) for s in variants({'seed':1})]==[2,2,8,8,14]
    assert sum(spec['family_weights'].values())==3


@pytest.mark.parametrize('mode',['official_test_statistic','per_sample_discrepancy'])
def test_sigreg_replication_semantics_and_resume(mode):
    torch.manual_seed(3)
    a = SIGReg(mode)
    b = copy.deepcopy(a)
    x = torch.randn(9,64,requires_grad=True)
    y = x.detach().repeat_interleave(3,0).requires_grad_(True)
    p,raw,d = a(x)
    q,rawq,dq = b(y)
    torch.testing.assert_close(rawq,3*raw,rtol=2e-5,atol=1e-6)
    torch.testing.assert_close(d,dq,rtol=2e-5,atol=1e-6)
    ga = torch.autograd.grad(d,x)[0]
    gb = torch.autograd.grad(dq,y)[0].reshape(9,3,64).sum(1)
    torch.testing.assert_close(ga,gb,rtol=2e-4,atol=1e-6)
    saved = io.BytesIO()
    torch.save(a.state_dict(),saved)
    saved.seek(0)
    clone = SIGReg(mode)
    clone.load_state_dict(torch.load(saved,weights_only=True))
    torch.testing.assert_close(a(x)[0],clone(x)[0])
    assert torch.isfinite(a(torch.zeros(9,64))[0])


def test_legacy_scale_witness_and_direct_anchor():
    torch.manual_seed(11)
    m = NeighborhoodModel('mace').eval()
    scaled = copy.deepcopy(m)
    a = .25
    with torch.no_grad():
        scaled.encoder.angular_weights[-1].weight.mul_(a)
        scaled.encoder.angular_weights[-1].bias.mul_(a)
        scaled.encoder.compress[0].weight[:,128:].div_(a*a)
        scaled.query[0].weight[:,-16:].div_(a)
        for degree in range(4):
            rows = slice(degree*8+4,degree*8+8)
            scaled.gates.weight[rows].mul_(a)
            scaled.gates.bias[rows].mul_(a)
        scaled.bond.weight.div_(a)
    state = torch.randn(4,248)
    position = torch.randn(4,7,3)
    lag = torch.randn(4,7)
    transform = lambda z:torch.cat((z[:,:128],a*z[:,128:]),-1)
    x = m.predict(state,None,position,lag)
    y = scaled.predict(transform(state),None,position,lag)
    torch.testing.assert_close(x[0],y[0],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(a*x[1],y[1],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(m.bond(state[:,128:]),scaled.bond(a*state[:,128:]))
    target = torch.randn_like(x[1])
    torch.testing.assert_close((y[1]-a*target).square().mean(),a*a*(x[1]-target).square().mean())
    fixed = torch.randn(2,120)
    assert scaled_error(a*fixed,fixed,torch.ones(4,4)).mean()>0
    assert scaled_error(fixed,fixed,torch.ones(4,4)).mean()==0


def test_true_fcc_magnitudes_cubic_stabilizer_parity_and_smoothness():
    x = torch.tensor(fcc())
    y = o3.spherical_harmonics([1,2,4,6],x[1:],normalize=True,normalization='component')
    means = y.mean(0).split([3,5,9,13])
    assert means[0].norm()<1e-6 and means[1].norm()<1e-6
    torch.testing.assert_close(means[2].norm()/3,torch.tensor(.1909406539564933),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(means[3].norm()/13**.5,torch.tensor(.5745242597140698),atol=1e-6,rtol=1e-6)
    graph = torch.zeros(len(x),dtype=torch.long)
    q = moments(x,graph,1)
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.,1.),repeat=3):
            r = torch.eye(3)[list(perm)]*torch.tensor(signs)[:,None]
            if torch.linalg.det(r)>0: rotations.append(r)
    assert len(rotations)==24
    for r in rotations:
        torch.testing.assert_close(moments(x@r.T,graph,1),q,atol=2e-6,rtol=2e-6)
    r = o3.rand_matrix()
    torch.testing.assert_close(moments(x@r.T,graph,1),q@IRREPS.D_from_matrix(r).T,atol=2e-6,rtol=3e-5)
    torch.testing.assert_close(moments(-x,graph,1),q@IRREPS.D_from_matrix(-torch.eye(3)).T,atol=2e-6,rtol=3e-5)
    # Continuous nearest-12/13 exchange and a support crossing.
    base = torch.cat((x,torch.tensor([[2.001,2.,0.],[7.999,0.,0.]])))
    g = torch.zeros(len(base),dtype=torch.long)
    differences = []
    for eps in (.01,.001,.0001):
        a,b = base.clone(),base.clone()
        a[-2:,0] += eps
        b[-2:,0] -= eps
        differences.append((moments(a,g,1)-moments(b,g,1)).norm())
    assert differences[2]<differences[1]<differences[0]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual cuEquivariance CUDA test')
@pytest.mark.parametrize('precision',['float32','bf16'])
def test_encoder_fcc_export_rotation_and_serialization(precision):
    torch.manual_seed(9)
    model = Encoder().cuda().eval()
    r = o3.rand_matrix()
    x = fcc()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        a = model(move(collate([sample(x)],'mace'),'cuda'))
        b = model(move(collate([sample(x@r.numpy().T)],'mace'),'cuda'))
    torch.testing.assert_close(a[:,:128],b[:,:128],atol=2e-3 if precision=='bf16' else 5e-5,rtol=2e-3 if precision=='bf16' else 5e-5)
    torch.testing.assert_close(a[:,128:]@IRREPS.D_from_matrix(r).cuda().T,b[:,128:],atol=2e-4 if precision=='bf16' else 2e-5,rtol=2e-3)
    clone = Encoder().cuda().eval()
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        u = clone.export(move(collate([sample(x)],'mace'),'cuda'))
        v = model.export(move(collate([sample(x)],'mace'),'cuda'))
    torch.testing.assert_close(u['invariant'],v['invariant'])
    assert u['layout']==LAYOUT.metadata()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual cuEquivariance CUDA replay')
@pytest.mark.parametrize('arm',['A','E'])
def test_minimal_views_and_irregular_replay(arm):
    from src.training_methods.neighborhood_jepa.v2.runtime import training_step
    torch.manual_seed(7)
    s,p,m,t,z = fixture(arm,n=2)
    a = Model().cuda()
    b = Model().cuda()
    b.load_state_dict(a.state_dict())
    oa = Objective(m,s).cuda()
    ob = copy.deepcopy(oa)
    samples = [sample(fcc()*(1+.003*i)) for i in range(2*len(p.views))]
    target = move(t,'cuda')
    torch.manual_seed(89)
    encoded = a.encoder(move(collate(samples,'mace'),'cuda'))
    loss,_ = oa(a,encoded,target)
    loss.backward()
    packed = [collate(samples[i:i+3],'mace') for i in range(0,len(samples),3)]
    torch.manual_seed(89)
    replay,_,_ = training_step(b,ob,packed,target,'float32')
    assert abs(float(loss.detach())-replay)<2e-4
    ga = torch.cat([v.grad.flatten() for v in a.parameters() if v.grad is not None])
    gb = torch.cat([v.grad.flatten() for v in b.parameters() if v.grad is not None])
    assert (ga-gb).norm()/ga.norm()<.003


def test_packed_real_producer_identity_and_causality():
    root = Path('/home/ids/vmorozov/training-cache/neighborhood_jepa/v2-native-al-20260920')
    if not (root/'manifest.json').exists(): pytest.skip('Registered native-Al fixture not mounted')
    s = variants({'seed':1})[-1]
    data = Data(root,s)
    _,target = pack([data[data.train[0]],data[data.train[1]]],128)
    assert (target['times'][:,2]==.75).all()
    assert len(torch.unique(target['query_atom_ids'][0]))==7
    model = Model()
    encoded = torch.randn(2*len(data.plan.views),248,requires_grad=True)
    a = model.predictions(encoded,target,data.plan)
    changed = encoded.detach().reshape(2,len(data.plan.views),248).clone()
    for i,(t,_) in enumerate(data.plan.views):
        if t==2: changed[:,i] += 19
    b = model.predictions(changed.flatten(0,1),target,data.plan)
    for u,v in zip(a,b): torch.testing.assert_close(u,v)


def test_all_view_reference_loss_gradient_matches_required_union():
    s,p,m,t,z = fixture('E',n=3)
    full = RequiredViewPlan.from_spec(s,True)
    full_z = torch.randn(3,len(full.views),248,requires_grad=True)
    selected = [full.slot(*view) for view in p.views]
    small_z = full_z.detach()[:,selected].clone().requires_grad_(True)
    full_t = copy.deepcopy(t)
    full_t['moments'] = torch.randn(3,len(full.views),120)
    t['moments'] = full_t['moments'][:,selected]
    model = Model()
    oa,ob = Objective(m,s),Objective(m,s,True)
    ob.load_state_dict(oa.state_dict())
    a,_ = oa(model,small_z.flatten(0,1),t)
    b,_ = ob(model,full_z.flatten(0,1),full_t)
    ga = torch.autograd.grad(a,small_z)[0]
    gb = torch.autograd.grad(b,full_z)[0]
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(ga,gb[:,selected])
    unused = [i for i in range(len(full.views)) if i not in selected]
    assert gb[:,unused].count_nonzero()==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Legacy MACE scale witness on CUDA')
def test_actual_encoder_rescaling_preserves_invariant_and_scalar_tasks():
    torch.manual_seed(92)
    model = NeighborhoodModel('mace').cuda().eval()
    other = NeighborhoodModel('mace').cuda().eval()
    other.load_state_dict(model.state_dict())
    a = .3
    with torch.no_grad():
        other.encoder.angular_weights[-1].weight.mul_(a)
        other.encoder.angular_weights[-1].bias.mul_(a)
        other.encoder.compress[0].weight[:,128:].div_(a*a)
        other.bond.weight.div_(a)
        batch = move(collate([sample(fcc()),sample(fcc()*1.1)],'mace'),'cuda')
        x,y = model.encoder(batch),other.encoder(batch)
        torch.testing.assert_close(x[:,:128],y[:,:128],rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(a*x[:,128:],y[:,128:],rtol=2e-5,atol=2e-5)
        g = torch.zeros(2,dtype=torch.long,device='cuda')
        torch.testing.assert_close(model.physical(x[:,:128],g),other.physical(y[:,:128],g),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(model.bond(x[:,128:]),other.bond(y[:,128:]),rtol=2e-5,atol=2e-5)
    # Weight decay is not a preserved data objective along the witness.
    assert sum(p.square().sum() for p in model.parameters()) != sum(p.square().sum() for p in other.parameters())


@pytest.mark.parametrize('kind',['bcc','hcp','disordered','thermal','vacancy','strain'])
def test_additional_geometric_oracles(kind):
    x = torch.tensor(fcc())
    torch.manual_seed(119)
    if kind=='bcc': x=torch.tensor([[0.,0.,0.]]+list(itertools.product((-2.,2.),repeat=3)))
    if kind=='hcp':
        # Ideal c/a=sqrt(8/3), unit nearest-neighbor spacing, 12-neighbor shell.
        angle = torch.arange(6)*torch.pi/3
        equator = torch.stack((angle.cos(),angle.sin(),torch.zeros(6)),-1)
        angle = torch.arange(3)*2*torch.pi/3+torch.pi/6
        triangle = torch.stack((angle.cos()/3**.5,angle.sin()/3**.5,torch.ones(3)*(2/3)**.5),-1)
        x = torch.cat((torch.zeros(1,3),equator,triangle,triangle*torch.tensor([1,1,-1])))*2.8
    if kind=='disordered': x=torch.cat((torch.zeros(1,3),torch.randn(30,3)*2))
    if kind=='thermal': x=x+torch.randn_like(x)*.05; x[0]=0
    if kind=='vacancy': x=x[:-1]
    if kind=='strain': x=x@torch.tensor([[1.1,.05,0],[0,.95,0],[0,0,1.]])
    graph=torch.zeros(len(x),dtype=torch.long)
    q=moments(x,graph,1)
    r=o3.rand_matrix()
    torch.testing.assert_close(moments(x@r.T,graph,1),q@IRREPS.D_from_matrix(r).T,atol=2e-6,rtol=5e-5)
    torch.testing.assert_close(moments(x[torch.randperm(len(x))],graph,1),q,atol=2e-6,rtol=2e-6)


def test_fixed_target_predictability_positive_and_negative_controls():
    rng=np.random.default_rng(119)
    x=rng.normal(size=(512,8));y=x@rng.normal(size=(8,4))
    fitted=np.linalg.lstsq(x[:256],y[:256],rcond=None)[0]
    actual=np.mean((x[256:]@fitted-y[256:])**2)
    shuffled=np.mean((x[256:]@fitted-y[256:][rng.permutation(256)])**2)
    assert actual<1e-20 and shuffled>1.
    # Future independent noise must not pass the same predictability gate.
    noise=rng.normal(size=(512,4))
    fitted=np.linalg.lstsq(x[:256],noise[:256],rcond=None)[0]
    assert np.mean((x[256:]@fitted-noise[256:])**2)>.9*np.mean(noise[256:]**2)


def test_fixed_fixture_physical_head_can_overfit_without_future_inputs():
    torch.manual_seed(21)
    model=Model()
    fixed=torch.randn(16,128)
    target=fixed[:,:8].repeat(1,11)[:,:85]
    group=torch.zeros(16,dtype=torch.long)
    optimizer=torch.optim.Adam(model.physical_decoder.parameters(),lr=.01)
    initial=(model.physical(fixed,group)-target).square().mean().detach()
    for _ in range(150):
        optimizer.zero_grad()
        loss=(model.physical(fixed,group)-target).square().mean()
        loss.backward()
        optimizer.step()
    assert loss.detach()<initial*.02


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Compiled CUDA BF16 replay test')
def test_compiled_bf16_replay_and_optimizer_resume():
    from src.training_methods.neighborhood_jepa.v2.runtime import training_step
    from src.training_methods.shared_pretraining.compilation import compile_encoder
    torch._dynamo.reset()
    torch.manual_seed(37)
    s,p,m,t,z=fixture('B',n=4)
    a,b=Model().cuda(),Model().cuda()
    b.load_state_dict(a.state_dict())
    oa,ob=Objective(m,s).cuda(),Objective(m,s).cuda()
    samples=[sample(fcc()*(1+.01*i)) for i in range(8)]
    target=move(t,'cuda')
    packed=[collate(samples[i:i+3],'mace') for i in range(0,8,3)]
    compile_encoder(b.encoder,move(packed[0],'cuda'),'bf16')
    torch.manual_seed(100)
    with torch.autocast('cuda',dtype=torch.bfloat16): encoded=a.encoder(move(collate(samples,'mace'),'cuda')).float()
    loss,_=oa(a,encoded,target)
    loss.backward()
    torch.manual_seed(100)
    value,_,_=training_step(b,ob,packed,target,'bf16')
    assert abs(float(loss.detach())-value)<.002
    ga=torch.cat([v.grad.flatten() for v in a.parameters() if v.grad is not None])
    gb=torch.cat([v.grad.flatten() for v in b.parameters() if v.grad is not None])
    relative=float((ga-gb).norm()/ga.norm())
    assert relative<.01,relative
    # Serialize model, optimizer and projection counter, then reproduce the next update.
    opt=torch.optim.AdamW(b.parameters(),lr=.0002)
    opt.step()
    buffer=io.BytesIO()
    torch.save(dict(model=b.state_dict(),objective=ob.state_dict(),optimizer=opt.state_dict()),buffer)
    buffer.seek(0)
    saved=torch.load(buffer,weights_only=False)
    c=Model().cuda();c.load_state_dict(saved['model'])
    oc=Objective(m,s).cuda();oc.load_state_dict(saved['objective'])
    cop=torch.optim.AdamW(c.parameters(),lr=.0002);cop.load_state_dict(saved['optimizer'])
    opt.zero_grad(set_to_none=True);cop.zero_grad(set_to_none=True)
    vb,_,_=training_step(b,ob,packed,target,'bf16')
    vc,_,_=training_step(c,oc,packed,target,'bf16')
    assert abs(vb-vc)<.002
    torch.nn.utils.clip_grad_norm_(b.parameters(),1.);torch.nn.utils.clip_grad_norm_(c.parameters(),1.)
    opt.step();cop.step()
    for pb,pc in zip(b.parameters(),c.parameters()):
        torch.testing.assert_close(pb,pc,atol=2e-5,rtol=2e-4)
