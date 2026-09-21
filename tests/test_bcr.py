import copy
import itertools
import numpy as np
import pytest
import torch
from e3nn import o3
from src.training_methods.bcr.data import pack,corrupt,extract_patch,edges,BalancedStream,allowed_levels
from src.training_methods.bcr.model import BCR,Decoder
from src.training_methods.bcr.objective import per_environment

torch.set_num_threads(1)

def config():
    return dict(arm='bcr',encoder=dict(d0=2.,n_ref=8.,radius=4.,channels=4,code_dim=8,cutoff=3.),decoder=dict(irreps='4x0e + 3x1o + 2x2e'))

def fixture():
    g=torch.Generator().manual_seed(2)
    x=torch.randn(2,9,3,generator=g);x[:,0]=0
    return pack(list(x.numpy()))

def noise(batch):return corrupt(batch,[.04,.12],2.,torch.Generator().manual_seed(4))[:3]


def test_gradient_export_reload_and_no_clean_sidepath():
    torch.manual_seed(9);m=BCR(config());b=fixture();y,e,s=noise(b);p,z=m(b,y,s)
    assert torch.equal(z,m.encode(b))
    loss=per_environment(p,e,b,4.).mean();loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
    assert m.encoder.readout[-1].weight.grad.norm()>0
    second=BCR(config());second.load_state_dict(m.state_dict());assert torch.equal(second.encode(b),z)
    old=p.detach();b['positions']=b['positions']*2
    again=m.decoder(y['positions'],y['species'],y['center'],y['mask'],z,torch.log(s/2))
    assert torch.equal(old,again)
    with pytest.raises(TypeError):m.decoder(y['positions'],y['species'],y['center'],y['mask'],z,torch.log(s/2),clean_edges=torch.zeros(2,1))


def test_symmetry_permutation_padding_batch_and_inversion():
    torch.manual_seed(9);m=BCR(config()).eval();b=fixture();y,e,s=noise(b);p,z=m(b,y,s)
    for R in [o3.rand_matrix(),-torch.eye(3)]:
        br=dict(b,positions=b['positions']@R.T);yr=dict(y,positions=y['positions']@R.T)
        pr,zr=m(br,yr,s)
        torch.testing.assert_close(z,zr,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(pr,p@R.T,atol=1e-6,rtol=1e-5)
    perm=torch.tensor([4,2,0,8,7,3,1,6,5]);bp={k:v[:,perm] for k,v in b.items()};yp={k:v[:,perm] for k,v in y.items()}
    pp,zp=m(bp,yp,s);torch.testing.assert_close(z,zp,atol=1e-6,rtol=1e-5);torch.testing.assert_close(pp,p[:,perm],atol=1e-6,rtol=1e-5)
    single={k:v[:1] for k,v in b.items()};torch.testing.assert_close(z[:1],m.encode(single),atol=1e-6,rtol=1e-5)
    padded={k:torch.cat([v,torch.zeros((2,5,*v.shape[2:]),dtype=v.dtype)],1) for k,v in b.items()}
    torch.testing.assert_close(z,m.encode(padded),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(m.encode(b),m.encode(b),rtol=0,atol=0)


def test_corruption_law_loss_weights_and_levels():
    b=pack([np.zeros((10001,3),np.float32)]);y,e,s,l=corrupt(b,[.08],2.,torch.Generator().manual_seed(33))
    assert e[0,0].abs().sum()==0
    assert abs(float(e[0,1:].var())-1)<.03
    assert e[0,1:].mean(0).norm()>1e-5
    assert torch.equal(e,corrupt(b,[.08],2.,torch.Generator().manual_seed(33))[1])
    torch.testing.assert_close(y['positions'],s[:,None,None]*e)
    clean=fixture();p=torch.ones_like(clean['positions']);zero=torch.zeros_like(p)
    L=per_environment(p,zero,clean,4.);torch.testing.assert_close(L,torch.ones(2))
    p[clean['center']]=1e6;torch.testing.assert_close(L,per_environment(p,zero,clean,4.))
    with pytest.raises(ValueError):allowed_levels([.01,.02],2.,.1)
    assert allowed_levels([.01,.02,.12],2.,.01)==[.12]


def test_noisy_graph_and_coincidence_gradients():
    d=Decoder(2.,8,'4x0e + 3x1o + 2x2e');b=fixture();b['positions'][0,1]=b['positions'][0,2]
    b['positions'].requires_grad_();z=torch.randn(2,8,requires_grad=True)
    p=d(b['positions'],b['species'],b['center'],b['mask'],z,torch.zeros(2));p.square().sum().backward()
    assert torch.isfinite(b['positions'].grad).all() and torch.isfinite(z.grad).all()
    before=edges(b['positions'],b['mask'],4.)
    changed=b['positions'].detach().clone();changed[:,1]+=100
    assert edges(changed,b['mask'],4.).shape[1]<before.shape[1]


def test_pbc_images_skew_overflow_and_translation():
    cell=np.array([[3.,0,0],[2.5,2,0],[.2,.1,4.]])
    x=np.array([[.2,.1,.1],[2.8,.2,.1]]);ids=np.array([40,70]);r=3.2
    result=extract_patch(x,cell,0,ids,r)
    expected=[]
    for i,v in enumerate(x):
        for shift in itertools.product(range(-4,5),repeat=3):
            q=v-x[0]+np.array(shift)@cell
            if np.linalg.norm(q)<r:expected.append(tuple(np.round(q,5)))
    np.testing.assert_allclose(sorted(map(tuple,result['positions'])),sorted(expected),atol=2e-6)
    translated=extract_patch(x+np.array([1,2,3]),cell,0,ids,r)
    np.testing.assert_allclose(sorted(map(tuple,translated['positions'])),sorted(expected),atol=2e-6)
    with pytest.raises(ValueError,match='overflow'):extract_patch(x,cell,0,ids,r,max_atoms=1)


def test_accumulation_and_sampler_replay():
    torch.manual_seed(9);a=BCR(config());b=copy.deepcopy(a);x=fixture();y,e,s=noise(x)
    per_environment(a(x,y,s)[0],e,x,4.).mean().backward()
    for i in range(2):
        part={k:v[i:i+1] for k,v in x.items()};noisy={k:v[i:i+1] for k,v in y.items()}
        (per_environment(b(part,noisy,s[i:i+1])[0],e[i:i+1],part,4.).mean()/2).backward()
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
    records=[dict(root=i//4,source=i//2,block=i%2) for i in range(12)]
    s=BalancedStream(records,9);s.draw(10);state=copy.deepcopy(s.state_dict());expected=s.draw(12);s.load_state_dict(state)
    np.testing.assert_array_equal(s.draw(12),expected)


def test_cubic_fcc_symmetries():
    x=np.array([[0,0,0]]+[v for v in itertools.product((-1,0,1),repeat=3) if sum(a*a for a in v)==2],dtype=np.float32)
    m=BCR(config()).eval();b=pack([x]);z=m.encode(b)
    count=0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            R=np.eye(3)[list(perm)]*np.array(signs)[:,None]
            if np.linalg.det(R)>0:
                count+=1;torch.testing.assert_close(z,m.encode(pack([x@R.T])),atol=1e-6,rtol=1e-5)
    assert count==24


def test_matching_root_bootstrap_and_negative_gain():
    from src.training_methods.bcr.evaluate import derangement,matching_bins,paired_root_gain
    roots=['a','b','c','d'];cov=np.ones((4,3));bins=matching_bins(cov)
    idx=derangement(roots,[400]*4,cov,bins,3)
    assert len(set(idx))==4 and all(roots[i]!=roots[j] for i,j in enumerate(idx))
    assert (derangement(['a']*4,[400]*4,cov,bins,3)==-1).all()
    r=paired_root_gain([2,2,2,2],[1,1,1,1],roots,draws=30)
    assert r['gain']==-1 and r['ci95']==[-1,-1]
    assert paired_root_gain([1],[0],['a'],draws=10)['gain']<0
    assert paired_root_gain([1],[1],['a'],draws=10)['ci95'] is None
    # Duplicate anchors within roots do not create narrower uncertainty.
    one=paired_root_gain([1,2],[2,2],['a','b'],seed=4,draws=200)
    two=paired_root_gain([1,1,2],[2,2,2],['a','a','b'],seed=4,draws=200)
    assert one['ci95']==two['ci95'] and one['gain']==two['gain']


def test_resume_exact_optimizer_noise_and_stream(tmp_path):
    import json,hashlib
    from src.training_methods.bcr.runtime import train
    x=fixture()['positions'].numpy();root=tmp_path/'data';root.mkdir()
    np.savez(root/'patches.npz',positions=x.reshape(-1,3),offsets=np.array([0,9,18]))
    records=[dict(root=str(i),source=i,block=0,split='train') for i in range(2)]
    manifest=dict(records=records,d0=2.,n_ref=8.,radius_A=4.,noise_levels=[.04,.12],identity='unit-fixture',patches_sha256=hashlib.sha256((root/'patches.npz').read_bytes()).hexdigest())
    (root/'manifest.json').write_text(json.dumps(manifest))
    cfg=dict(config(),seed=41,updates=4,batch_size=2,microbatch=1)
    train(cfg,root,tmp_path/'full');train(cfg,root,tmp_path/'resume',stop_after=2);train(cfg,root,tmp_path/'resume')
    a=torch.load(tmp_path/'full/technical/last.pt',weights_only=False);b=torch.load(tmp_path/'resume/technical/last.pt',weights_only=False)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    assert a['stream']==b['stream']
    assert torch.equal(a['noise_rng'],b['noise_rng'])
    for k in a['optimizer']['state']:
        for key in a['optimizer']['state'][k]:torch.testing.assert_close(a['optimizer']['state'][k][key],b['optimizer']['state'][k][key],atol=0,rtol=0)


def test_support_boundary_fixed_and_reextracted():
    from src.training_methods.bcr.data import taper
    torch.manual_seed(4);m=BCR(config()).eval();cell=np.eye(3)*20
    x=np.array([[0.,0,0],[1,0,0],[0,1,0],[3.999999,0,0]]);ids=np.arange(4)
    before=extract_patch(x,cell,0,ids,4.)['positions'];x[-1,0]+=2e-6
    after=extract_patch(x,cell,0,ids,4.)['positions']
    assert len(before)==4 and len(after)==3
    torch.testing.assert_close(m.encode(pack([before])),m.encode(pack([after])),atol=1e-6,rtol=1e-5)
    r=torch.tensor([3.19999,3.2,3.20001,3.99999,4.],requires_grad=True)
    w=taper(r,4.);assert w[-1]==0 and w[1]==1
    w.sum().backward();assert torch.isfinite(r.grad).all()


def test_loss_padding_duplication_and_model_controls():
    x=fixture();y,e,s=noise(x);p=torch.randn_like(e)
    base=per_environment(p,e,x,4.)
    def pad(v):return torch.cat((v,torch.zeros((len(v),3,*v.shape[2:]),dtype=v.dtype)),1)
    padded={k:pad(v) for k,v in x.items()}
    torch.testing.assert_close(base,per_environment(pad(p),pad(e),padded,4.))
    doubled={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in x.items()}
    torch.testing.assert_close(base.mean(),per_environment(p.repeat(2,1,1),e.repeat(2,1,1),doubled,4.).mean())
    for arm in ('unconditional','frozen_random','denoising','vicreg'):
        cfg=dict(config(),arm=arm);model=BCR(cfg)
        if arm in ('unconditional','frozen_random'):assert not any(p.requires_grad for p in model.encoder.parameters())
        if arm=='denoising':assert model.encode(x).shape[-1]==2*cfg['encoder']['channels']
        if arm!='vicreg':assert model(x,y,s)[0].shape==e.shape


def test_probe_selection_and_training_only_transforms():
    from src.training_methods.bcr.probes import frozen_probes,select_checkpoint,descriptors
    rng=np.random.default_rng(3);z=rng.normal(size=(30,4));y=z[:,:2]
    records=[dict(root=str(i),temperature_K=400) for i in range(10)]
    result=frozen_probes(z[:10],z[10:20],z[20:],y[:10],y[10:20],y[20:],records,updates=2)
    assert result['ridge']['all']['standardized_rmse']<1e-4
    with pytest.raises(ValueError):select_checkpoint([dict(structural_retention_pass=False,robustness_pass=True)])
    assert select_checkpoint([dict(structural_retention_pass=True,robustness_pass=True,development_nmse=.7,checkpoint='a')])=='a'
    targets,cov=descriptors(list(fixture()['positions'].numpy()),4.)
    assert set(targets)=={'radial','angular','rich'} and all(np.isfinite(v).all() for v in targets.values())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires allocated CUDA GPU')
def test_cuda_accumulation_and_symmetry():
    torch.manual_seed(4);model=BCR(config()).cuda();batch={k:v.cuda() for k,v in fixture().items()};noisy,e,s=noise(batch)
    pred,z=model(batch,noisy,s);R=o3.rand_matrix().cuda();rp,rz=model(dict(batch,positions=batch['positions']@R.T),dict(noisy,positions=noisy['positions']@R.T),s)
    torch.testing.assert_close(z,rz,atol=1e-6,rtol=1e-5);torch.testing.assert_close(rp,pred@R.T,atol=1e-6,rtol=1e-5)
    other=copy.deepcopy(model);per_environment(pred,e,batch,4.).mean().backward()
    for i in range(2):
        a={k:v[i:i+1] for k,v in batch.items()};b={k:v[i:i+1] for k,v in noisy.items()}
        (per_environment(other(a,b,s[i:i+1])[0],e[i:i+1],a,4.).sum()/2).backward()
    for a,b in zip(model.parameters(),other.parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)


def test_precision_gate_and_prototypes(tmp_path):
    from src.training_methods.bcr.runtime import train
    from src.training_methods.bcr.probes import prototypes
    with pytest.raises(ValueError,match='parity'):train(dict(precision='bf16'),tmp_path,tmp_path)
    with pytest.raises(ValueError,match='parity'):train(dict(compile=True),tmp_path,tmp_path)
    assert set(prototypes())=={'FCC','HCP','BCC','icosahedral','disordered'}


def test_root_balanced_evaluation_and_initialization():
    from src.training_methods.bcr.data import balanced_subset
    records=[dict(root=i//10) for i in range(40)];chosen=balanced_subset(records,list(range(40)),12)
    assert len(set(chosen))==12 and all(sum(records[i]['root']==r for i in chosen)==3 for r in range(4))
    states=[]
    for arm in ('bcr','unconditional','vicreg','denoising'):
        torch.manual_seed(51);states.append(BCR(dict(config(),arm=arm)).encoder.state_dict())
    for state in states[1:]:
        for k in state:torch.testing.assert_close(state[k],states[0][k],rtol=0,atol=0)
