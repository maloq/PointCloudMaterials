import numpy as np
import torch
import pytest
from src.research.crystallization_transfer.model import ContextHead,tensor_invariants
from e3nn import o3
from src.research.crystallization_transfer.data import Corpus,representatives
from src.research.crystallization_transfer.metrics import evaluate
from src.research.crystallization_transfer.queue import variants
from src.research.forecast_crystallization.local_metrics import first_sustained_onset,risk_windows
from scipy.spatial import cKDTree


def test_tensor_context_rotation_and_finite_gradients():
    torch.manual_seed(7)
    spec=dict(radius_A=25,aggregation='attention',equivariant=True,baseline=None)
    head=ContextHead(spec).eval();z=torch.randn(2,7,416,requires_grad=True);g=torch.randn(2,7,4);g[:,:,3]=0;g[:,0,:3]=0
    c=torch.randn(2,7);d=torch.randn(2,136);R=o3.rand_matrix();D=o3.Irreps('32x0e+32x1o+32x2e').D_from_matrix(R)
    rotated=torch.cat((z[...,:128],z[...,128:]@D.T),-1);rg=g.clone();rg[...,:3]=g[...,:3]@R.T
    torch.testing.assert_close(head(z,g,c,d),head(rotated,rg,c,d),atol=3e-5,rtol=3e-5)
    head(z,g,c,d).sum().backward();assert torch.isfinite(z.grad).all()
    zero=torch.zeros(1,1,416,requires_grad=True);tensor_invariants(zero,torch.zeros(1,1,4)).sum().backward();assert torch.isfinite(zero.grad).all()


def fixture_corpus():
    c=Corpus.__new__(Corpus);rng=np.random.default_rng(9);c.plan=dict(anchors=[64,68],config=dict(origin_stride_frames=4),lags=[1,4,12,32,64,128],sources=[dict(id=s,temperature_K=400) for s in range(4)])
    c.arrays={};c.rows=[]
    for s in range(4):
        mapping=np.arange(167*2*7).reshape(167,2,7)
        c.arrays[s]=dict(mapping=mapping,features=rng.normal(size=(167*2*7,416)).astype(np.float32),relative=np.zeros((167,2,7,3),np.float32),event=np.array([[2,6],[1,6]]),packet=rng.normal(size=(2,801,128)),order=rng.normal(size=(2,801,8)),atom_ids=np.array([10,20]),onset=np.array([72,801]),centers=np.tile(np.array([[[0,0,0],[10,0,0]]]),(167,1,1)),boxes=np.full((167,3),100.))
        c.arrays[s]['centers']=c.arrays[s]['centers'].astype(np.float32)
        c.rows.extend((s,a,k,400) for a in range(2) for k in range(2))
    c.source_ids=np.array([r[0] for r in c.rows]);c.events=np.array([c.arrays[s]['event'][a,k] for s,a,k,_ in c.rows]);return c


def test_history_is_causal_and_repeat_control_is_literal():
    c=fixture_corpus();spec=dict(mode='frozen',history_ps=48,radius_A=25)
    batch=c.inputs([0],spec);a=c.arrays[0];wanted=a['features'][a['mapping'][[0,12,15,16],0]]
    np.testing.assert_array_equal(batch['features'][0],wanted.reshape(-1,416))
    repeated=c.inputs([0],dict(spec,repeat=True));r=repeated['features'].reshape(1,4,7,416)
    for t in range(1,4):torch.testing.assert_close(r[:,0],r[:,t],atol=0,rtol=0)
    # Alter every available future feature and target, leaving observations unchanged.
    a['features'][a['mapping'][17:].ravel()]=12345;a['packet'][:,65:]=54321
    torch.testing.assert_close(batch['features'],c.inputs([0],spec)['features'],atol=0,rtol=0)
    torch.testing.assert_close(batch['descriptor'],c.inputs([0],spec)['descriptor'],atol=0,rtol=0)


def test_spatial_representatives_follow_periodic_geometry():
    rng=np.random.default_rng(3);p=rng.uniform(0,50,(3000,3));box=np.full(3,50.);tree=cKDTree(p,boxsize=box)
    ids=representatives(p,0,tree,box);d=p[ids]-p[0];d-=box*np.round(d/box);r=np.linalg.norm(d,axis=1)
    assert ids[0]==0 and len(set(ids))==7 and np.all(r[1:4]<=12) and np.all((r[4:]>12)&(r[4:]<=25))


def test_metrics_include_misses_calibration_and_spatial_population():
    c=fixture_corpus();test=list(range(8,16));cal=list(range(8));logits=np.full((8,6),-2.,np.float32)
    m=evaluate(c,test,logits,cal,logits)
    assert np.isfinite(m['test_event_nll']) and len(m['classification'])==6
    assert m['timing'][-1]['missed_windows']==4
    assert m['spatial'][-1]['nearby_pairs']==4
    assert m['classification'][-1]['false_positive_rate']==0


def test_queue_covers_all_four_requests_and_source_sustained_event():
    specs=variants();assert len({s['name'] for s in specs})==len(specs)
    assert {s['mode'] for s in specs}=={'frozen','finetune','scratch'}
    assert {s['mode'] for s in specs if s['equivariant']}=={'frozen','finetune'}
    assert {s['history_ps'] for s in specs}=={0,3,12,48}
    crystal=np.array([[False,False,True,True,True,False,False]])
    onset=first_sustained_onset(crystal,3);assert onset.item()==2
    assert not risk_windows(crystal,onset,np.array([4,6]),3).any()


@pytest.mark.parametrize('epoch_training',[False,True])
def test_frozen_training_loop_and_metric_exports_on_cuda(tmp_path,monkeypatch,epoch_training):
    import pytest,time,json
    if not torch.cuda.is_available():pytest.skip('CUDA integration')
    from src.research.crystallization_transfer import runtime
    from src.data.structural_pretraining.prepare import save_json
    c=fixture_corpus();cache=tmp_path/'cache';output=tmp_path/'run'
    config=dict(cache=str(cache),output=str(output),seed=7,updates=2,batch_size=8,microbatch_size=2,evaluate_every=1,selection_per_source=4,head_lr=.0005,encoder_lr=.00003,preparation_processes=1,prefetch_batches=2,origin_stride_frames=4)
    if epoch_training:config['batch_size']=3
    plan=dict(c.plan,config=config,identity='synthetic-integration-test',checkpoint_sha256='synthetic',scale=9.192189)
    for source,role in zip(plan['sources'],('train','selection','calibration','test')):
        source['split']=role;sid=source['id'];folder=cache/str(sid);folder.mkdir(parents=True)
        for name,a in c.arrays[sid].items():np.save(folder/f'{name}.npy',a)
        np.save(folder/'risk.npy',np.ones((2,2),bool))
        save_json(folder/'complete.json',dict(identity=plan['identity']))
        save_json(folder/'features.json',dict(checkpoint_sha256='synthetic'))
    monkeypatch.setattr(runtime,'parent_state',lambda _: {})
    spec=dict(name='synthetic-frozen',mode='frozen',history_ps=0,radius_A=0,aggregation='mlp',equivariant=False,baseline=None)
    if epoch_training:spec['training']=dict(sources=1,window_fraction=1.,epochs=2,budget='epochs')
    assert runtime.fit(plan,spec,time.time()+3600)
    root=output/'technical/runs'/spec['name'];s=json.loads((root/'status.json').read_text())
    assert s['state']=='complete' and s['step']==(4 if epoch_training else 2)
    if epoch_training:
        population=json.loads((root/'training-population.json').read_text())
        assert population['eligible_windows']==4 and population['samples']==8
        assert json.loads((root/'metrics.json').read_text())['training']['complete_epochs']==2
    assert (root/'test-index.npz').exists() and (output/'tables/synthetic-frozen.csv').exists()


def test_mean_context_is_literal_weighted_embedding_average():
    spec=dict(radius_A=25,aggregation='mean',equivariant=False,baseline=None)
    head=ContextHead(spec).eval();z=torch.randn(2,7,416);g=torch.zeros(2,7,4);g[:,:,0]=torch.arange(7);c=torch.randn(2,7);d=torch.zeros(2,136)
    u=g[...,:3].norm(dim=-1)/25;w=1-10*u**3+15*u**4-6*u**5
    expected=head.output(torch.cat((z[:,0,:128],(z[...,:128]*w[...,None]).sum(1)/w.sum(1,keepdim=True),c),-1))
    torch.testing.assert_close(head(z,g,c,d),expected)


def test_radius_masks_outside_observations_and_preserves_inside_gradients():
    for radius in (6,12,18,25):
        for tensor in (False,True):
            head=ContextHead(dict(radius_A=radius,aggregation='attention',equivariant=tensor)).eval()
            z=torch.randn(2,21,416,requires_grad=True);g=torch.zeros(2,21,4)
            g[:,:,0]=torch.tensor([0,3,8,11,14,20,24]*3)
            g[:,:,3]=torch.tensor([-12]*7+[-3]*7+[0]*7)
            c=torch.randn(2,7);d=torch.zeros(2,136);outside=g[:,:,0]>=radius
            y=head(z,g,c,d);changed=z.detach().clone();changed[outside]+=100
            torch.testing.assert_close(y,head(changed,g,c,d),atol=0,rtol=0)
            y.sum().backward();assert torch.isfinite(z.grad).all()
            assert torch.count_nonzero(z.grad[outside])==0
            assert z.grad[~outside].abs().sum()>0


def training_corpus():
    c=Corpus.__new__(Corpus);c.plan=dict(config=dict(seed=19,batch_size=7),sources=[])
    c.rows=[];c.groups={};c.splits=dict(train=[],selection=[999],calibration=[1000],test=[1001])
    for sid in range(12):
        c.plan['sources'].append(dict(id=sid,temperature_K=400+50*(sid//6)))
        n=11+sid;indices=np.arange(len(c.rows),len(c.rows)+n);c.groups[sid]=indices
        c.rows.extend((sid,0,0,400) for _ in indices);c.splits['train'].extend(indices.tolist())
    return c


def test_nested_training_data_and_exact_source_weighted_epochs():
    from src.research.crystallization_transfer.training import configure_training,epoch_batch,sample_count
    selected=[]
    for count in (4,8,12):
        c=training_corpus();before={k:list(v) for k,v in c.splits.items() if k!='train'}
        summary=configure_training(c,dict(training=dict(sources=count,window_fraction=1.,epochs=3,budget='full_data_epochs')))
        selected.append(set(c.groups));assert {k:c.splits[k] for k in before}==before
        assert summary['updates']==3*int(np.ceil(summary['full_training_windows']/7))
        total_weights=[c.training_weights[ids].sum() for ids in c.groups.values()]
        np.testing.assert_allclose(total_weights,total_weights[0],rtol=1e-6)
        all_ids=sum((epoch_batch(c,k) for k in range(summary['updates_per_epoch'])),[])
        assert len(set(all_ids))==len(all_ids)==summary['eligible_windows']
        assert set(all_ids)==set(c.splits['train'])
        actual=sum(len(epoch_batch(c,k)) for k in range(summary['updates']))
        assert actual==sample_count(summary['updates'],summary['eligible_windows'],7)
        # Resuming a process at any step produces the same IDs.
        c._epoch=-1;first=epoch_batch(c,3);epoch_batch(c,summary['updates_per_epoch']);assert first==epoch_batch(c,3)
    assert selected[0]<selected[1]<selected[2]
    small=training_corpus();full=training_corpus()
    for corpus,fraction in ((small,.25),(full,1.)):
        configure_training(corpus,dict(training=dict(sources=12,window_fraction=fraction,epochs=1,budget='epochs')))
    for sid in small.groups:assert set(small.groups[sid])<set(full.groups[sid])


def test_microbatch_weighting_includes_short_last_batch():
    from src.research.crystallization_transfer.training import weighted_microbatch_loss
    value=torch.randn(11,requires_grad=True);weights=torch.rand(11)
    split=sum(weighted_microbatch_loss(value[i:i+4],weights[i:i+4],11) for i in range(0,11,4))
    whole=(value*weights).mean();torch.testing.assert_close(split,whole)
    torch.testing.assert_close(torch.autograd.grad(split,value)[0],torch.autograd.grad(whole,value)[0])


def test_scaling_queue_varies_one_factor_and_includes_all_encoder_modes():
    import json
    from pathlib import Path
    config=json.loads(Path('configs/crystallization_transfer/mace_scaling_20260919.json').read_text())
    specs=variants(config);assert len(specs)==52 and len({s['name'] for s in specs})==52
    for mode,tensor in [('frozen',False),('finetune',False),('scratch',False),('frozen',True),('finetune',True)]:
        group=[s for s in specs if (s['mode'],s['equivariant'],s['aggregation'])==(mode,tensor,'attention')]
        assert {s['radius_A'] for s in group}=={0,6,12,18,25}
        assert {s['training']['epochs'] for s in group}=={1,3,6}
        assert {s['training']['sources'] for s in group}=={30,60,90}
        for s in group:
            t=s['training']
            assert sum([s['radius_A']!=25,t['epochs']!=3,t['sources']!=90,t['window_fraction']!=1])<=1


def test_reused_cache_keeps_parent_identity_and_evaluation_population(tmp_path):
    from src.research.crystallization_transfer.data import freeze
    from src.data.structural_pretraining.prepare import file_hash,save_json
    parent=tmp_path/'parent';parent.mkdir();(parent/'parent.pt').write_bytes(b'pinned checkpoint')
    original=dict(identity='original-cache',config=dict(cache='/cache'),checkpoint_sha256=file_hash(parent/'parent.pt'),sources=[dict(id=1)])
    save_json(parent/'plan.json',original)
    config=dict(output=str(tmp_path/'extension'),cache='/cache',reuse_plan=str(parent/'plan.json'))
    plan=freeze(config)
    assert plan['cache_identity']=='original-cache' and plan['identity']!='original-cache'
    assert plan['sources']==original['sources'] and freeze(config)==plan
