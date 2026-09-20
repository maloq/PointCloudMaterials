import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from src.research.crystallization_transfer.attention import TrainingMoments,AttentionBlock,AdaptiveContextHead
from e3nn import o3
from src.research.crystallization_transfer.refinement import variants,expanded_tasks
from src.research.crystallization_transfer.adaptive import gradient_step,rates
from src.research.local_predictability.metrics import hazard_loss


def spec():
    config=json.loads(Path('configs/crystallization_transfer/mace_adaptive_20260919.json').read_text())
    return variants(config['refinement'])[0]


def inputs(b=6):
    torch.manual_seed(17);z=torch.randn(b,21,416);g=torch.randn(b,21,4)
    g[:,:,:3]*=5;g[:,::7,:3]=0;g[:,:,3]=torch.tensor([-12]*7+[-3]*7+[0]*7)
    return z,g,torch.randn(b,7)


def test_differentiable_normalization_cancels_common_drift_and_excludes_masked_tokens():
    norm=TrainingMoments(8,1e-8);x=torch.randn(5,7,8,requires_grad=True);w=torch.ones(5,7);w[:,-1]=0
    sw=torch.arange(1,6).float();offset=torch.ones(8,requires_grad=True)
    y=norm(x,w,sw);moved=norm(x+offset,w,sw);torch.testing.assert_close(y,moved,atol=2e-6,rtol=2e-6)
    random_weight=torch.randn_like(y)*w[...,None];(moved*random_weight).sum().backward()
    assert offset.grad.abs().max()<1e-5
    assert torch.count_nonzero(x.grad[:,-1])==0
    norm.calibrate(x.detach(),w);norm.eval();expected=norm(x,w)
    torch.testing.assert_close(expected[:2],norm(x[:2],w[:2]),atol=0,rtol=0)
    assert torch.isfinite(expected).all()


@pytest.mark.parametrize('topology',['mean','spatial','temporal','factorized','joint'])
@pytest.mark.parametrize('tensor',[False,True])
def test_attention_rotation_permutation_and_strict_radius_support(topology,tensor):
    s=dict(spec(),attention=topology,equivariant=tensor,radius_A=12)
    head=AdaptiveContextHead(s);z,g,c=inputs();g[:,6::7,:3]=30
    x,w=head.inputs(z,g);head.normalization.calibrate(x,w);head.eval()
    y=head(z,g,c);R=o3.rand_matrix();D=o3.Irreps('32x0e+32x1o+32x2e').D_from_matrix(R)
    rz=torch.cat((z[...,:128],z[...,128:]@D.T),-1);rg=g.clone();rg[...,:3]=g[...,:3]@R.T
    torch.testing.assert_close(y,head(rz,rg,c),atol=5e-5,rtol=5e-5)
    order=torch.tensor([0,4,3,6,5,1,2]);permutation=torch.cat([order+7*t for t in range(3)])
    torch.testing.assert_close(y,head(z[:,permutation],g[:,permutation],c),atol=2e-6,rtol=2e-6)
    z.requires_grad_();outside=w==0;changed=z.detach().clone();changed[outside]+=100
    torch.testing.assert_close(y,head(changed,g,c),atol=0,rtol=0)
    head(z,g,c).sum().backward();assert torch.count_nonzero(z.grad[outside])==0


def test_temporal_attention_does_not_read_later_keys():
    block=AttentionBlock(16,4,True);x=torch.randn(2,3,16);g=torch.zeros(2,3,4);g[:,:,3]=torch.tensor([-12.,-3.,0.]);w=torch.ones(2,3)
    y=block(x,g,w,causal=True);changed=x.clone();changed[:,1:]+=100
    torch.testing.assert_close(y[:,0],block(changed,g,w,causal=True)[:,0],atol=0,rtol=0)


@pytest.mark.parametrize('replay',[True,False])
def test_encoder_gradient_replay_matches_full_batch_with_unequal_microbatches(monkeypatch,replay):
    from src.research.crystallization_transfer import runtime
    monkeypatch.setattr(runtime,'cuda',lambda b:b)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Linear(10,416);self.head=AdaptiveContextHead(spec())
        def encode(self,b):return self.encoder(b['x'])
    torch.manual_seed(31);a=Toy();b=copy.deepcopy(a);_,g,c=inputs(7)
    data=dict(x=torch.randn(7,21,10),geometry=g,condition=c,event=torch.tensor([0,1,2,3,4,5,6]),loss_weight=torch.arange(1,8).float()/4)
    features=a.encode(data);logits=a.head(features,g,c,sample_weight=data['loss_weight'])
    loss=(hazard_loss(logits,data['event'])*data['loss_weight']).mean();loss.backward()
    chunks=[{k:v[start:stop] for k,v in data.items()} for start,stop in [(0,3),(3,6),(6,7)]]
    actual=gradient_step(b,chunks,True,replay=replay);np.testing.assert_allclose(actual,float(loss.detach()),rtol=1e-6)
    for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
        if pa.grad is None:assert pb.grad is None;continue
        torch.testing.assert_close(pa.grad,pb.grad,atol=3e-6,rtol=3e-4,msg=name)


def test_promotions_use_selection_only_and_wait_for_every_screen(tmp_path):
    config=json.loads(Path('configs/crystallization_transfer/mace_adaptive_20260919.json').read_text());screens=variants(config['refinement'])
    assert len(screens)==52
    assert len(expanded_tasks(config,tmp_path))==52
    for i,s in enumerate(screens):
        folder=tmp_path/'runs'/s['name'];folder.mkdir(parents=True)
        (folder/'status.json').write_text(json.dumps(dict(state='complete',best_selection_nll=float(i),test_event_nll=float(-i))))
    tasks=expanded_tasks(config,tmp_path);assert len(tasks)==62
    promotions=json.loads((tmp_path/'promotions.json').read_text());chosen={s['mode']:s['screen'] for s in promotions['selection'] if s['rank']==1}
    assert chosen=={m:f'{m}-reference-E6' for m in ('finetune','scratch','frozen')}
    assert {s['training']['epochs'] for s in promotions['tasks']}=={12,24}
    assert expanded_tasks(config,tmp_path)==tasks


def test_unfreezing_has_separate_warmup_and_budget_matched_schedule():
    s=spec();head,encoder=rates(0,1000,100,s);assert head>0 and encoder==0
    assert rates(99,1000,100,s)[1]==0
    assert 0<rates(100,1000,100,s)[1]<rates(190,1000,100,s)[1]
    assert rates(999,1000,100,s)[1]<rates(500,1000,100,s)[1]


def test_adaptive_runtime_calibration_export_and_short_batch_on_cuda(tmp_path,monkeypatch):
    import time
    if not torch.cuda.is_available():pytest.skip('CUDA integration')
    from test_crystallization_transfer import fixture_corpus
    from src.research.crystallization_transfer import runtime
    from src.data.structural_pretraining.prepare import save_json
    c=fixture_corpus();cache=tmp_path/'cache';output=tmp_path/'run'
    config=dict(cache=str(cache),output=str(output),seed=7,batch_size=3,microbatch_size=2,evaluate_every=1,
        selection_per_source=4,normalization_per_source=2,preparation_processes=1,prefetch_batches=2,origin_stride_frames=4,encoder_backward='direct')
    plan=dict(c.plan,config=config,identity='adaptive-integration-test',checkpoint_sha256='synthetic',scale=9.192189)
    for source,role in zip(plan['sources'],('train','selection','calibration','test')):
        source['split']=role;sid=source['id'];folder=cache/str(sid);folder.mkdir(parents=True)
        for name,a in c.arrays[sid].items():np.save(folder/f'{name}.npy',a)
        np.save(folder/'risk.npy',np.ones((2,2),bool));save_json(folder/'complete.json',dict(identity=plan['identity']))
        save_json(folder/'features.json',dict(checkpoint_sha256='synthetic'))
    monkeypatch.setattr(runtime,'parent_state',lambda _: {})
    s=dict(spec(),name='adaptive-synthetic',mode='frozen',warmup_epochs=0,encoder_lr=0.,training=dict(budget='epochs',epochs=2,sources=1,window_fraction=1.))
    assert runtime.fit(plan,s,time.time()+3600)
    folder=output/'technical/runs'/s['name'];status=json.loads((folder/'status.json').read_text())
    assert status['state']=='complete' and status['step']==4
    checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['model']['head.normalization.calibrated']
    assert (output/'tables/adaptive-synthetic.csv').exists()
    assert len(np.load(folder/'normalization-indices.npy'))==2
    assert runtime.fit(plan,s,time.time()+3600)  # Completed fits do not rerun.


def test_continuations_reuse_frozen_code_and_preserve_slurm_gpu_assignment(tmp_path,monkeypatch):
    from src.research.crystallization_transfer.queue import submit_continuations
    from src.experiment_runner import slurm
    root=tmp_path/'technical';code=root/'code';(code/'configs').mkdir(parents=True)
    (code/'configs/run.json').write_text('{}');(root/'submissions.json').write_text('[]')
    recipe=dict(output=str(tmp_path),frozen_config='configs/run.json',partition='RTX6000PRO,H100',hours=16,cpus=8,memory='64G',slots=[dict(lane=3,after_job=123)])
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe));scripts=[]
    monkeypatch.setattr(slurm,'submit_sbatch',lambda content,path:(scripts.append(content) or '456'))
    submit_continuations(path)
    assert '#SBATCH --dependency=afterany:123' in scripts[0]
    assert '#SBATCH --time=16:00:00' in scripts[0]
    assert 'CUDA_VISIBLE_DEVICES=' not in scripts[0]
    assert str(code) in scripts[0]
    assert json.loads((root/'continuations.json').read_text())[0]['job_id']=='456'
    with pytest.raises(FileExistsError):submit_continuations(path)
