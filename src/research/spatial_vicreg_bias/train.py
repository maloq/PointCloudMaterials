"""Matched four-view GeoFormer VICReg: the alignment relation is the treatment."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from src.models import build_encoder
from src.training_methods.shared.vicreg import VICRegLoss
from src.training_methods.shared_pretraining.queue import deadline_for_job
from src.research.structural_state.common import save_checkpoint, write_json, sha, digest
from src.research.supervised_onset.tracking import tracked_run
from .data import load


class PairEncoder(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.encoder = build_encoder(recipe)
        self.vicreg = VICRegLoss.from_config(recipe, input_dim=128)

    def forward(self, x):
        z = self.encoder.forward_features(x)
        return z, self.vicreg.project_features(z)


def initialization(recipe, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return PairEncoder(OmegaConf.create(recipe)).cuda()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def settings(config, index):
    c = load(config)
    seed = c['training']['seed_values'][index // len(c['training']['alphas'])]
    alpha = c['training']['alphas'][index % len(c['training']['alphas'])]
    name = f'S{alpha:g}-seed{seed}'
    root = Path(c['output'])/name
    root.mkdir(parents=True, exist_ok=True)
    recipe = OmegaConf.to_container(OmegaConf.load(c['encoder_recipe']), resolve=True)
    manifest = json.loads((Path(c['cache'])/'manifest.json').read_text())
    identity = digest(dict(config=c, data_identity=manifest['identity'], recipe=recipe,
                           train_sha256=sha(Path(__file__))))
    study = SimpleNamespace(root=root, technical=root/'technical', identity=identity,
        config=dict(c, branch='self_supervised', seed=seed, alpha=alpha,
                    encoder_inputs=dict(geometry='observed periodic relative coordinates',
                        parent_atoms=128, consumed_atoms_per_view=80, views_per_parent=4,
                        history=0, motion=False, conditions=[], species=False,
                        relaxation=False, training_only_teachers=[]), predictor_inputs=None))
    return c, seed, alpha, name, recipe, study


def views(parents, indices, c, seed):
    """Exactly the same tensors for every alpha, including the RNG inside the encoder."""
    torch.manual_seed(seed)
    n = len(parents); rows = torch.arange(n, device='cuda')
    chosen = torch.randint(8, (n,), device='cuda')
    near = indices[rows, chosen].long()
    a = parents[:, :80]
    b = torch.gather(parents, 1, near[..., None].expand(-1, -1, 3))
    b = b - b[:, :1]
    x = torch.stack([a, a, b, b], 0).clone()
    x += torch.randn_like(x) * c['geometry']['jitter_A']
    mirror = torch.rand((4,n,1), device='cuda') < c['geometry']['mirror_probability']
    x[..., 0] *= torch.where(mirror, -1., 1.)
    return (x/c['geometry']['length_scale_A']).flatten(0,1)


def loss_terms(y, alpha):
    y = y.float().reshape(4, -1, 128)
    if not torch.isfinite(y).all():
        raise FloatingPointError('Nonfinite projected embedding; no replacement is permitted')
    a1,a2,b1,b2 = y
    same = ((a1-a2).square().mean() + (b1-b2).square().mean())/2
    cross = ((a1-b2).square().mean() + (b1-a2).square().mean())/2
    variance = torch.relu(1-torch.sqrt(y.var(1, unbiased=False)+1e-4)).mean()
    centered = y-y.mean(1, keepdim=True)
    cov = centered.transpose(1,2)@centered/(y.shape[1]-1)
    covariance = (cov.square().sum((1,2))-cov.diagonal(dim1=1,dim2=2).square().sum(1)).mean()/128
    total = 25*((1-alpha)*same+alpha*cross)+25*variance+covariance
    return total, dict(same=same, cross=cross, variance=variance, covariance=covariance)


def step_seed(seed, epoch, batch):
    return int(np.random.SeedSequence([seed, epoch, batch, 20260929]).generate_state(1)[0])


def fit(config, index):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c, seed, alpha, name, recipe, study = settings(config, index)
    root=study.root; tc=c['training']; cache=Path(c['cache'])
    if (root/'technical/complete.json').exists():
        record=json.loads((root/'technical/complete.json').read_text())
        if record['identity'] != study.identity: raise ValueError('Completed training identity changed')
        return True
    deadline=deadline_for_job()
    model=initialization(recipe,seed)
    forward=torch.compile(model,mode='default',fullgraph=True,dynamic=False) if tc['compile_forward'] else model
    optimizer=torch.optim.AdamW(model.parameters(), lr=tc['learning_rate'], weight_decay=tc['weight_decay'])
    last=root/'checkpoints/last.pt'
    epoch=0; offset=0; update=0
    initial_hash=state_hash(model)
    common=Path(c['output'])/'technical/initializations';common.mkdir(parents=True,exist_ok=True)
    with (common/f'seed{seed}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record=common/f'seed{seed}.json'
        expected=dict(seed=seed,initial_hash=initial_hash,recipe=recipe)
        if record.exists() and json.loads(record.read_text())!=expected:
            raise ValueError('Paired treatments no longer share an initialization')
        if not record.exists():write_json(record,expected)
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['identity'] != study.identity: raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        epoch,offset,update=saved['epoch'],saved['offset'],saved['update']
        if saved['initial_hash'] != initial_hash: raise ValueError('Common initialization changed')
    def snapshot(path, ep, off):
        save_checkpoint(path, dict(format='spatial_vicreg_bias_v1',identity=study.identity,
            model=model.state_dict(),optimizer=optimizer.state_dict(),recipe=recipe,
            epoch=ep,offset=off,update=update,initial_hash=initial_hash,seed=seed,alpha=alpha,
            data_identity=json.loads((cache/'manifest.json').read_text())['identity']))
    if not last.exists():
        snapshot(root/'checkpoints/epoch-00.pt',0,0)
    parents=torch.as_tensor(np.load(cache/'train_parents.npy'),device='cuda')
    indices=torch.as_tensor(np.load(cache/'train_views.npy'),device='cuda')
    n=len(parents); batches=math.ceil(n/tc['batch_size'])
    if n != 1157760: raise ValueError('Training must cover every fixed train anchor/frame')
    write_json(root/'technical/input-contract.json',dict(study.config,
        initial_hash=initial_hash,training_rows=n,updates_per_epoch=batches,
        final_batch_parents=n%tc['batch_size'],epoch_definition='one full fixed structural pass',
        checkpoint_selector='fixed completed passes, no metric selection',
        tensor_producer_sha256=sha(Path(__file__)),training_labels_consumed=[]))
    started=time.monotonic(); begin_update=update
    with tracked_run(study,name) as online:
        while epoch<tc['epochs']:
            order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(n)
            model.train()
            for first in range(offset,n,tc['batch_size']):
                batch=first//tc['batch_size']
                row=torch.as_tensor(order[first:first+tc['batch_size']],device='cuda')
                x=views(parents[row],indices[row],c,step_seed(seed,epoch,batch))
                progress=update/batches
                if progress<tc['warmup_epochs']:
                    lr=tc['learning_rate']*(.05+.95*progress/tc['warmup_epochs'])
                else:
                    phase=(progress-tc['warmup_epochs'])/(tc['epochs']-tc['warmup_epochs'])
                    lr=tc['min_learning_rate']+(tc['learning_rate']-tc['min_learning_rate'])*(1+math.cos(math.pi*phase))/2
                optimizer.param_groups[0]['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): _,y=forward(x)
                loss,terms=loss_terms(y,alpha)
                loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();update+=1
                offset=min(first+tc['batch_size'],n)
                if update%25==0:
                    values=dict(optimizer_update=update,**{'train/'+k:float(v.detach()) for k,v in terms.items()},
                        **{'train/loss':float(loss.detach()),'train/learning_rate':lr,
                           'train/gradient_norm':float(gradient),'train/completed_passes':epoch+offset/n})
                    online.log(values)
                if update%tc['checkpoint_every_updates']==0 or time.time()>deadline:
                    snapshot(last,epoch,offset)
                    write_json(root/'technical/progress.json',dict(state='training',epoch=epoch,offset=offset,
                        update=update,loss=float(loss.detach()),seconds_per_update=(time.monotonic()-started)/(update-begin_update),
                        job=os.environ['SLURM_JOB_ID']))
                if time.time()>deadline:
                    online.summary['training_state']='awaiting_continuation'
                    return False
            epoch+=1;offset=0
            snapshot(root/f'checkpoints/epoch-{epoch:02d}.pt',epoch,0)
            snapshot(last,epoch,0)
            print(json.dumps(dict(name=name,epoch=epoch,update=update,loss=float(loss.detach()))),flush=True)
        online.summary['training_state']='complete'
        online.summary['completed_epochs']=epoch
    write_json(root/'technical/complete.json',dict(identity=study.identity,epochs=epoch,updates=update,
        final_checkpoint_sha256=sha(root/f'checkpoints/epoch-{epoch:02d}.pt'),initial_hash=initial_hash))
    return True


def smoke(config):
    """Local diagnostic on a Slurm GPU; no training run or online logger is created."""
    c=load(config);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    recipe=OmegaConf.to_container(OmegaConf.load(c['encoder_recipe']),resolve=True)
    parents=torch.as_tensor(np.load(Path(c['cache'])/'sources/860/train_parents.npy')[:256],device='cuda')
    ids=torch.as_tensor(np.load(Path(c['cache'])/'sources/860/train_views.npy')[:256],device='cuda')
    model=initialization(recipe,17);initial=state_hash(model)
    forward=torch.compile(model,mode='default',fullgraph=True,dynamic=False) if c['training']['compile_forward'] else model
    x=views(parents,ids,c,123)
    if not torch.equal(x,views(parents,ids,c,123)):raise ValueError('View replay failed')
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
    start=time.monotonic();records=[];timings=[]
    for i in range(8):
        optimizer.zero_grad(set_to_none=True)
        x=views(parents,ids,c,step_seed(17,0,i))
        torch.cuda.synchronize();step_start=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16): _,y=forward(x)
        losses=[loss_terms(y,a)[0] for a in (0.,.5,1.)]
        if not torch.allclose(losses[1],(losses[0]+losses[2])/2,atol=1e-5,rtol=1e-6):
            raise ValueError('Alignment mixing changed shared regularizers')
        losses[1].backward()
        g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize();timings.append(time.monotonic()-step_start)
        records.append(dict(losses=[float(v.detach()) for v in losses],gradient=float(g)))
    torch.cuda.synchronize()
    if state_hash(model)==initial:raise ValueError('Optimizer did not update encoder')
    model.eval();a=parents[:,:80]/c['geometry']['length_scale_A']
    with torch.inference_mode():
        z=model(a)[0];repeat=model(a)[0];small=model(a[:64])[0]
    error=float((z-repeat).abs().max());shape=float((z[:64]-small).abs().max())
    if error!=0 or not torch.allclose(z[:64],small,atol=1e-5,rtol=1e-4):
        raise ValueError(f'Evaluation repeatability failed: same={error}, batch_shape={shape}')
    result=dict(state='complete',created_online_runs=0,initial_hash=initial,steps=records,
        seconds_per_update=(time.monotonic()-start)/8,steady_seconds_per_update=float(np.median(timings[2:])),
        per_update_seconds=timings,peak_gpu_GB=torch.cuda.max_memory_allocated()/1e9,
        matched_batch_max_error=error,batch_shape_max_error=shape,
        producer_sha256=sha(Path(__file__)))
    write_json(Path(c['output'])/'technical/smoke.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['train','smoke'])
    p.add_argument('--config',required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.stage=='smoke':smoke(a.config)
    elif not fit(a.config,a.index):raise SystemExit(75)
