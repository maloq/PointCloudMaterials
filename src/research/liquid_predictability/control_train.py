"""MACE sensitivity controls and rich-descriptor prediction through one state."""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.array_exports import ArrayExport
from src.experiment_runner.checkpoints import TrainingState
from src.experiment_runner.artifacts import implementation_hashes
from src.research.crystal_vector.model import JointCrystalVector, vcreg
from src.research.crystal_vector.train import compile_model, deadline
from src.research.equivariant_context.model import geometry
from src.research.supervised_onset.tracking import tracked_run, local_evaluation, update_training_summary
from .data import config
from .descriptor_data import load
from .descriptor_fit import targets, quantities, metrics

FAMILIES = ('geometry','bond_order','cna','tda')


def table(root, name, rows, *, family='liquid_controls'):
    return write_metric_rows(rows, root, family=family, name=name)


class ControlMACE(JointCrystalVector):
    def __init__(self, enc, c, outputs):
        super().__init__(enc,c)
        del self.distance_head, self.direction_channels, self.direction_offset, self.combine
        # The only decoder input is one exported context state. Every patch
        # shares the same MACE, and no target descriptor is an input feature.
        self.context_export=nn.Sequential(nn.LayerNorm(c['predictor']['width']),nn.Linear(c['predictor']['width'],self.latent_dim),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(self.latent_dim,2*self.latent_dim),nn.SiLU(),nn.Linear(2*self.latent_dim,outputs))
        self.register_buffer('output_mask',torch.ones(outputs))

    def forward(self,batch):
        z,v=self.encode(batch['positions']);z=z[batch['inverse']];v=v[batch['inverse']]
        g=geometry(batch['actual'],batch['actual']);s=self.stem(z)+self.geometry(g['node']);fields={1:v}
        for block in self.blocks:s,fields=block(s,fields,g)
        state=self.context_export(s.mean(1))
        return dict(prediction=self.readout(state).float()*self.output_mask,z=z.float(),v=v.float(),state=state.float())


class ControlData:
    def __init__(self,c,arm,device,small=False):
        dc=config(resolve_path(arm['descriptor_config']));self.dc=dc
        x,self.rows,self.columns,self.manifest=load(dc);self.device=device
        self.split={r:np.flatnonzero(self.rows['role']==r) for r in ('train','selection','calibration','test')}
        geom=config(resolve_path(arm['geometry']))
        if sha(Path(geom['index']))!=geom['index_sha256']:raise ValueError('Geometry index changed')
        with np.load(geom['index']) as a:indices=a['indices'];actual=a['actual']
        if len(indices)!=len(x):raise ValueError('Geometry and descriptor row counts disagree')
        if small:
            rng=np.random.default_rng(c['seed']);chosen=[]
            for role,ids in self.split.items():
                self.split[role]=np.sort(rng.choice(ids,min(len(ids),c['batch_size']),replace=False));chosen.extend(self.split[role])
        else:chosen=np.arange(len(x))
        unique=np.unique(indices[chosen]);self.indices=torch.as_tensor(np.searchsorted(unique,indices),device=device)
        self.actual=torch.as_tensor(actual,device=device)
        self.positions=torch.empty((len(unique),80,3),device=device,dtype=torch.float32)
        offset=0
        for bank in geom['banks']:
            lo,hi=np.searchsorted(unique,[offset,offset+bank['rows']])
            if hi>lo:
                if sha(Path(bank['path']))!=bank['sha256']:raise ValueError(f'Changed coordinates {bank["path"]}')
                values=np.load(bank['path'],mmap_mode='r')
                for start in range(lo,hi,8192):
                    end=min(start+8192,hi)
                    self.positions[start:end].copy_(torch.tensor(np.array(values[unique[start:end]-offset]),device=device))
            offset+=bank['rows']
        if len(unique) and unique[-1]>=offset:raise ValueError('Unresolved patch bank index')
        self.weights={r:self.rows['weights'][ids]/self.rows['weights'][ids].sum() for r,ids in self.split.items()}
        self.feature_task=arm['task']=='features'
        if self.feature_task:
            # Fit transforms on the complete declared training cohort, including
            # during local numerical checks; no validation/test statistics.
            train=np.flatnonzero(self.rows['role']=='train');w=self.rows['weights'][train];w=w/w.sum()
            mean=np.zeros(x.shape[1]);second=mean.copy()
            for start in range(0,len(train),1024):
                ix=train[start:start+1024];a=np.asarray(x[ix],float);ww=w[start:start+len(ix)]
                mean+=ww@a;second+=ww@(a*a)
            std=np.sqrt(np.maximum(second-mean*mean,0));self.mean=mean;self.scale=std.clip(1e-4);self.active=std>=1e-4
            family=np.array([v['family'] for v in self.columns]);loss_weight=np.zeros(len(mean))
            for f in FAMILIES:
                mask=(family==f)&self.active
                if not mask.any():raise ValueError(f'No varying targets in family {f}')
                loss_weight[mask]=1/(len(FAMILIES)*mask.sum())
            self.loss_weight=torch.tensor(loss_weight,dtype=torch.float32,device=device)
            self.target=torch.empty(x.shape,dtype=torch.float32,device=device)
            for start in range(0,len(x),2048):
                a=(np.asarray(x[start:start+2048],float)-mean)/self.scale
                self.target[start:start+len(a)].copy_(torch.tensor(a,dtype=torch.float32,device=device))
            self.outputs=x.shape[1]
        else:
            self.target=torch.tensor(targets(self.rows['target'],dc),device=device);self.outputs=len(dc['distance_edges_A'])

    def batch(self,ids):
        ix=torch.as_tensor(ids,device=self.device);patches,inverse=torch.unique(self.indices[ix].flatten(),return_inverse=True)
        return dict(positions=self.positions[patches],inverse=inverse.reshape(-1,25),actual=self.actual[ix],target=self.target[ix])

    def loss(self,prediction,target):
        if self.feature_task:
            return .5*((prediction-target).square()*self.loss_weight).sum(1)+.5*math.log(2*math.pi)
        return nn.functional.cross_entropy(prediction,target,reduction='none')


@torch.no_grad()
def validation(model,data,c):
    model.eval();score=0.
    ids=data.split['selection'];w=data.weights['selection']
    for start in range(0,len(ids),c['batch_size']):
        b=data.batch(ids[start:start+c['batch_size']])
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(b)
        score+=float(w[start:start+len(b['target'])]@data.loss(o['prediction'],b['target']).cpu().numpy())
    return score


@torch.no_grad()
def export(model,data,c,study):
    root=study.root/'analyses/prediction-v1';tech=root/'technical';tech.mkdir(parents=True,exist_ok=True);model.eval()
    count = len(data.rows['ids'])
    specifications = dict(predictions=((count, data.outputs), np.float32),
                          states=((count, model.latent_dim), np.float32))
    with ArrayExport(tech, specifications) as arrays:
        for start in range(0, count, c['batch_size']):
            b = data.batch(np.arange(start, min(start+c['batch_size'], count)))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(b)
            arrays.write(start, predictions=out['prediction'].cpu().numpy(),
                         states=out['state'].cpu().numpy())
    pred=np.load(tech/'predictions.npy',mmap_mode='r');scores=[];features=[];summary={}
    if data.feature_task:
        families=np.array([v['family'] for v in data.columns]);fw=data.loss_weight.cpu().numpy()
        for role,ids in data.split.items():
            w=data.weights[role];error=np.zeros(data.outputs);baseline=error.copy();target_mu=error.copy();second=error.copy()
            for start in range(0,len(ids),1024):
                ix=ids[start:start+1024];ww=w[start:start+len(ix)];y=data.target[torch.as_tensor(ix,device=data.device)].cpu().numpy().astype(float)
                error+=ww@((np.asarray(pred[ix],float)-y)**2);baseline+=ww@(y*y);target_mu+=ww@y;second+=ww@(y*y)
            for f in FAMILIES:
                take=(families==f)&data.active;model_mse=float(error[take].mean());base_mse=float(baseline[take].mean())
                scores.append(dict(role=role,family=f,varying_features=int(take.sum()),standardized_mse=model_mse,
                    training_mean_mse=base_mse,skill_over_training_mean=1-model_mse/base_mse))
                summary[f'evaluation/{role}/{f}_standardized_mse']=model_mse
            for j,col in enumerate(data.columns):
                var=second[j]-target_mu[j]**2
                features.append(dict(role=role,feature=col['name'],family=col['family'],trained=bool(data.active[j]),
                    rmse_native_units=float(np.sqrt(error[j])*data.scale[j]),standardized_mse=float(error[j]),
                    r2=1-float(error[j]/var) if var>1e-10 else None,training_mean_mse=float(baseline[j])))
            summary[f'evaluation/{role}/balanced_gaussian_nll']=float(.5*(error@fw)+.5*math.log(2*math.pi))
        table(root,'features',features,family=c.get('metric_family','liquid_controls'))
    else:
        from scipy.special import softmax
        probability=softmax(np.asarray(pred,dtype=float),axis=1);value=quantities(probability,data.rows['target'],data.dc)
        np.savez(tech/'distance-predictions.npz',ids=data.rows['ids'],probability=probability.astype(np.float32),**value)
        for role,ids in data.split.items():
            score=metrics({k:v[ids] for k,v in value.items()},data.rows['target'][ids],data.weights[role])
            scores.append(dict(role=role,**score));summary.update({f'evaluation/{role}/{k}':v for k,v in score.items()})
    table(root,'scores',scores,family=c.get('metric_family','liquid_controls'))
    np.save(tech/'row-ids.npy',data.rows['ids'])
    write_json(tech/'complete.json',dict(identity=study.identity,checkpoint_sha256=sha(study.technical/'best.pt'),
               files={n:sha(tech/n) for n in ('predictions.npy','states.npy','row-ids.npy')},summary=summary))
    return summary


def run(c,name,preflight=False):
    arm=next(a for a in c['mace_arms'] if a['name']==name);torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);device=torch.device('cuda')
    root=resolve_path(c['output'])/'mace'/name;tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    if not preflight and (root/'analyses/prediction-v1/technical/complete.json').exists():return
    data=ControlData(c,arm,device,small=preflight)
    model=ControlMACE(c['encoder_config'],c,data.outputs).to(device)
    if data.feature_task:model.output_mask.copy_(torch.as_tensor(data.active,device=device))
    # Independent random initialization; parent architecture only, no weights.
    rng=np.random.default_rng(c['seed']);ix=rng.choice(data.split['train'],min(128,len(data.split['train'])),p=data.weights['train'])
    patch=torch.unique(data.indices[torch.as_tensor(ix,device=device)].flatten())[:2048]
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z,v=model.encode(data.positions[patch])
    model.scalar_mean.copy_(z.mean(0));model.scalar_scale.copy_(z.std(0,unbiased=False).clamp_min(1e-4));model.vector_scale.copy_(v.square().mean((0,2)).sqrt().clamp_min(1e-4))
    if not data.feature_task:
        ids=data.split['train'];p=np.bincount(data.target[torch.as_tensor(ids,device=device)].cpu().numpy(),weights=data.weights['train'],minlength=data.outputs)+1e-8;p/=p.sum()
        with torch.no_grad():model.readout[-1].weight.mul_(.01);model.readout[-1].bias.copy_(torch.tensor(np.log(p),device=device))
    else:
        with torch.no_grad():model.readout[-1].weight.mul_(.01);model.readout[-1].bias.zero_()
    if preflight:
        model.train();b=data.batch(rng.choice(data.split['train'],c['batch_size'],p=data.weights['train']))
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b)
        loss=data.loss(out['prediction'],b['target']).mean()+vcreg(out,c['regularization'])[0]
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite control preflight')
        return dict(arm=name,finite=True,loss=float(loss.detach()),gradient_norm=float(norm),outputs=data.outputs,
                    batch=c['batch_size'],parameters=sum(p.numel() for p in model.parameters()))
    binding=dict(config=c,arm=arm,dataset=data.manifest['identity'],code={p.name:sha(p) for p in Path(__file__).parent.glob('control_*.py')},
        execution=implementation_hashes('src/experiment_runner/artifacts.py',
            'src/experiment_runner/array_exports.py','src/experiment_runner/checkpoints.py',
            'src/experiment_runner/metric_docs.py','src/experiment_runner/wandb_tracking.py'))
    identity=digest(binding)
    if (tech/'identity.json').exists() and config(tech/'identity.json')!=binding:raise ValueError('Control fit identity changed')
    write_json(tech/'identity.json',binding)
    settings=dict(c,branch='self_supervised' if data.feature_task else 'synthetic_control',
                  wandb=dict(c['wandb'],display_name=f'Liquid information | {name} | MACE128'))
    study=SimpleNamespace(root=root,technical=tech,identity=identity,config=settings)
    write_json(tech/'prediction-context.json',dict(encoder=dict(geometry_only=True,channels=128,embedding=128,radius_A=8,nearest_candidates=80,
        species=False,history=False,motion=False,conditions=[],initialization='scratch'),
        predictor=dict(patches=25,relative_offsets=True,vector_messages=2,exported_context_dimension=128,conditions=[],
        target_features_as_inputs=False),observation=arm['domain'],target=arm['task'],
        target_description='all 3536 rich context features, equal weight per family' if data.feature_task else 'synthetic distance distribution',
        teacher='fixed analytic rich descriptors' if data.feature_task else None,
        full_cell_relaxation_has_external_context=arm['domain']=='relaxed',
        fixed_dataset=c['fixed_dataset'],batch=c['batch_size'],tracking='online scientific encoder training'))
    if data.feature_task:np.savez(tech/'target-standardization.npz',mean=data.mean,scale=data.scale,active=data.active)
    enc_ids={id(p) for p in model.encoder.parameters()}|{id(p) for p in model.vector_export.parameters()}
    groups=[dict(params=[p for p in model.parameters() if id(p) in enc_ids],lr=c['training']['encoder_lr']),
            dict(params=[p for p in model.parameters() if id(p) not in enc_ids],lr=c['training']['head_lr'])]
    opt=torch.optim.AdamW(groups,weight_decay=c['training']['weight_decay'],fused=True)
    epoch = step = update = 0
    best = float('inf')
    if (tech/'last.pt').exists():
        state = TrainingState.read(tech/'last.pt', identity=identity, device=device)
        state.restore(model, opt, rng, restore_torch_rng=True)
        saved = state.payload
        epoch, step, update, best = (saved[k] for k in ('epoch', 'step', 'update', 'best'))
    compile_model(model,data,c)
    def save(path):
        TrainingState.capture(
            model, opt, rng, identity=identity, capture_torch_rng=True,
            encoder=model.encoder.state_dict(), encoder_config=c['encoder_config'],
            config=c, arm=arm, epoch=epoch, step=step, update=update, best=best,
        ).save(path)
    total = c['training']['blocks'] * c['training']['updates_per_block']
    stop = deadline(c)
    if (tech/'complete.json').exists():
        model.load_state_dict(torch.load(tech/'best.pt',map_location=device,weights_only=False)['model']);summary=export(model,data,c,study)
        update_training_summary(study,'fit',summary,evaluation='rich-feature-prediction' if data.feature_task else 'synthetic-sensitivity')
        return
    tracking=tracked_run(study,'fit',job_type='encoder')
    with tracking as log:
        log.summary.update({'data/train_rows':len(data.split['train']),'model/parameters':sum(p.numel() for p in model.parameters()),
            'training/batch_size':c['batch_size'],'training/objective':'family-balanced fixed-variance Gaussian NLL' if data.feature_task else 'categorical NLL',
            'checkpoint/selector':'validation target likelihood; VCReg excluded'})
        while epoch<c['training']['blocks']:
            model.train()
            started = time.time()
            while step<c['training']['updates_per_block']:
                if time.time()>stop:
                    save(tech/'last.pt')
                    write_json(tech/'state.json', dict(state='checkpointed', update=update))
                    return
                ids = rng.choice(data.split['train'], c['batch_size'], p=data.weights['train'])
                b = data.batch(ids)
                opt.zero_grad(set_to_none=True)
                factor=min((update+1)/128,1)*(.05+.95*.5*(1+math.cos(math.pi*update/total)))
                for group,base in zip(opt.param_groups,(c['training']['encoder_lr'],c['training']['head_lr'])):group['lr']=base*factor
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b)
                nll = data.loss(out['prediction'], b['target']).mean()
                reg = vcreg(out, c['regularization'])[0] * min((update+1)/512, 1)
                loss = nll + reg
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite control loss {name}/{update}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                opt.step()
                step += 1
                update += 1
                if update%16==0:
                    record=dict(optimizer_update=update,**{'train/target_nll':float(nll.detach()),'train/vcreg':float(reg.detach()),'train/gradient_norm':float(norm)})
                    log.log(record)
                    with (tech/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    write_json(tech/'state.json',dict(state='training',update=update,total_updates=total))
                if update%64==0:save(tech/'last.pt')
            score = validation(model, data, c)
            epoch += 1
            step = 0
            if not np.isfinite(score):raise FloatingPointError('Nonfinite validation score')
            if score < best:
                best = score
                save(tech/'best.pt')
                log.summary['checkpoint/selected_block'] = epoch
            save(tech/'last.pt')
            save(tech/f'block-{epoch:02d}.pt')
            record=dict(optimizer_update=update,block=epoch,seconds=time.time()-started,**{'validation/target_nll':score})
            log.log(record)
            with (tech/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        write_json(tech/'complete.json',dict(identity=identity,updates=update,best_sha256=sha(tech/'best.pt')))
        model.load_state_dict(torch.load(tech/'best.pt',map_location=device,weights_only=False)['model']);log.summary.update(export(model,data,c,study))
        write_json(tech/'state.json',dict(state='complete',update=update))
