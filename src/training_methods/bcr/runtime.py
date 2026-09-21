"""FP32 training with exact stream/noise/optimizer replay and explicit launch gates."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .model import BCR
from .data import BalancedStream,pack,corrupt
from .objective import per_environment,vicreg


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def implementation_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}


def load_data(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if hashlib.sha256((path/'patches.npz').read_bytes()).hexdigest()!=manifest['patches_sha256']:raise ValueError('Patch cache hash changed')
    with np.load(path/'patches.npz') as a:patches=[a['positions'][a['offsets'][i]:a['offsets'][i+1]].copy() for i in range(len(a['offsets'])-1)]
    return patches,manifest


def checkpoint_steps(updates):return {0,updates,*[max(1,round(updates*f)) for f in (.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9)]}


def lr_at(step,total,peak=3e-4,minimum=3e-6):
    warm=max(1,round(.05*total))
    if step<warm:return peak*(step+1)/warm
    return minimum+(peak-minimum)*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))


def save(path,payload):
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def train(config,data_path,output,device='cpu',stop_after=None,deadline=None):
    if config.get('precision','float32')!='float32' or config.get('compile',False):
        raise ValueError('BCR-v1 deploys tested eager FP32 only; mixed/compiled execution requires a separate parity release')
    patches,manifest=load_data(data_path);records=manifest['records'];indices=[i for i,r in enumerate(records) if r['split']=='train']
    if not indices:raise ValueError('No training roots')
    if config['encoder']['d0']!=manifest['d0'] or config['encoder']['n_ref']!=manifest['n_ref'] or config['encoder']['radius']!=manifest['radius_A']:
        raise ValueError('Encoder calibration differs from frozen training manifest')
    config=dict(config,data_identity=manifest['identity']);cfgid=identity(config)
    if config['updates']>256:
        gate=json.loads(Path(config['gate_receipt']).read_text())
        if (not gate['G0_pass'] or not gate['real_overfit_pass'] or gate['model_contract']!=identity(config['encoder'])
            or gate['decoder_contract']!=identity(config.get('decoder',{}))
            or gate['model_sha256']!=hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()
            or gate['implementation_hashes']!=implementation_hashes()
            or gate['correctness_suite_sha256']!=hashlib.sha256(Path('tests/test_bcr.py').read_bytes()).hexdigest()):
            raise ValueError('Large launch requires passing G0 and real overfit for this model contract')
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);torch.set_num_threads(config.get('threads',1))
    model=BCR(config).to(device);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config.get('lr',3e-4),betas=(.9,.95),weight_decay=1e-5)
    stream=BalancedStream([records[i] for i in indices],config['seed']+101);noise=torch.Generator().manual_seed(config['seed']+211)
    root=Path(output);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    start=0;counts=np.zeros(len(manifest['noise_levels']),dtype=np.int64)
    if (tech/'last.pt').exists():
        old=torch.load(tech/'last.pt',map_location=device,weights_only=False)
        if old['config_identity']!=cfgid or old['data_identity']!=manifest['identity']:raise ValueError('Resume contract/data changed')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);stream.load_state_dict(old['stream']);noise.set_state(old['noise_rng'].cpu())
        torch.set_rng_state(old['torch_rng'].cpu());np.random.set_state(old['numpy_rng']);start=old['step'];counts=np.array(old['level_counts'])
        if device!='cpu':torch.cuda.set_rng_state_all([s.cpu() for s in old['cuda_rng']])
    def payload(step):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),stream=stream.state_dict(),noise_rng=noise.get_state(),
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device!='cpu' else [],
            step=step,config=config,config_identity=cfgid,data_identity=manifest['identity'],level_counts=counts.tolist())
    if start==0:save(tech/'initial.pt',payload(0))
    environment=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions={p:importlib.metadata.version(p) for p in ('torch','e3nn','mace-torch','numpy')},
        total_parameters=sum(p.numel() for p in model.parameters()),active_parameters=sum(p.numel() for p in params),
        tensor_product_paths=[b.paths for b in model.decoder.blocks],config=config,data_identity=manifest['identity'],
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (tech/'manifest.json').write_text(json.dumps(environment,indent=2)+'\n')
    begin=time.monotonic();model.train();completed=start;limit=min(config['updates'],stop_after or config['updates'])
    for step in range(start,limit):
        if deadline is not None and time.time()>deadline-120:
            save(tech/'last.pt',payload(completed));break
        selection=stream.draw(config['batch_size']);ids=[indices[i] for i in selection]
        clean=pack([patches[i] for i in ids],device);noisy,epsilon,sigma,levels=corrupt(clean,manifest['noise_levels'],manifest['d0'],noise)
        counts+=np.bincount(levels.cpu().numpy(),minlength=len(counts));optimizer.zero_grad(set_to_none=True)
        lr=lr_at(step,config['updates'],config.get('lr',3e-4),config.get('min_lr',3e-6))
        for group in optimizer.param_groups:group['lr']=lr
        total=0.;gradz=0.;logs={};micro=config.get('microbatch',config['batch_size'])
        if model.arm=='vicreg':
            # Full statistical batch. Microbatch-wise variance/covariance would
            # change the comparator objective and is deliberately not supported.
            loss=vicreg(model.encode(clean),model.encode(noisy));loss.backward();total=float(loss.detach())
        else:
            values=[];shells={name:[] for name in ('interior','middle','outer','unweighted')};codes=[]
            for first in range(0,len(ids),micro):
                sl=slice(first,first+micro);a={k:v[sl] for k,v in clean.items()};b={k:v[sl] for k,v in noisy.items()}
                pred,z=model(a,b,sigma[sl]);losses=per_environment(pred,epsilon[sl],a,manifest['radius_A'])
                if z.requires_grad:z.retain_grad()
                loss=losses.sum()/len(ids)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite noise objective: update {step}, anchors {ids[first:first+micro]}')
                loss.backward();total+=float(loss.detach());values.extend(losses.detach().cpu().tolist())
                if z.grad is not None:gradz+=float(z.grad.norm())
                codes.append(z.detach().cpu())
                for name in shells:shells[name].extend(per_environment(pred.detach(),epsilon[sl],a,manifest['radius_A'],name).cpu().tolist())
            for level in range(len(counts)):
                rows=levels.cpu().numpy()==level
                if rows.any():logs[str(manifest['noise_levels'][level])]=float(np.array(values)[rows].mean())
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        measure_update=step%50==0
        before={name:p.detach().clone() for name,p in model.named_parameters()} if measure_update else {}
        optimizer.step();completed=step+1
        record=dict(step=step+1,loss=total,by_sigma=logs,gradient_norm=float(norm),code_gradient_norm=gradz,lr=lr,
            level_counts=counts.tolist(),anchor_exposures=stream.exposures,seconds=time.monotonic()-begin)
        if measure_update:
            record['relative_update']={}
            for label in ('encoder','decoder'):
                names=[n for n in before if n.startswith(label+'.')]
                numerator=sum((p.detach()-before[n]).square().sum() for n,p in model.named_parameters() if n in names)
                denominator=sum(before[n].square().sum() for n in names)
                record['relative_update'][label]=float(torch.sqrt(numerator/denominator.clamp_min(1e-30)))
        if model.arm!='vicreg':
            record['by_source']={str(src):float(np.mean([v for v,i in zip(values,ids) if records[i]['source']==src])) for src in {records[i]['source'] for i in ids}}
            record['by_shell']={k:float(np.nanmean(v)) if np.isfinite(v).any() else None for k,v in shells.items()}
            if step%50==0:
                from .evaluate import code_statistics
                record['code']=code_statistics(torch.cat(codes))
        with (tech/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step+1 in checkpoint_steps(config['updates']) or step+1==limit:
            state=payload(step+1);save(tech/'last.pt',state);save(tech/f'step-{step+1:06d}.pt',state)
    (tech/'status.json').write_text(json.dumps(dict(state='complete' if completed==config['updates'] else 'checkpointed',step=completed,
        selection='pending structural retention and robustness gates; no best checkpoint claimed'),indent=2))
    return model
