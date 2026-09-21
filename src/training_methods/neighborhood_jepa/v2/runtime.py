"""Immutable v2 fits with exact full-statistical-batch encoder gradient replay."""
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json, digest, file_hash
from src.data.structural_pretraining.batches import move
from src.training_methods.shared_pretraining.compilation import compile_encoder, compilation_counters
from src.training_methods.structural_pretraining.objective import block_errors, PHYSICAL_BLOCKS
from src.experiment_runner.metric_docs import write_metric_table
from ..runtime import encode
from .model import Model
from .objective import Objective
from .data import Data, Batches, loader
from .contracts import LAYOUT
from .geometry import scaled_error, blocks


def identity(config,spec,data):
    import importlib.metadata as metadata
    modules = [Path(__file__).parent,Path('src/models/encoders'),Path('src/data/structural_pretraining')]
    files = {str(p):file_hash(p) for directory in modules for p in sorted(directory.glob('*.py'))}
    for p in (Path('src/training_methods/neighborhood_jepa/model.py'),Path('src/training_methods/neighborhood_jepa/runtime.py')):
        files[str(p)] = file_hash(p)
    return dict(protocol=config['protocol'],config=config,spec=spec,data_identity=data.manifest['identity'],
        layout=LAYOUT.metadata(),normalization='raw independent snapshot export in train/eval/heads',
        dependencies={name:metadata.version(name) for name in ('torch','e3nn','mace-torch','cuequivariance','lejepa')},
        files=files,sampling='uniform native-Al training anchors without replacement within each update',
        selection='minimum source-equal present Physical85 + .25 TDA144; development only',
        seed=config['seed'],updates=config['updates'],anchor_draws=config['updates']*config['batch_size'])


def training_step(model,objective,batches,target,precision,diagnose=False):
    encoded = encode(model,batches,precision).detach().requires_grad_(True)
    loss,terms = objective(model,encoded,target)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite v2 objective: {terms}')
    diagnostics = dict(objective.diagnostics)
    if diagnose:
        for name,value in terms.items():
            gradient = torch.autograd.grad(value,encoded,retain_graph=True)[0]
            diagnostics[f'export_gradient/{name}'] = float(gradient.norm())
    loss.backward()
    offset = 0
    for batch in batches:
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            replay = model.encoder(move(batch,'cuda')).float()
        replay.backward(encoded.grad[offset:offset+len(replay)])
        offset += len(replay)
    if offset != len(encoded):
        raise ValueError('Replay omitted an encoded observation')
    return float(loss.detach()),{k:float(v.detach()) for k,v in terms.items()},diagnostics


def fixed_baselines(data):
    """Train-only physical mean, geometry ridge and scalar mean reversion per temperature."""
    arrays,temps = [],[]
    for record in data.manifest['shards']:
        if record['split'] != 'train':
            continue
        value = np.load(data.parent/'shards'/record['id']/'physical.npy').astype(np.float64)
        arrays.append(value)
        temps.extend([record['temperature_K']]*len(value))
    raw = np.concatenate(arrays)
    norm = data.manifest['normalization']['physical']
    p = (raw-np.array(norm['mean']))/np.array(norm['std'])
    temperatures = np.asarray(temps)
    result = {}
    for temperature in np.unique(temperatures):
        x,y = p[temperatures==temperature,0],p[temperatures==temperature,1]
        mean = y.mean(0)
        design = np.c_[x,np.ones(len(x))]
        penalty = np.eye(86)*len(x)*.01
        penalty[-1,-1] = 0
        ridge = np.linalg.solve(design.T@design+penalty,design.T@y)
        alpha = np.sum((x-mean)*(y-mean))/max(np.sum((x-mean)**2),1e-12)
        result[int(temperature)] = dict(mean=mean,ridge=ridge,alpha=alpha)
    return result


def selection_phases(data, config):
    """Resolve immutable development labels once per dataset instance."""
    if hasattr(data, '_selection_phases'):
        return data._selection_phases
    phase = []
    assay = json.loads(resolve_path(config['crystallization_plan']).read_text())
    cache = resolve_path(assay['config']['cache'])
    source_lookup = {s['lineage']:s for s in assay['sources']}
    source_arrays, query_arrays = {}, {}
    for index in data.selection:
        record,row = data.rows[index]
        source = source_lookup[record['lineage']]['id']
        if source not in source_arrays:
            folder = cache/str(source)
            source_arrays[source] = (np.load(folder/'atom_ids.npy'), np.load(folder/'labels.npy',mmap_mode='r'))
        atom_ids, labels = source_arrays[source]
        if record['id'] not in query_arrays:
            query_arrays[record['id']] = np.load(data.parent/'shards'/record['id']/'query_atom_ids.npy',mmap_mode='r')
        query_id = query_arrays[record['id']][row,0]
        center = np.flatnonzero(atom_ids==query_id)
        if len(center)!=1:
            raise ValueError(f'Development phase label does not match tracked center: source={source}, shard={record["id"]}, row={row}, atom={query_id}')
        phase.append(int(labels[center[0],record['frame']]))
    data._selection_phases = np.array(phase)
    return data._selection_phases


@torch.no_grad()
def evaluate(model,objective,data,config,baselines):
    model.eval()
    objective.eval()
    ids = data.selection
    batches = [ids[i:i+16] for i in range(0,len(ids),16)]
    records = []
    phase = selection_phases(data, config)
    outputs = {k:[] for k in ('invariant','equivariant','projected','physical_target','physical_prediction','future_prediction','index','temperature_K','query_atom_ids','frame','geometry_by_degree','eq_block_rms','target_eq_block_rms','baseline_predictions')}
    names = ['physical','tda','geometry','future_physical','physical_persistence','condition_mean','geometry_ridge','mean_reversion']
    for packed,target in loader(data,batches,config['microbatch'],view_slots=[data.plan.slot(1,0)]):
        target = move(target,'cuda')
        current = encode(model,packed,config['precision'])
        group = target['group']
        p = objective.physical_errors(model,current[:,:128],target['physical'][:,0],group)
        t = objective.tda_errors(model,current[:,:128],target['tda'][:,0],group)
        geom = scaled_error(current[:,128:],target['moments'][:,data.plan.slot(1,0)],model.encoder.geometry_scales).mean(-1)
        normalized = (target['physical']-objective.physical_mean)/objective.physical_std
        # All arms have the same predictor; A's untrained future head is explicitly diagnostic.
        conditioned = torch.cat((current[:,:128]+model.condition((target['temperature_K'][:,None]-460.)/100.),current[:,128:]),-1)
        predicted,_ = model.predict(conditioned,None,target['position'][:,:1],target['times'][:,2:3])
        fpred = model.physical(predicted[:,0],group)
        error = lambda x:block_errors(x,normalized[:,1],PHYSICAL_BLOCKS).mean(-1)
        reference = []
        for row,temp in zip(normalized[:,0].cpu().numpy(),target['temperature_K'].cpu().tolist()):
            fit = baselines[int(temp)]
            reference.append([fit['mean'],np.r_[row,1.]@fit['ridge'],fit['mean']+fit['alpha']*(row-fit['mean'])])
        reference = torch.tensor(np.array(reference),device='cuda',dtype=torch.float32)
        records.append(torch.stack((p,t,geom,error(fpred),error(normalized[:,0]),
                                    error(reference[:,0]),error(reference[:,1]),error(reference[:,2])),-1).cpu().numpy())
        for name,value in dict(invariant=current[:,:128],equivariant=current[:,128:],projected=model.projector(current[:,:128]),
                physical_target=normalized,physical_prediction=model.physical(current[:,:128],group),future_prediction=fpred,
                geometry_by_degree=scaled_error(current[:,128:],target['moments'][:,data.plan.slot(1,0)],model.encoder.geometry_scales),
                eq_block_rms=torch.stack([b.square().mean(-1).sqrt() for b in blocks(current[:,128:])],1),
                target_eq_block_rms=torch.stack([b.square().mean(-1).sqrt() for b in blocks(target['moments'][:,data.plan.slot(1,0)])],1),
                baseline_predictions=reference,
                **{k:target[k] for k in ('index','temperature_K','query_atom_ids','frame')}).items():
            outputs[name].append(value.cpu().numpy())
    values = np.concatenate(records)
    source = np.array([data.rows[i][0]['source'] for i in ids])
    roots = np.array([data.rows[i][0]['lineage'] for i in ids])
    average = np.mean([values[source==s].mean(0) for s in np.unique(source)],0)
    metrics = dict(zip(names,map(float,average)))
    metrics.update(selection_score=metrics['physical']+.25*metrics['tda'],windows=len(ids),sources=len(np.unique(source)),
        future_head_trained=bool(objective.spec['future_weight']))
    arrays = {k:np.concatenate(v) for k,v in outputs.items()}
    arrays.update(errors=values,error_names=np.array(names),sources=source,lineages=roots,ptm_phase=np.array(phase))
    for label,mask in [('noncrystalline',~np.isin(phase,[1,2,3])),('crystalline',np.isin(phase,[1,2,3]))]:
        if mask.any():
            subset = np.mean([values[mask & (source==sid)].mean(0) for sid in np.unique(source[mask])],0)
            metrics[label] = dict(zip(names,map(float,subset)),windows=int(mask.sum()),sources=len(np.unique(source[mask])))
    metrics['per_degree_geometry'] = {str(ell):float(arrays['geometry_by_degree'][:,i].mean()) for i,ell in enumerate(LAYOUT.degrees)}
    for name in ('invariant','projected'):
        spectrum = np.linalg.eigvalsh(np.cov(arrays[name],rowvar=False)).clip(0)
        metrics[f'{name}_covariance_trace'] = float(spectrum.sum())
        metrics[f'{name}_effective_rank'] = float(spectrum.sum()**2/max(np.square(spectrum).sum(),1e-20))
        arrays[f'{name}_spectrum'] = spectrum
    return metrics,arrays


def run(config,spec,deadline):
    if config.get('gpu_devices') == [0,1]:
        from .parallel import configure_host
        configure_host()
    root = resolve_path(config['output'])/'technical/runs'/spec['name']
    root.mkdir(parents=True,exist_ok=True)
    data = Data(resolve_path(config['cache']),spec)
    metadata = identity(config,spec,data)
    run_id = digest(metadata)
    manifest_path = root/'manifest.json'
    if manifest_path.exists() and digest(json.loads(manifest_path.read_text())) != run_id:
        raise ValueError('Immutable v2 run manifest changed; refusing resume')
    save_json(manifest_path,metadata)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    model = Model(channels=config.get('encoder_channels',32)).cuda()
    objective = Objective(data.manifest,spec).cuda()
    model.encoder.geometry_scales.copy_(torch.tensor(data.manifest['geometry_scales'],device='cuda'))
    last = root/'last.pt'
    saved = torch.load(last,map_location='cuda',weights_only=False) if last.exists() else None
    if saved:
        if saved['identity'] != run_id:
            raise ValueError('V2 checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        objective.load_state_dict(saved['objective'])
    first = next(iter(loader(data,[data.train[:2]],config['microbatch'])))
    if config['compile']:
        compile_encoder(model.encoder,move(first[0][0],'cuda'),config['precision'])
    parallel = None
    if config.get('gpu_devices') == [0,1]:
        from .parallel import ParallelEncoder
        parallel = ParallelEncoder(model,first[0][0],config['precision'],config['compile'])
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    heads = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([dict(params=encoder,peak=spec['encoder_lr']),
                                  dict(params=heads,peak=spec['head_lr'])],weight_decay=1e-4)
    step,best = 0,float('inf')
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        step,best = saved['step'],saved['best']
        torch.set_rng_state(saved['rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    if parallel is not None:
        parallel.synchronize()
    baselines = fixed_baselines(data)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGUSR1,request_stop)
    def checkpoint(name):
        payload = dict(identity=run_id,manifest=metadata,spec=spec,model=model.state_dict(),objective=objective.state_dict(),
            optimizer=optimizer.state_dict(),step=step,best=best,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        temporary = root/(name+'.building')
        torch.save(payload,temporary)
        temporary.replace(root/name)
    wb = None
    if config['wandb']:
        import wandb
        wb = wandb.init(entity='teshbek',project='PointCloudMaterials',group=config['wandb_group'],
            id='nj-v2-'+run_id[:16],name=spec['name'],resume='allow',dir=str(root),config=metadata)
    def validate():
        nonlocal best
        metrics,arrays = evaluate(model,objective,data,config,baselines)
        metrics['step'] = step
        with (root/'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics)+'\n')
        if metrics['selection_score'] < best:
            best = metrics['selection_score']
            checkpoint('best.pt')
            torch.save(dict(encoder=model.encoder.state_dict(),architecture='neighborhood_mace_v2',
                            manifest=metadata,identity=run_id,step=step,layout=LAYOUT.metadata()),root/'encoder.pt')
            np.savez_compressed(root/'selection_predictions.npz',**arrays)
            save_json(root/'metrics.json',metrics)
        if wb:
            wb.log({f'validation/{k}':v for k,v in metrics.items()},step=step)
    if step == 0:
        validate()
        checkpoint('last.pt')
    started = time.monotonic()
    for packed,target in loader(data,Batches(data,config['batch_size'],config['seed'],step,config['updates']),config['microbatch'],config['loader_workers']):
        if stop or time.time()>deadline-180:
            checkpoint('last.pt')
            save_json(root/'status.json',dict(state='checkpointed',step=step))
            if parallel is not None: parallel.close()
            if wb: wb.finish()
            return False
        model.train()
        objective.train()
        optimizer.zero_grad(set_to_none=True)
        warm = max(1,int(config['updates']*.1))
        factor = min((step+1)/warm,1.) if step<warm else .01+.99*.5*(1+math.cos(math.pi*(step-warm)/(config['updates']-warm)))
        for group in optimizer.param_groups:
            group['lr'] = group['peak']*factor
        if parallel is None:
            loss,terms,diagnostics = training_step(model,objective,packed,move(target,'cuda'),config['precision'],step%128==0)
        else:
            loss,terms,diagnostics = parallel.step(model,objective,packed,move(target,'cuda'),step%128==0)
        diagnostics['encoder_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(encoder,1.,error_if_nonfinite=True))
        diagnostics['head_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True))
        optimizer.step()
        if parallel is not None: parallel.synchronize()
        step += 1
        if step%16==0 or step==1:
            record = dict(state='running',step=step,updates=config['updates'],loss=loss,terms=terms,
                diagnostics=diagnostics,epoch_equivalents=step*config['batch_size']/data.train_size,anchor_draws=step*config['batch_size'],
                encoded_observations=step*config['batch_size']*len(data.plan.views),seconds=time.monotonic()-started)
            save_json(root/'status.json',record)
            with (root/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if wb: wb.log({'loss/total':loss,**{f'loss/{k}':v for k,v in terms.items()}},step=step)
        if step%config['evaluate_every']==0 or step==config['updates']:
            validate()
            checkpoint('last.pt')
    metrics = json.loads((root/'metrics.json').read_text())
    metrics.update(total_updates=step,anchor_draws=step*config['batch_size'])
    write_metric_table(metrics,resolve_path(config['output']),family=config.get('metric_family','neighborhood_jepa_v2'),name=spec['name'])
    save_json(root/'compilation.json',compilation_counters())
    save_json(root/'status.json',dict(state='complete',step=step,selection_score=best,identity=run_id))
    if parallel is not None: parallel.close()
    if wb: wb.finish()
    return True
