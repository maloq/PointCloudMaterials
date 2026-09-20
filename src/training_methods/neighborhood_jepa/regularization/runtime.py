"""Matched single-GPU fits with full-batch losses, gradient replay and immutable budgets."""
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.data.structural_pretraining.batches import move
from src.training_methods.shared_pretraining.compilation import compile_encoder,compilation_counters
from src.experiment_runner.metric_docs import write_metric_table
from ..v2.runtime import training_step,fixed_baselines,evaluate as base_evaluate,identity as base_identity
from ..v2.data import Batches
from ..v2.parallel import configure_host
from ..runtime import encode
from .data import Data,loader
from .model import Model
from .objective import Objective,epiplexity


@torch.no_grad()
def calibrate_epi(model,objective,data,config):
    values=[];model.eval()
    for packed,target in loader(data,Batches(data,config['batch_size'],config['seed']+41,0,4),config['microbatch']):
        z=encode(model,packed,config['precision']).reshape(len(target['group']),len(data.plan.views),-1)
        p=model.projector(z[:,data.plan.slot(1,0),:128])
        values.append(epiplexity(p,target['reservoir'][:,0].cuda()))
    scale=torch.stack(values).mean()
    if not torch.isfinite(scale) or scale<1e-6:raise ValueError(f'Epi initial score is degenerate: {scale}')
    objective.epi_initial_scale.copy_(scale)


@torch.no_grad()
def evaluate(model,objective,data,config,baselines):
    metrics,arrays=base_evaluate(model,objective,data,config,baselines)
    target=np.stack([data.order_arrays[data.rows[i][0]['id']][data.rows[i][1],0] for i in arrays['index']])
    z=torch.tensor(arrays['invariant'],device='cuda')
    pred=model.order_decoder(z)
    normalized=(torch.tensor(target,device='cuda')-objective.order_mean)/objective.order_std
    error=(pred-normalized).square().cpu().numpy();sources=arrays['sources']
    order=np.mean([error[sources==s].mean(0) for s in np.unique(sources)],0)
    metrics['order']=float(order.mean());metrics['order_components']=dict(zip(data.order_manifest['names'],map(float,order)))
    # All arms share the original selection criterion, including the no-order control.
    # Order/future scores and test probes do not silently alter checkpoint selection.
    for label,mask in [('noncrystalline',~np.isin(arrays['ptm_phase'],[1,2,3])),('crystalline',np.isin(arrays['ptm_phase'],[1,2,3]))]:
        if mask.any():metrics[label]['order']=float(np.mean([error[mask&(sources==s)].mean() for s in np.unique(sources[mask])]))
    for label,mask in [('all',np.ones(len(z),bool)),('noncrystalline',~np.isin(arrays['ptm_phase'],[1,2,3]))]:
        for name in ('invariant','projected'):
            ranks={}
            for temperature in np.unique(arrays['temperature_K'][mask]):
                selected=arrays[name][mask&(arrays['temperature_K']==temperature)]
                spectrum=np.linalg.eigvalsh(np.cov(selected,rowvar=False)).clip(0)
                ranks[str(int(temperature))]=float(spectrum.sum()**2/max(np.square(spectrum).sum(),1e-20))
            metrics[f'{label}_{name}_rank_by_temperature']=ranks
    arrays.update(order_target=target,order_prediction=(pred*objective.order_std+objective.order_mean).cpu().numpy(),order_errors=error)
    return metrics,arrays


def run(config,spec,deadline):
    configure_host();torch.set_num_threads(1)
    root=resolve_path(config['output'])/'technical/runs'/spec['name'];root.mkdir(parents=True,exist_ok=True)
    data=Data(config,spec);metadata=base_identity(dict(config,updates=spec['updates']),spec,data)
    metadata.update(order_identity=data.order_manifest['identity'],
        initialization='same frozen v2 development-selected weights; new projector/order head and optimizer' if spec['initialization']=='warm' else spec['initialization'],
        export_normalization='Per-observation LayerNorm; no BatchNorm or train/eval-dependent normalization in the encoder')
    metadata['files'].update({str(p):file_hash(p) for p in Path(__file__).parent.glob('*.py')})
    init_path=resolve_path(spec.get('checkpoint',config['warm_checkpoint']))
    if spec['initialization']!='scratch':metadata['initial_checkpoint_sha256']=file_hash(init_path)
    if spec['regularizer']=='epi':metadata['reservoir_manifest_sha256']=file_hash(data.extra/'reservoir.json')
    run_id=digest(metadata);manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=metadata:raise ValueError('Immutable regularization run changed')
    save_json(manifest_path,metadata)
    torch.manual_seed(config['seed']);model=Model(config['encoder_channels'],spec,config['seed']).cuda()
    objective=Objective(data.manifest,data.order_manifest,spec).cuda()
    last=root/'last.pt';saved=torch.load(last,map_location='cuda',weights_only=False) if last.exists() else None
    if saved:
        if saved['identity']!=run_id:raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model']);objective.load_state_dict(saved['objective'])
    elif spec['initialization']=='warm':
        model.initialize(torch.load(init_path,map_location='cpu',weights_only=False))
    elif spec['initialization']=='continuation':
        parent=torch.load(init_path,map_location='cpu',weights_only=False)
        model.load_state_dict(parent['model']);objective.load_state_dict(parent['objective'])
    elif spec['initialization']=='scratch':pass  # Constructor initialization, no learned parent weights.
    else:raise ValueError(spec['initialization'])
    model.encoder.geometry_scales.copy_(torch.tensor(data.manifest['geometry_scales'],device='cuda'))
    example=next(iter(loader(data,[data.train[:2]],config['microbatch'])))[0][0]
    if config['compile']:compile_encoder(model.encoder,move(example,'cuda'),config['precision'])
    if not saved and spec['regularizer']=='epi' and spec['initialization']!='continuation':calibrate_epi(model,objective,data,config)
    encoder=list(model.encoder.parameters());eid={id(p) for p in encoder};heads=[p for p in model.parameters() if id(p) not in eid]
    optimizer=torch.optim.AdamW([dict(params=encoder,peak=spec['encoder_lr']),dict(params=heads,peak=spec['head_lr'])],weight_decay=1e-4)
    step,best=0,float('inf')
    if saved:
        optimizer.load_state_dict(saved['optimizer']);step,best=saved['step'],saved['best']
        torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    baselines=fixed_baselines(data);stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
    def checkpoint(name):
        temporary=root/(name+'.building')
        torch.save(dict(identity=run_id,manifest=metadata,spec=spec,model=model.state_dict(),objective=objective.state_dict(),
            optimizer=optimizer.state_dict(),step=step,best=best,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)
        temporary.replace(root/name)
    wb=None
    if config['wandb']:
        import wandb
        wb=wandb.init(entity='teshbek',project='PointCloudMaterials',group=config['wandb_group'],
                     id='nj-reg-'+run_id[:16],name=spec['name'],resume='allow',dir=str(root),config=metadata)
    def validate():
        nonlocal best
        metrics,arrays=evaluate(model,objective,data,config,baselines);metrics['step']=step
        with (root/'validation.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        if metrics['selection_score']<best:
            best=metrics['selection_score'];checkpoint('best.pt')
            np.savez_compressed(root/'selection_predictions.npz',**arrays);save_json(root/'metrics.json',metrics)
        if wb:wb.log({f'validation/{k}':metrics[k] for k in ('selection_score','order','future_physical','invariant_effective_rank','projected_effective_rank')},step=step)
    if not saved:validate();checkpoint('last.pt')
    started=time.monotonic()
    stream=loader(data,Batches(data,config['batch_size'],config['seed'],step,spec['updates']),config['microbatch'],config['loader_workers'])
    for packed,target in stream:
        if stop or time.time()>deadline-240:
            checkpoint('last.pt');save_json(root/'status.json',dict(state='checkpointed',step=step,reason='allocation_deadline_or_signal'))
            if wb:wb.finish()
            return False
        model.train();objective.train();optimizer.zero_grad(set_to_none=True)
        warm=max(1,int(spec['updates']*.1));factor=min((step+1)/warm,1.) if step<warm else .01+.99*.5*(1+math.cos(math.pi*(step-warm)/(spec['updates']-warm)))
        for group in optimizer.param_groups:group['lr']=group['peak']*factor
        loss,terms,diag=training_step(model,objective,packed,move(target,'cuda'),config['precision'],step%128==0)
        diag['encoder_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(encoder,1.,error_if_nonfinite=True))
        diag['head_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True));optimizer.step();step+=1
        if step%16==0 or step==1:
            record=dict(state='running',step=step,updates=spec['updates'],loss=loss,terms=terms,diagnostics=diag,
                epoch_equivalents=step*config['batch_size']/data.train_size,seconds=time.monotonic()-started)
            save_json(root/'status.json',record)
            with (root/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if wb:wb.log({'loss/total':loss,**{f'loss/{k}':v for k,v in terms.items()},**{f'regularizer/{k}':v for k,v in diag.items() if k in ('vicreg_total','vicreg_variance','vicreg_covariance','epi_score','sigreg_discrepancy')}},step=step)
        if step%config['evaluate_every']==0 or step==spec['updates']:validate();checkpoint('last.pt')
    metrics=json.loads((root/'metrics.json').read_text());metrics.update(total_updates=step,anchor_draws=step*config['batch_size'])
    write_metric_table(metrics,resolve_path(config['output']),family=config['metric_family'],name=spec['name'])
    save_json(root/'compilation.json',compilation_counters());save_json(root/'status.json',dict(state='complete',step=step,selection_score=best,identity=run_id))
    if wb:wb.finish()
    return True
