"""Resumable fixed-budget transfer fits; cached frozen features and process-fed raw graphs."""
import copy
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.data.structural_pretraining.batches import collate,move
from src.models.encoders.structural import StructuralMACE
from src.training_methods.shared_pretraining.compilation import compile_encoder
from src.training_methods.shared_pretraining.runtime import configure
from src.research.local_predictability.metrics import hazard_loss,source_weights
from .data import Corpus,graph
from .model import Predictor
from .metrics import evaluate
from .training import configure_training,epoch_batch,weighted_microbatch_loss


def setup():
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    configure();torch.set_num_threads(1)


def parent_state(plan):
    path=resolve_path(plan['config']['output'])/'technical/parent.pt'
    if file_hash(path)!=plan['checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    p=torch.load(path,map_location='cpu',weights_only=False)
    return {k.removeprefix('encoder.'):v for k,v in p['model'].items() if k.startswith('encoder.')}


@torch.no_grad()
def extract(plan,source,model=None):
    root=resolve_path(plan['config']['cache'])/str(source['id']);receipt=root/'features.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['checkpoint_sha256']!=plan['checkpoint_sha256'] or file_hash(root/'features.npy')!=r['sha256']:raise ValueError('Feature cache changed')
        return model
    a={p.stem:np.load(p,mmap_mode='r') for p in root.glob('*.npy')}
    n=len(a['offsets'])-1;b=plan['config']['extraction_batch_size']
    if model is None:
        model=StructuralMACE().cuda().eval();model.load_state_dict(parent_state(plan),strict=True)
        example=move(collate([graph(a,0,plan['scale'])],'mace'),'cuda');compile_encoder(model,example,'bf16')
    result=np.lib.format.open_memmap(root/'features.building.npy',mode='w+',dtype=np.float32,shape=(n,416))
    def pack(start):return collate([graph(a,g,plan['scale']) for g in range(start,min(start+b,n))],'mace')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(pack,0)
        for start in range(0,n,b):
            batch=move(future.result(),'cuda')
            if start+b<n:future=pool.submit(pack,start+b)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(batch,return_equivariant=True).float()
            if z.shape!=(min(b,n-start),416) or not torch.isfinite(z).all():raise ValueError('Invalid tensor feature output')
            result[start:start+len(z)]=z.cpu().numpy()
    result.flush();del result;(root/'features.building.npy').replace(root/'features.npy')
    save_json(receipt,dict(checkpoint_sha256=plan['checkpoint_sha256'],sha256=file_hash(root/'features.npy'),graphs=n))
    return model


_worker=None
_spec=None
_config=None


def init_worker(plan,spec):
    global _worker,_spec,_config
    setup();_worker=Corpus(plan);_spec=spec;_config=plan['config']
    if 'training' in spec:configure_training(_worker,spec)


def pack(corpus,indices,spec):
    raw=spec['mode']!='frozen';batch=corpus.inputs(indices,spec,raw=raw)
    if raw:batch['graph_batch']=collate(batch.pop('graphs'),'mace')
    return batch


def prepare_update(step):
    rng=np.random.default_rng(np.random.SeedSequence([_config['seed'],step]))
    ids=epoch_batch(_worker,step) if 'training' in _spec else _worker.sample(rng,_config['batch_size'])
    m=_config['microbatch_size'] if _spec['mode']!='frozen' else len(ids);result=[]
    for k in range(0,len(ids),m):
        selected=ids[k:k+m];observation=_spec
        if _spec.get('protocol')=='adaptive_v1' and _spec['mode']=='finetune' and step<_spec['head_warmup_updates']:
            observation=dict(_spec,mode='frozen')
        batch=pack(_worker,selected,observation)
        batch['loss_weight']=torch.from_numpy(_worker.training_weights[selected].copy()) if 'training' in _spec else torch.ones(len(selected))
        result.append(batch)
    return result


def cuda(batch):
    return {k:move(v,'cuda') if isinstance(v,dict) else v.to('cuda',non_blocking=True) for k,v in batch.items()}


@torch.no_grad()
def predict(model,corpus,indices,spec):
    model.eval();values=[];size=16 if model.encoder is not None else 512
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(pack,corpus,indices[:size],spec)
        for start in range(0,len(indices),size):
            batch=cuda(future.result())
            if start+size<len(indices):future=pool.submit(pack,corpus,indices[start+size:start+2*size],spec)
            values.append(model(batch).cpu().numpy())
    model.train();return np.concatenate(values)


@torch.no_grad()
def normalize(model,corpus,spec,seed):
    ids=corpus.sample(np.random.default_rng(seed),2048)
    # Fixed train-only normalizers; tensor channels are contracted invariantly instead.
    b=corpus.inputs(ids,dict(spec,history_ps=0,radius_A=0),raw=False)
    z=b['features'][:,0,:128]
    if spec['mode']=='scratch':
        zs=[]
        for start in range(0,256,16):
            rb=cuda(pack(corpus,ids[start:start+16],dict(spec,history_ps=0,radius_A=0)))
            with torch.autocast('cuda',dtype=torch.bfloat16):zs.append(model.encoder(rb['graph_batch']).float().cpu())
        z=torch.cat(zs)
    model.head.mean.copy_(z.mean(0));model.head.scale.copy_(z.std(0,unbiased=False).clamp_min(1e-3))
    d=b['descriptor'];model.head.descriptor_mean.copy_(d.mean(0));model.head.descriptor_scale.copy_(d.std(0,unbiased=False).clamp_min(1e-3))
    events=corpus.events[ids];hazards=[]
    for k in range(6):hazards.append(np.clip((np.sum(events==k)+.5)/(np.sum(events>=k)+1),1e-4,1-1e-4))
    out=model.head.output if isinstance(model.head.output,torch.nn.Linear) else model.head.output[-1]
    out.bias.copy_(torch.tensor(np.log(np.array(hazards)/(1-np.array(hazards))),device='cuda',dtype=torch.float32))


def fit(plan,spec,deadline):
    if spec.get('protocol')=='adaptive_v1':
        from .adaptive import fit as fit_adaptive
        return fit_adaptive(plan,spec,deadline)
    setup();config=plan['config'];root=resolve_path(config['output'])/'technical/runs'/spec['name'];root.mkdir(parents=True,exist_ok=True)
    # Independent fits must not consume one another's per-code compiler guard budget.
    if spec['mode']!='frozen':torch._dynamo.reset()
    status=root/'status.json'
    if status.exists() and json.loads(status.read_text())['state']=='complete':return True
    if spec.get('baseline')=='persistence':
        corpus=Corpus(plan);cal=corpus.splits['calibration'];test=corpus.splits['test']
        metrics=evaluate(corpus,test,np.full((len(test),6),-30.,np.float32),cal,np.full((len(cal),6),-30.,np.float32))
        save_json(root/'metrics.json',metrics)
        from src.experiment_runner.metric_docs import write_metric_table
        flat={key:{str(r['horizon_ps']):r for r in metrics[key]} for key in ('classification','timing','spatial')}
        flat['test_event_nll']=metrics['test_event_nll']
        write_metric_table(flat,resolve_path(config['output']),family='crystallization_transfer',name=spec['name'])
        save_json(status,dict(state='complete',step=0,test_event_nll=metrics['test_event_nll']));return True
    stop=False
    def request_stop(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    torch.manual_seed(config['seed']);corpus=Corpus(plan);model=Predictor(spec,parent_state(plan)).cuda()
    updates=config['updates']
    if 'training' in spec:
        summary=configure_training(corpus,spec);updates=summary['updates']
        save_json(root/'training-population.json',summary)
        np.save(root/'training-indices.npy',np.array(corpus.splits['train'],dtype=np.int64))
    selected=[]
    for sid in np.unique(corpus.source_ids[corpus.splits['selection']]):
        ids=np.array(corpus.splits['selection']);ids=ids[corpus.source_ids[ids]==sid]
        rng=np.random.default_rng(np.random.SeedSequence([config['seed'],int(sid)]))
        selected.extend(sorted(rng.choice(ids,min(config['selection_per_source'],len(ids)),replace=False).tolist()))
    weights=source_weights(corpus.source_ids[selected]);events=torch.tensor(corpus.events[selected])
    params=[dict(params=model.head.parameters(),lr=config['head_lr'],base_lr=config['head_lr'])]
    if model.encoder is not None:params.append(dict(params=model.encoder.parameters(),lr=config['encoder_lr'],base_lr=config['encoder_lr']))
    optimizer=torch.optim.AdamW(params,weight_decay=1e-4);step=0;best=float('inf')
    if (root/'last.pt').exists():
        saved=torch.load(root/'last.pt',map_location='cuda',weights_only=False)
        if saved['identity']!=plan['identity'] or saved['spec']!=spec:raise ValueError('Training resume identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step=saved['step'];best=saved['best']
    else:normalize(model,corpus,spec,config['seed'])
    if model.encoder is not None:
        example=cuda(pack(corpus,corpus.splits['train'][:1],dict(spec,history_ps=0,radius_A=0)))['graph_batch']
        compile_encoder(model.encoder,example,'bf16')
    def save(name):
        path=root/name;temp=path.with_suffix('.building.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,spec=spec,identity=plan['identity']),temp);temp.replace(path)
    pool=ProcessPoolExecutor(max_workers=config['preparation_processes'],mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(plan,spec))
    depth=config['prefetch_batches'];pending={k:pool.submit(prepare_update,k) for k in range(step,min(step+depth,updates))}
    started=time.monotonic()
    try:
        while step<updates and not stop and time.time()<deadline-900:
            tick=time.monotonic();batches=pending.pop(step).result();wait=time.monotonic()-tick
            if step+depth<updates:pending[step+depth]=pool.submit(prepare_update,step+depth)
            warm=min(1.,(step+1)/max(1,updates//10));cos=.05+.95*.5*(1+math.cos(math.pi*step/updates))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*warm*cos
            optimizer.zero_grad(set_to_none=True);total=0.
            samples=sum(len(b['event']) for b in batches)
            for b in batches:
                b=cuda(b);loss=weighted_microbatch_loss(hazard_loss(model(b),b['event']),b['loss_weight'],samples)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite objective {spec["name"]} update {step}')
                loss.backward();total+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();step+=1
            with (root/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=total,input_wait_seconds=wait,seconds=time.monotonic()-tick,gradient_norm=float(norm)))+'\n')
            if step%config['evaluate_every']==0 or step==updates:
                logits=predict(model,corpus,selected,spec);value=float(weights@hazard_loss(torch.tensor(logits),events).numpy())
                if value<best:best=value;save('best.pt')
                save('last.pt');save_json(status,dict(state='running',step=step,planned_updates=updates,best_selection_nll=best,validation_nll=value,seconds=time.monotonic()-started))
        save('last.pt')
    finally:pool.shutdown(wait=True,cancel_futures=True)
    if step<updates:
        save_json(status,dict(state='checkpointed',step=step,best_selection_nll=best if math.isfinite(best) else None));return False
    save_json(status,dict(state='evaluating',step=step,best_selection_nll=best))
    model.load_state_dict(torch.load(root/'best.pt',map_location='cuda',weights_only=False)['model'])
    # Full natural eligible calibration/test population; no test-based model choices.
    calibration=corpus.splits['calibration'];test=corpus.splits['test']
    results={}
    for name,ids in [('calibration',calibration),('test',test)]:
        path=root/f'{name}-logits.npy'
        if path.exists():results[name]=np.load(path)
        else:
            results[name]=predict(model,corpus,ids,spec)
            temporary=path.with_name(path.stem+'.building.npy');np.save(temporary,results[name]);temporary.replace(path)
    metrics=evaluate(corpus,test,results['test'],calibration,results['calibration'])
    from src.experiment_runner.metric_docs import write_metric_table
    flat={key:{str(r['horizon_ps']):r for r in metrics[key]} for key in ('classification','timing','spatial')}
    flat['test_event_nll']=metrics['test_event_nll']
    if 'training' in spec:
        flat['training']={k:corpus.training_summary[k] for k in ('sources','eligible_windows','updates','samples','complete_epochs','partial_epoch_updates')}
        flat['training']['selected_step']=torch.load(root/'best.pt',map_location='cpu',weights_only=False)['step']
        metrics['training']=dict(flat['training'])
    write_metric_table(flat,resolve_path(config['output']),family='crystallization_transfer',name=spec['name'])
    save_json(root/'metrics.json',metrics)
    np.savez(root/'test-index.npz',indices=test,source=corpus.source_ids[test],event=corpus.events[test],rows=np.array([corpus.rows[i] for i in test]))
    save_json(status,dict(state='complete',step=step,best_selection_nll=best,test_event_nll=metrics['test_event_nll'],seconds=time.monotonic()-started))
    return True
