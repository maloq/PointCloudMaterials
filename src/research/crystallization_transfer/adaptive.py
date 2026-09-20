"""Full-batch normalized heads with exact microbatch encoder gradient replay."""
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import json
import math
import multiprocessing
import signal
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json
from src.training_methods.shared_pretraining.compilation import compile_encoder
from src.research.local_predictability.metrics import hazard_loss,source_weights
from .data import Corpus
from .model import Predictor
from .training import configure_training


def reference_indices(corpus,count,seed):
    indices=[]
    for sid,rows in sorted(corpus.groups.items()):
        rng=np.random.default_rng(np.random.SeedSequence([seed,2901,int(sid)]))
        indices.extend(rng.choice(rows,min(count,len(rows)),replace=False).tolist())
    return indices


@torch.no_grad()
def calibrate(model,corpus,spec,indices,cached=False):
    from .runtime import pack,cuda
    was_training=model.training;model.eval();xs=[];ws=[];zs=[];gs=[];cs=[]
    observation=dict(spec,mode='frozen') if cached else spec
    size=128 if observation['mode']=='frozen' else 16
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(pack,corpus,indices[:size],observation)
        for start in range(0,len(indices),size):
            batch=cuda(future.result())
            if start+size<len(indices):future=pool.submit(pack,corpus,indices[start+size:start+2*size],observation)
            features=model.encode(batch);x,w=model.head.inputs(features,batch['geometry']);xs.append(x);ws.append(w)
            if sum(len(z) for z in zs)<64:
                zs.append(features);gs.append(batch['geometry']);cs.append(batch['condition'])
    x=torch.cat(xs);w=torch.cat(ws);model.head.normalization.calibrate(x,w)
    z=torch.cat(zs)[:64];g=torch.cat(gs)[:64];c=torch.cat(cs)[:1].expand(len(z),-1)
    logits=model.head(z,g,c)
    health=dict(geometry_logit_std=float(logits.std(0,unbiased=False).mean()),
        scalar_feature_std_rms=float(model.head.normalization.variance[:128].mean().sqrt()),
        normalization_scale_median=float((model.head.normalization.variance+model.head.normalization.eps).sqrt().median()))
    model.train(was_training);return health


def gradient_step(model,batches,update_encoder,replay=True):
    """Head moments see the whole statistical batch; replay only the encoder VJP."""
    from .runtime import cuda
    device_batches=[cuda(b) for b in batches]
    if update_encoder and not replay:features=torch.cat([model.encode(b) for b in device_batches])
    else:
        with torch.no_grad():features=torch.cat([model.encode(b) for b in device_batches])
        features=features.detach().requires_grad_(update_encoder)
    metadata={k:torch.cat([b[k] for b in device_batches]) for k in ('geometry','condition','event','loss_weight')}
    logits=model.head(features,metadata['geometry'],metadata['condition'],sample_weight=metadata['loss_weight'])
    loss=(hazard_loss(logits,metadata['event'])*metadata['loss_weight']).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptive full-batch hazard loss')
    loss.backward()
    if update_encoder and replay:
        gradient=features.grad;start=0
        for b in device_batches:
            stop=start+len(b['event']);actual=model.encode(b)
            actual.backward(gradient[start:stop]);start=stop
    if update_encoder and not any(p.grad is not None for p in model.encoder.parameters()):raise RuntimeError('Encoder backward produced no parameter gradients')
    return float(loss.detach())


def rates(step,updates,warmup_steps,spec):
    head_warm=min(1.,(step+1)/max(1,updates//20))
    head=spec['head_lr']*head_warm*(.05+.95*.5*(1+math.cos(math.pi*step/updates)))
    if step<warmup_steps:return head,0.
    elapsed=step-warmup_steps;length=max(1,updates-warmup_steps)
    warm=min(1.,(elapsed+1)/max(1,min(512,length//10)))
    encoder=spec['encoder_lr']*warm*(.05+.95*.5*(1+math.cos(math.pi*elapsed/length)))
    return head,encoder


def fit(plan,spec,deadline):
    from .runtime import setup,parent_state,init_worker,prepare_update,pack,cuda,predict
    from .metrics import evaluate
    from src.experiment_runner.metric_docs import write_metric_table
    setup();torch._dynamo.reset();config=plan['config'];root=resolve_path(config['output'])/'technical/runs'/spec['name'];root.mkdir(parents=True,exist_ok=True)
    if config['encoder_backward'] not in ('direct','replay'):raise ValueError(f'Unknown encoder backward mode: {config["encoder_backward"]}')
    status=root/'status.json'
    if status.exists() and json.loads(status.read_text())['state']=='complete':return True
    stop=False
    def request_stop(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    torch.manual_seed(config['seed']);corpus=Corpus(plan);population=configure_training(corpus,spec)
    updates=population['updates'];warmup_steps=spec['warmup_epochs']*population['updates_per_epoch']
    spec=dict(spec,head_warmup_updates=warmup_steps)
    save_json(root/'training-population.json',population);np.save(root/'training-indices.npy',corpus.splits['train'])
    model=Predictor(spec,parent_state(plan)).cuda();parameters=[dict(params=model.head.parameters(),lr=spec['head_lr'])]
    save_json(root/'model.json',dict(total_parameters=sum(p.numel() for p in model.parameters()),
        head_parameters=sum(p.numel() for p in model.head.parameters()),
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()) if model.encoder is not None else 0,
        encoder_backward=config['encoder_backward']))
    if model.encoder is not None:parameters.append(dict(params=model.encoder.parameters(),lr=spec['encoder_lr']))
    optimizer=torch.optim.AdamW(parameters,weight_decay=spec['weight_decay']);step=0;best=float('inf')
    references=reference_indices(corpus,config['normalization_per_source'],config['seed'])
    np.save(root/'normalization-indices.npy',references)
    selected=[]
    for sid in np.unique(corpus.source_ids[corpus.splits['selection']]):
        rows=np.array(corpus.splits['selection']);rows=rows[corpus.source_ids[rows]==sid]
        rng=np.random.default_rng(np.random.SeedSequence([config['seed'],int(sid)]))
        selected.extend(sorted(rng.choice(rows,min(config['selection_per_source'],len(rows)),replace=False).tolist()))
    weights=source_weights(corpus.source_ids[selected]);events=torch.tensor(corpus.events[selected])
    if (root/'last.pt').exists():
        saved=torch.load(root/'last.pt',map_location='cuda',weights_only=False)
        if saved['identity']!=plan['identity'] or saved['spec']!=spec:raise ValueError('Adaptive resume identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step=saved['step'];best=saved['best']
        torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    else:
        target=corpus.events[references];hazards=np.array([(np.sum(target==k)+.5)/(np.sum(target>=k)+1) for k in range(6)])
        with torch.no_grad():model.head.output[-1].bias.copy_(torch.tensor(np.log(hazards/(1-hazards)),device='cuda',dtype=torch.float32))
    if model.encoder is not None:
        example=cuda(pack(corpus,corpus.splits['train'][:1],dict(spec,history_ps=0,radius_A=0)))['graph_batch']
        compile_encoder(model.encoder,example,'bf16')
    def save(name):
        temp=root/(name+'.building')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,spec=spec,
            identity=plan['identity'],rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temp);temp.replace(root/name)
    started=time.monotonic()
    def validate():
        nonlocal best
        cached=spec['mode']=='frozen' or (spec['mode']=='finetune' and step<=warmup_steps)
        health=calibrate(model,corpus,spec,references,cached=cached)
        observation=dict(spec,mode='frozen') if cached else spec
        logits=predict(model,corpus,selected,observation);value=float(weights@hazard_loss(torch.tensor(logits),events).numpy())
        if not math.isfinite(value):raise FloatingPointError(f'Nonfinite validation NLL: {value}')
        if value<best:best=value;save('best.pt')
        save('last.pt');save_json(status,dict(state='running',step=step,planned_updates=updates,best_selection_nll=best,validation_nll=value,health=health,seconds=time.monotonic()-started))
        with (root/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,selection_nll=value,**health))+'\n')
    if step==0:validate()
    pool=ProcessPoolExecutor(max_workers=config['preparation_processes'],mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(plan,spec))
    depth=config['prefetch_batches'];pending={k:pool.submit(prepare_update,k) for k in range(step,min(step+depth,updates))}
    try:
        while step<updates and not stop and time.time()<deadline-900:
            tick=time.monotonic();batches=pending.pop(step).result();wait=time.monotonic()-tick
            if step+depth<updates:pending[step+depth]=pool.submit(prepare_update,step+depth)
            head_lr,encoder_lr=rates(step,updates,warmup_steps,spec);optimizer.param_groups[0]['lr']=head_lr
            if model.encoder is not None:optimizer.param_groups[1]['lr']=encoder_lr
            optimizer.zero_grad(set_to_none=True);active=model.encoder is not None and step>=warmup_steps
            loss=gradient_step(model,batches,active,replay=config['encoder_backward']=='replay')
            head_norm=torch.nn.utils.clip_grad_norm_(model.head.parameters(),5.,error_if_nonfinite=True)
            encoder_norm=torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),spec['encoder_clip'],error_if_nonfinite=True) if active else torch.tensor(0.)
            optimizer.step();step+=1
            with (root/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=loss,input_wait_seconds=wait,seconds=time.monotonic()-tick,
                head_gradient_norm=float(head_norm),encoder_gradient_norm=float(encoder_norm),head_lr=head_lr,encoder_lr=encoder_lr))+'\n')
            if step%config['evaluate_every']==0 or step in (updates,warmup_steps):validate()
        save('last.pt')
    finally:pool.shutdown(wait=True,cancel_futures=True)
    if step<updates:
        save_json(status,dict(state='checkpointed',step=step,planned_updates=updates,best_selection_nll=best));return False
    save_json(status,dict(state='evaluating',step=step,best_selection_nll=best))
    selected_checkpoint=torch.load(root/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(selected_checkpoint['model'])
    calibration=corpus.splits['calibration'];test=corpus.splits['test'];results={}
    for name,ids in [('calibration',calibration),('test',test)]:
        path=root/f'{name}-logits.npy'
        if path.exists():results[name]=np.load(path)
        else:
            results[name]=predict(model,corpus,ids,spec);temporary=path.with_name(path.stem+'.building.npy');np.save(temporary,results[name]);temporary.replace(path)
    metrics=evaluate(corpus,test,results['test'],calibration,results['calibration'])
    flat={key:{str(r['horizon_ps']):r for r in metrics[key]} for key in ('classification','timing','spatial')};flat['test_event_nll']=metrics['test_event_nll']
    metrics['training']={k:population[k] for k in ('sources','eligible_windows','updates','samples','complete_epochs','partial_epoch_updates')}
    metrics['training'].update(selected_step=selected_checkpoint['step'],head_warmup_updates=warmup_steps);flat['training']=metrics['training']
    write_metric_table(flat,resolve_path(config['output']),family='crystallization_transfer',name=spec['name']);save_json(root/'metrics.json',metrics)
    np.savez(root/'test-index.npz',indices=test,source=corpus.source_ids[test],event=corpus.events[test],rows=np.array([corpus.rows[i] for i in test]))
    save_json(status,dict(state='complete',step=step,best_selection_nll=best,test_event_nll=metrics['test_event_nll'],seconds=time.monotonic()-started))
    return True
