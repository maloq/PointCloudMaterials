"""Matched frozen-snapshot onset readouts with independent train/selection/calibration/test roles."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash,save_json,digest
from src.research.crystallization_transfer.data import Corpus
from src.research.crystallization_transfer.metrics import evaluate
from src.research.local_predictability.metrics import hazard_loss,source_weights
from src.experiment_runner.metric_docs import write_metric_table


def prepare_population(config):
    import resource
    _,hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    plan_path = resolve_path(config['crystallization_plan'])
    plan = json.loads(plan_path.read_text())
    corpus = Corpus(plan,require_features=False)
    root = resolve_path(config['output'])/'technical/crystallization'
    root.mkdir(parents=True,exist_ok=True)
    path = root/'population.npz'
    if path.exists():
        receipt = json.loads((root/'population.json').read_text())
        if receipt['plan_sha256']!=file_hash(plan_path) or receipt['sha256']!=file_hash(path):
            raise ValueError('Crystallization evaluation population changed')
        return corpus,path
    graph,condition,descriptor,roles = [],[],[],[]
    role_by_id = {s['id']:s.get('validation_role',s['split']) for s in plan['sources']}
    for sid,ai,ci,temp in corpus.rows:
        anchor = plan['anchors'][ai]
        a = corpus.arrays[sid]
        graph.append(a['mapping'][anchor//4,ci,0])
        condition.append([*[float(temp==t) for t in (400,450,500,510,520)],anchor*.75/600,(anchor*.75/600)**2])
        descriptor.append(np.r_[a['packet'][ci,anchor],a['order'][ci,anchor]])
        roles.append(role_by_id[sid])
    np.savez(path,source=corpus.source_ids,graph=graph,event=corpus.events,condition=np.array(condition,np.float32),
        descriptor=np.array(descriptor,np.float32),role=np.array(roles),rows=np.array([r[:3] for r in corpus.rows]))
    save_json(root/'population.json',dict(sha256=file_hash(path),plan_sha256=file_hash(plan_path),rows=len(graph),
        counts={k:len(v) for k,v in corpus.splits.items()},protocol='single current local snapshot; natural at-risk origins; existing 150 independent native-Al sources'))
    return corpus,path


def freeze_record(config,item,population,corpus):
    checkpoint = Path(item['checkpoint']).resolve()
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    # Verify pretraining ancestry before looking at any assay test outcomes.
    if item['kind'] in ('v2','v2_large','regularization'):
        manifest = json.loads((resolve_path(saved['manifest']['config']['cache'])/'manifest.json').read_text())
        roots = set(manifest['train_roots'])|set(manifest['selection_roots'])
    elif item['kind']=='v1':
        # Each v1 run uses the immutable tracked-six release captured in its frozen configuration.
        conf = json.loads(Path(item['training_config']).read_text())
        manifest = json.loads((resolve_path(conf['cache'])/'manifest.json').read_text())
        roots = {r['lineage'] for r in manifest['shards']}
    else:
        manifest = json.loads((resolve_path(saved['identity']['config']['release'])/'manifest.json').read_text())
        roots = {s['lineage'] for s in manifest['sources'] if s['split'] in ('train','selection') and 'lineage' in s}
    protected = {s['lineage'] for s in corpus.plan['sources'] if s.get('validation_role',s['split']) in ('calibration','test')}
    if roots & protected:
        raise ValueError(f'Pretraining assay leakage for {item["name"]}: {roots & protected}')
    folder = resolve_path(config['output'])/'technical/crystallization'/item['name']
    folder.mkdir(parents=True,exist_ok=True)
    record = dict(item,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),directory=str(folder),
        assay_plan=str(resolve_path(config['crystallization_plan'])),assay_cache=str(resolve_path(corpus.plan['config']['cache'])),
        population=str(population),population_sha256=file_hash(population),protected_overlap=[],
        encoder_source_files={str(p.relative_to(Path(item['producer_code']))):file_hash(p)
                              for p in (Path(item['producer_code'])/'src/models/encoders').glob('*.py')})
    path = folder/'record.json'
    if path.exists() and json.loads(path.read_text())!=record:
        raise ValueError('Frozen crystallization encoder record changed')
    save_json(path,record)
    return path,folder


def fit_readout(config,corpus,features,population,folder,name,deadline):
    directory = folder/name
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metrics.json').exists(): return True
    ids = {k:np.asarray(v) for k,v in corpus.splits.items()}
    condition = population['condition']
    train = ids['train']
    # Source-weighted training moments; held-out population never calibrates features.
    w = source_weights(corpus.source_ids[train])
    mean = np.einsum('n,nd->d',w,features[train]).astype(np.float32)
    scale = np.sqrt(np.einsum('n,nd->d',w,(features[train]-mean)**2)).clip(.001).astype(np.float32)
    x = np.c_[(features-mean)/scale,condition].astype(np.float32)
    x = torch.tensor(x,device='cuda')
    y = torch.tensor(corpus.events,device='cuda')
    torch.manual_seed(config['probe_seed'])
    model = (nn.Linear(x.shape[1],6) if name=='linear' else
             nn.Sequential(nn.Linear(x.shape[1],128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,6))).cuda()
    output = model if name=='linear' else model[-1]
    train_event = corpus.events[train]
    hazard = np.array([(w@(train_event==k)+1e-6)/(w@(train_event>=k)+2e-6) for k in range(6)]).clip(1e-5,1-1e-5)
    with torch.no_grad():
        output.bias.copy_(torch.tensor(np.log(hazard/(1-hazard)),device='cuda',dtype=torch.float32))
        output.weight.mul_(.01)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['probe_lr'],weight_decay=1e-4)
    step,best = 0,float('inf')
    last = directory/'last.pt'
    protocol_id = digest(dict(config=config,features=file_hash(folder/'record.json'),readout=name))
    if last.exists():
        saved = torch.load(last,map_location='cuda',weights_only=False)
        if saved['identity']!=protocol_id: raise ValueError('Readout resume identity changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step,best = saved['step'],saved['best']
    def save(path):
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,
                        identity=protocol_id,mean=mean,scale=scale),path)
    @torch.no_grad()
    def logits(indices):
        return torch.cat([model(x[indices[i:i+2048]]) for i in range(0,len(indices),2048)]).cpu().numpy()
    weights = source_weights(corpus.source_ids[ids['selection']])
    while step<config['probe_updates']:
        if time.time()>deadline-180:
            save(last)
            return False
        rng = np.random.default_rng(np.random.SeedSequence([config['probe_seed'],step]))
        chosen = corpus.sample(rng,config['probe_batch_size'])
        factor = min(1.,(step+1)/64)*(.05+.95*.5*(1+math.cos(math.pi*step/config['probe_updates'])))
        optimizer.param_groups[0]['lr'] = config['probe_lr']*factor
        optimizer.zero_grad(set_to_none=True)
        loss = hazard_loss(model(x[chosen]),y[chosen]).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Frozen hazard probe diverged')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step%config['probe_selection_every']==0 or step==config['probe_updates']:
            selected = logits(ids['selection'])
            score = float(weights@hazard_loss(torch.tensor(selected),torch.tensor(corpus.events[ids['selection']])).numpy())
            if score<best:
                best = score
                save(directory/'best.pt')
            save(last)
            save_json(directory/'status.json',dict(state='running',step=step,best_selection_nll=best))
    model.load_state_dict(torch.load(directory/'best.pt',map_location='cuda',weights_only=False)['model'])
    calibration = logits(ids['calibration'])
    test = logits(ids['test'])
    np.savez_compressed(directory/'predictions.npz',test=test,calibration=calibration,
        test_indices=ids['test'],calibration_indices=ids['calibration'],test_sources=corpus.source_ids[ids['test']])
    metrics = evaluate(corpus,ids['test'],test,ids['calibration'],calibration)
    metrics.update(best_selection_nll=best,readout=name,encoder=folder.name,
        split_roles='existing train/selection/calibration/test; historical assay test, not a newly untouched cohort',
        input='one frozen current invariant snapshot + known temperature/time',
        training_updates=step,probe_seed=config['probe_seed'])
    save_json(directory/'metrics.json',metrics)
    table = {key:{str(r['horizon_ps']):r for r in metrics[key]} for key in ('classification','timing','spatial')}
    table.update(test_event_nll=metrics['test_event_nll'],best_selection_nll=best)
    write_metric_table(table,resolve_path(config['output'])/'crystallization',family='neighborhood_crystallization_v2',name=folder.name+'-'+name)
    save_json(directory/'status.json',dict(state='complete',step=step,test_event_nll=metrics['test_event_nll']))
    return True


def run(config,item,deadline):
    import resource
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    torch.set_num_threads(1)
    corpus,population_path = prepare_population(config)
    population = np.load(population_path)
    if item['kind'] == 'baseline':
        folder = resolve_path(config['output'])/'technical/crystallization'/item['name']
        folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/'record.json',dict(item,population_sha256=file_hash(population_path)))
        if item['name']=='geometry-baseline':features=population['descriptor']
        elif item['name']=='geometry-only-baseline':
            # Packet geometry: first 80 + five radial moments; append order8.
            features=population['descriptor'][:,np.r_[0:80,112:117,128:136]]
        elif item['name']=='condition-baseline':features=np.zeros((len(corpus.rows),0),np.float32)
        else:raise ValueError(item['name'])
        for name in ('linear','mlp'):
            if not fit_readout(config,corpus,features,population,folder,name,deadline): return False
        save_json(folder/'status.json',dict(state='complete',encoder=item['name']))
        report(config)
        return True
    record,folder = freeze_record(config,item,population_path,corpus)
    bootstrap = 'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1),run_name="__main__")'
    worker = Path(__file__).with_name('extract.py').resolve()
    with (folder/'extraction.log').open('a') as log:
        subprocess.run([sys.executable,'-c',bootstrap,item['producer_code'],str(worker),'--record',str(record)],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    features = np.empty((len(corpus.rows),128),np.float32)
    for sid in np.unique(corpus.source_ids):
        features[corpus.source_ids==sid] = np.load(folder/'features'/f'{sid}.npy')
    for name in ('linear','mlp'):
        if not fit_readout(config,corpus,features,population,folder,name,deadline): return False
    save_json(folder/'status.json',dict(state='complete',encoder=item['name']))
    report(config)
    return True


def report(config):
    root = resolve_path(config['output'])
    lines = ['# Frozen-encoder crystallization comparison','',
        'One current local snapshot, frozen encoder; matched linear/MLP discrete-time hazards. '
        'Temperature and elapsed time are supplied to all probes. Lower NLL/MAE is better; higher AP is better. '
        'Test sources are the existing historical assay, not a new untouched test set. One training seed.',
        '', '| Encoder | Probe | Event NLL | 9 ps AP | 24 ps AP | 96 ps AP | 24 ps timing MAE (ps) | 24 ps misses / events |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for path in sorted((root/'technical/crystallization').glob('*/*/metrics.json')):
        m = json.loads(path.read_text())
        rows = {r['horizon_ps']:r for r in m['classification']}
        timing = next(r for r in m['timing'] if r['horizon_ps']==24.)
        fmt = lambda v:'—' if v is None else f'{v:.4f}'
        lines.append(f"| {m['encoder']} | {m['readout']} | {m['test_event_nll']:.4f} | {fmt(rows[9.]['average_precision'])} | {fmt(rows[24.]['average_precision'])} | {fmt(rows[96.]['average_precision'])} | {fmt(timing['detected_window_timing_mae_ps'])} | {timing['missed_windows']} / {timing['true_events']} |")
    lines.extend(['','Timing MAE is conditional on detected event windows; misses are reported alongside it. '
        'Thresholds target 5% FPR on independent calibration sources. Full metrics include AUROC, Brier, calibration, '
        'event recall, alarms and sampled-center spatial scores, with whole-source uncertainty.',
        '', 'This table updates after each completed encoder assay; missing rows are pending, not failures or zero scores.'])
    (root/'CRYSTALLIZATION.md').write_text('\n'.join(lines)+'\n')
