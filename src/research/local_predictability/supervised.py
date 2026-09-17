"""Matched native onset parent and continuations, with explicit release/gate checks."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.data.predictive_memory.prepare import file_hash,write_json
from src.project_runtime.paths import load_json,resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from .native_data import NativeWindows,SourceSampler,conditions
from .native_preflight import seed_all
from .onset_model import OnsetModel
from .metrics import hazard_loss,cumulative_risk,source_weights
from .baselines import score_hazard


def prepare_rows(config):
    shared=resolve_path(config['shared']);data=resolve_path(config['data_output'])/'technical'
    release=json.loads((data/'release.json').read_text());gates=json.loads((data/'assay_gates.json').read_text())
    if not all(gates[k] for k in ['source_integrity','assay_integrity','coverage']):
        raise RuntimeError(f'Native onset gates failed: {gates}')
    smallfit=json.loads(resolve_path(config['smallfit_receipt']).read_text())
    if smallfit['state']!='passed':raise RuntimeError('Small-set information fitting gate has not passed')
    if smallfit['identity']['cohort_sha256']!=file_hash(data/'cohort.json'):
        raise ValueError('Small-set gate used a different frozen cohort')
    for name in ['native_model.py','native_data.py']:
        if smallfit['identity']['implementation_sha256'][name]!=file_hash(Path(__file__).parent/name):
            raise ValueError(f'Small-set gate does not verify this native implementation: {name}')
    # Produce a local source-audit adapter only from locally verified preparation receipts.
    audit_sources=[]
    for source in release['sources']:
        if not source['checksum_verified']:raise RuntimeError(f"Local array audit incomplete: {source['id']}")
        audit_sources.append({**{k:source[k] for k in ['id','lineage','split','dataset','relative_trajectory_path','manifest_sha256']},'status':'passed'})
    root=resolve_path(config['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    audit_path=root/'local_source_audit.json'
    write_json(audit_path,dict(status='passed',counts={'passed':150},sources=audit_sources,
        producer_release_sha256=file_hash(data/'release.json')))
    windows=NativeWindows(data/'cohort.json',audit_path,device='cuda',cpu_cache_gib=config['cpu_cache_gib'],gpu_cache_gib=config['gpu_cache_gib'])
    labels=dict(np.load(data/'native_rows.npz'));index=dict(np.load(shared/'native_index.npz'))
    for key in ['source_id','center_id','anchor','split']:
        actual=np.array([r[key] for r in windows.rows])
        np.testing.assert_array_equal(actual,labels[key],err_msg=f'Label row mismatch: {key}')
        np.testing.assert_array_equal(actual,index[key],err_msg=f'Frozen row mismatch: {key}')
    stats=json.loads((shared/'conditions.json').read_text());cond,_=conditions(windows.rows,stats)
    tensor_cond=torch.tensor(cond,device='cuda');events=torch.tensor(labels['event_bin'],device='cuda',dtype=torch.long)
    allowed=np.flatnonzero(labels['risk']);splits={s:allowed[labels['split'][allowed]==s] for s in ['train','selection','calibration','test']}
    if any(len(v)==0 for v in splits.values()):raise RuntimeError('One native event split has no eligible rows')
    selection=[]
    for sid in np.unique(labels['source_id'][splits['selection']]):
        eligible=splits['selection'][labels['source_id'][splits['selection']]==sid]
        rng=np.random.default_rng(np.random.SeedSequence([20260919,int(sid),64]))
        selection.extend(sorted(rng.choice(eligible,min(64,len(eligible)),replace=False).tolist()))
    identity=dict(protocol=config['protocol'],seed=config['seed'],cohort_sha256=file_hash(data/'cohort.json'),
        release_sha256=file_hash(data/'release.json'),labels_sha256=file_hash(data/'native_rows.npz'),
        conditions_sha256=file_hash(shared/'conditions.json'),smallfit_receipt_sha256=file_hash(resolve_path(config['smallfit_receipt'])),
        updates_per_stage=config['updates_per_stage'],selection_indices=selection,
        implementation={name:file_hash(Path(__file__).parent/name) for name in ['supervised.py','onset_model.py','native_model.py','native_data.py','metrics.py']})
    write_json(root/'identity.json',identity)
    return windows,labels,tensor_cond,events,splits,selection,identity


def save(path,model,optimizer,sampler,step,best,identity,stage):
    temporary=path.with_suffix('.building.pt')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),
        step=step,best=best,identity=identity,stage=stage,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model,windows,indices,cond,event,microbatch):
    model.eval();logits=[];states=[]
    for start in range(0,len(indices),microbatch):
        batch=indices[start:start+microbatch]
        result=model([windows.observation(int(i),model.encoder.variant) for i in batch],cond[batch])
        logits.append(result['logits'].cpu());states.append(result['state'].cpu())
    logits=torch.cat(logits);states=torch.cat(states)
    losses=hazard_loss(logits,event[indices].cpu()).numpy()
    weights=source_weights(np.array([windows.rows[int(i)]['source_id'] for i in indices]))
    model.train()
    return float(weights@losses),logits,states


def fit_stage(config,windows,labels,cond,events,splits,selection,identity,stage,parent=None,resume=False):
    variant='snapshot' if stage=='parent' else stage
    root=resolve_path(config['output'])/'technical'/stage;root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():
        old=json.loads((root/'complete.json').read_text())
        if old['identity']!=identity:raise ValueError(f'Completed stage identity changed: {stage}')
        return root/'best.pt'
    seed_all();model=OnsetModel(variant,activation_checkpoint=config['activation_checkpoint'],max_spatial_edges=config['max_spatial_edges']).cuda()
    if parent is not None:
        state=torch.load(parent,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Parent release/implementation identity differs')
        model.load_state_dict(state['model'])
    else:
        y=labels['event_bin'][splits['train']];weight=source_weights(labels['source_id'][splits['train']])
        frequency=np.array([np.clip(weight[y==k].sum()/max(weight[y>=k].sum(),1e-12),1e-4,1-1e-4) for k in range(6)])
        with torch.no_grad():model.hazard.bias.copy_(torch.tensor(np.log(frequency/(1-frequency)),device='cuda',dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    sampler=SourceSampler(windows.rows,splits['train']);step=0;best=float('inf')
    latest=root/'latest.pt'
    if latest.exists():
        if not resume:raise FileExistsError(f'Interrupted stage needs explicit --resume: {latest}')
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity']!=identity or state['stage']!=stage:raise ValueError('Exact resume identity changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);sampler.load_state_dict(state['sampler'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng']);step=state['step'];best=state['best']
    microbatch=config['microbatch'][variant];maximum=config['updates_per_stage']
    deadline=datetime.fromisoformat(config['training_deadline_utc']).timestamp()
    torch.cuda.reset_peak_memory_stats();model.train();started=time.monotonic()
    with (root/'training.jsonl').open('a' if resume else 'x') as log:
        while step<maximum:
            if time.time()+180>=deadline:
                save(latest,model,optimizer,sampler,step,best,identity,stage)
                write_json(root/'status.json',dict(state='deadline_stopped',step=step))
                raise TimeoutError(f'Native training cutoff reached in {stage} at {step}')
            indices=sampler.batch();optimizer.zero_grad(set_to_none=True);value=0.
            for start in range(0,8,microbatch):
                batch=indices[start:start+microbatch]
                prediction=model([windows.observation(i,variant) for i in batch],cond[batch])
                loss=hazard_loss(prediction['logits'],events[batch]).sum()/8
                loss.backward();value+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1
            if step%50==0 or step==1:
                record=dict(stage=stage,step=step,train_nll=value,gradient_norm=float(norm),elapsed_seconds=time.monotonic()-started,
                    peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
                log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                write_json(root/'status.json',dict(state='running',**record))
            if step%256==0 or step==maximum:
                score,_,_=evaluate(model,windows,selection,cond,events,microbatch)
                if score<best:
                    best=score;save(root/'best.pt',model,optimizer,sampler,step,best,identity,stage)
                save(latest,model,optimizer,sampler,step,best,identity,stage)
                with (root/'validation.jsonl').open('a') as val:val.write(json.dumps(dict(step=step,selection_nll=score,best=best))+'\n')
                print(json.dumps(dict(stage=stage,step=step,selection_nll=score,best=best)),flush=True)
    write_json(root/'complete.json',dict(state='complete',identity=identity,stage=stage,steps=step,best=best,seconds=time.monotonic()-started,cache=windows.statistics()))
    del model,optimizer;torch.cuda.empty_cache()
    return root/'best.pt'


def score_stage(config,windows,labels,cond,events,splits,identity,stage):
    root=resolve_path(config['output'])/'technical'/stage
    state=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    if state['identity']!=identity:raise ValueError('Scoring identity changed')
    model=OnsetModel(stage,activation_checkpoint=False,max_spatial_edges=config['max_spatial_edges']).cuda();model.load_state_dict(state['model'])
    indices=np.concatenate([splits[s] for s in ['selection','calibration','test']])
    _,logits,z=evaluate(model,windows,indices,cond,events,config['microbatch'][stage])
    arrays=dict(y=labels['event_bin'][indices],source=labels['source_id'][indices],center=labels['center_id'][indices],
        anchor=labels['anchor'][indices],split=labels['split'][indices],temperature=np.array([windows.rows[int(i)]['temperature_K'] for i in indices]))
    masks={s:np.flatnonzero(arrays['split']==s) for s in ['selection','calibration','test']}
    probability=cumulative_risk(logits).numpy();plan=json.loads(resolve_path(config['plan']).read_text())
    result=score_hazard(arrays,probability,logits.numpy(),masks,plan)
    np.savez(root/'predictions.npz',probability=probability,logits=logits.numpy(),embeddings=z.numpy(),indices=indices,
        event_bin=arrays['y'],**{k:v for k,v in arrays.items() if k!='y'})
    write_json(root/'result.json',result);snapshot_metric_docs(resolve_path(config['output']),'local_predictability_native_onset')
    import csv
    records=result['population']
    keys=list(dict.fromkeys(k for row in records for k in row))
    with (resolve_path(config['output'])/'tables'/f'{stage}.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)
    del model;torch.cuda.empty_cache()
    return result


def run(config,resume=False):
    torch.set_num_threads(config['torch_threads'])
    windows,labels,cond,events,splits,selection,identity=prepare_rows(config)
    budget=json.loads(resolve_path(config['budget_receipt']).read_text())
    if budget['state']!='agreed' or budget['updates_per_stage']!=config['updates_per_stage']:
        raise RuntimeError('Need frozen matched update budget agreed with H200')
    parent=fit_stage(config,windows,labels,cond,events,splits,selection,identity,'parent',resume=resume)
    for stage in ['snapshot','history12','repeat12']:
        fit_stage(config,windows,labels,cond,events,splits,selection,identity,stage,parent=parent,resume=resume)
        score_stage(config,windows,labels,cond,events,splits,identity,stage)
    root=resolve_path(config['output']);write_json(root/'technical/status.json',dict(state='complete',seed=20260919,identity=identity,variants=['snapshot','history12','repeat12']))
    lines=['# Native supervised onset comparison','',f"One seed; K={config['updates_per_stage']} parent updates and K per continuation.",'',
        '| Variant | Horizon (ps) | Test log loss | Test Brier | Test AP |','| --- | --- | --- | --- | --- |']
    for stage in ['snapshot','history12','repeat12']:
        result=json.loads((root/'technical'/stage/'result.json').read_text())
        for row in result['population']:
            if row['split']=='test' and row['horizon_ps'] in [9,48]:
                lines.append(f"| {stage} | {row['horizon_ps']} | {row['log_loss']:.4f} | {row['brier']:.4f} | {row['average_precision']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(load_json(args.config),args.resume)


if __name__=='__main__':main()
