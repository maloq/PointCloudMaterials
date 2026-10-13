"""Native memory and matched physical training with the same predictive scores."""
import math
import os
import time
import numpy as np
import torch
from src.experiment_runner.wandb_tracking import online_training
from src.experiment_runner.metric_docs import write_metric_rows
from .data import root, read, digest, atomic_save, write_json, sha
from .memory import load, MemoryProcess, native_history, settings, FAMILY
from .model import predictive_scores
from .train import response_prediction, response_loss, objective


def value_scores(prediction, target, initial, weights, increments, cfg, *, physical):
    targets = target.unbind(1) if physical else [target]
    values = [predictive_scores(prediction, t, initial, weights, increments, cfg['likelihood_bandwidth']) for t in targets]
    return {k:torch.stack([v[k] for v in values]).mean(0) for k in values[0]}


def selection(model, c, arm, x, physical, ids, weights, increments):
    scores = []
    with torch.no_grad():
        for offset in range(0, len(ids), 32):
            chunk = ids[offset:offset+32]
            if arm['task'] == 'native':
                sample = x[chunk]; h = native_history(sample, arm); target = sample[:, 14:]
                predicted = model(h, 50, c['evaluation']['selection_particles'], c['seed']+offset)
            else:
                h = physical['history'][chunk, -1:]; target = physical['target'][chunk]
                predicted = model(h, 1, c['evaluation']['selection_particles'], c['seed']+offset, dense=True)
            row = value_scores(predicted, target, h[:, -1], weights, increments, c['training'], physical=arm['task']=='physical')
            scores.append({k:v.cpu() for k,v in row.items()})
    return {k:float(torch.cat([v[k] for v in scores]).mean()) for k in scores[0]}


def fit(c, arm, deadline):
    folder = root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists(): return True
    plan, data, x, bank, physical = load(c, device='cuda'); cfg = c['training']
    rate, increments, weights = settings(data, physical, arm, 'cuda')
    records = data['rows'] if arm['task'] == 'native' else bank
    train, valid = [np.flatnonzero([r['role'] == role for r in records]) for role in ['train','selection']]
    if arm['task'] == 'native':
        valid = np.random.default_rng(c['seed']).choice(valid, c['evaluation']['selection_rows'], replace=False)
    torch.manual_seed(arm['seed']); model = MemoryProcess(c, rate, arm).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    count = cfg['native_updates'] if arm['task']=='native' else cfg['physical_updates']
    batch = cfg['batch_size'] if arm['task']=='native' else cfg['physical_batch']
    # Same effective batch on all devices; record accumulation explicitly.
    memory = torch.cuda.get_device_properties(0).total_memory/2**30
    micro = min(batch, cfg['microbatch'] if memory >= 90 else (128 if memory >= 64 else 64))
    identity = digest([plan['identity'], arm]); rng = np.random.default_rng(arm['seed'])
    start = 0; history = []; best = math.inf; selected = 0
    if (folder/'last.pt').exists():
        old = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        if old['identity'] != identity: raise ValueError('Changed resumed fit identity')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        start, history, best, selected = old['update'], old['history'], old['best'], old['selected']
        rng.bit_generator.state = old['rng']; torch.set_rng_state(old['torch_rng'])
    def save(update):
        atomic_save(folder/'last.pt', dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
            update=update, history=history, best=best, selected=selected, rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), arm=arm))
    execution = dict(effective_batch=batch, microbatch=micro, memory_GiB=memory,
        allocation=os.environ.get('SLURM_JOB_ID'), resumed_from_update=start,
        precision='native BF16 network / FP32 state and losses; physical values and secants FP32',
        physical_source_precision='archived float64 integration/tangents; float32 forces/measurement JVPs')
    write_json(folder/'execution.json', execution)
    write_json(folder/'execution-segments'/f'{time.time_ns()}.json', execution)
    with online_training(c['wandb'], run_id='meas2-'+identity[:16], name=arm['name'], config=dict(c, arm=arm),
                         folder=folder, receipt_path=folder/'wandb.json', job_type='predictor', group=c['protocol'],
                         receipt_fields=dict(identity=identity)) as tracking:
        for update in range(start+1, count+1):
            if time.time() > deadline-600: save(update-1); return False
            begin = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
            warmup = min(update/cfg['warmup'], 1.)
            cosine = .01+.99*.5*(1+math.cos(math.pi*max(0,update-cfg['warmup'])/(count-cfg['warmup'])))
            for group in optimizer.param_groups: group['lr'] = cfg['lr']*warmup*cosine
            chosen = rng.choice(train, batch); seed = int(rng.integers(2**31))
            steps = int(rng.choice(cfg['horizons_steps'], p=cfg['horizon_probabilities'])) if arm['task']=='native' else 20
            totals = {}; derivative_total = 0.
            for offset in range(0, batch, micro):
                ids = chosen[offset:offset+micro]; factor = len(ids)/batch
                if arm['task']=='native':
                    observed = x[ids]; h = native_history(observed,arm); target = observed[:,14:14+steps]
                    with torch.autocast('cuda',dtype=torch.bfloat16): predicted = model(h,steps,cfg['particles'],seed+offset)
                else:
                    h = physical['history'][ids,-1:]; target = physical['target'][ids]
                    predicted = model(h,1,cfg['particles'],seed+offset,dense=True)
                scores = value_scores(predicted,target,h[:,-1],weights,increments,cfg,physical=arm['task']=='physical')
                (factor*objective(scores,cfg)).backward()
                for key,value in scores.items(): totals[key] = totals.get(key,0.)+factor*float(value.detach().mean())
                del predicted,scores
                if arm['response']:
                    response = response_prediction(model,h,physical['pairs'][ids],cfg['response_epsilon'],seed+offset,cfg['particles'])
                    loss = response_loss(response,physical['response'][ids],physical['normalizers'])
                    (factor*cfg['response_weight']*loss).backward(); derivative_total += factor*float(loss.detach())
                    del response,loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True); optimizer.step()
            row = dict(update=update,horizon_ps=steps*(.1 if arm['task']=='native' else .005),
                       **totals,response_loss=derivative_total,gradient_norm=float(norm))
            if update%cfg['selection_every']==0 or update==count:
                model.eval(); valid_scores = selection(model,c,arm,x,physical,valid,weights,increments)
                row.update({'selection_'+key:value for key,value in valid_scores.items()})
                if arm['task']=='physical':
                    with torch.no_grad():
                        for label,ids in [('train',train[:20]),('selection',valid)]:
                            prediction = response_prediction(model,physical['history'][ids,-1:],physical['pairs'][ids],
                                                            cfg['response_epsilon'],c['seed'],8)
                            row[label+'_response_loss'] = float(response_loss(prediction,physical['response'][ids],physical['normalizers']))
                if valid_scores['nll'] < best:
                    best,selected = valid_scores['nll'],update
                    atomic_save(folder/'best.pt',dict(identity=identity,model=model.state_dict(),update=update,arm=arm))
            history.append(row); tracking.log(row,step=update)
            if update==start+1 or update%cfg['save_every']==0: save(update)
            write_json(folder/'progress.json',dict(state='training',**row,total=count,seconds_per_update=time.monotonic()-begin))
            if update==start+1 or update%50==0: print(arm['name'],row,flush=True)
        tracking.summary.update(dict(selected_update=selected,selection_nll=best))
    save(count)
    write_metric_rows(history,folder.parent,family=FAMILY,name='learning',columns=list(dict.fromkeys(k for row in history for k in row)))
    write_json(folder/'complete.json',dict(state='complete',updates=count,selected_update=selected,
                                         checkpoint_sha256=sha(folder/'best.pt'),identity=identity))
    return True
