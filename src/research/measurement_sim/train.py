"""Matched likelihood/path/response fits with resumable online tracking."""
import math
import time
import numpy as np
import torch
from src.experiment_runner.wandb_tracking import online_training
from src.experiment_runner.metric_docs import write_metric_rows
from .data import root, cache, read, sha, digest, write_json, atomic_save
from .physical import load as physical_bank
from .model import DirectProcess, predictive_scores, feature_weights

FAMILY = 'measurement_sim'


def inputs(c):
    receipt = read(cache(c)/'data.json')
    if sha(cache(c)/'data.pt') != receipt['sha256']: raise ValueError('Changed measurement cache')
    data = torch.load(cache(c)/'data.pt', map_location='cpu', weights_only=False)
    bank = physical_bank(c)
    mean, scale = data['mean'].cuda(), data['scale'].cuda()
    x = (data['x'].cuda()-mean)/scale
    physical = dict(history=torch.stack([r['history'] for r in bank]).cuda(),
        pairs=torch.stack([r['pairs'] for r in bank]).cuda(),
        half_pairs=torch.stack([r['half_pairs'] for r in bank]).cuda(),
        target=torch.stack([r['target'] for r in bank]).cuda())
    physical = {k:(v-mean)/scale for k, v in physical.items()}
    physical['response'] = torch.stack([r['response'] for r in bank]).cuda()/scale[None, None, 153:, None]
    physical['initial_response'] = torch.stack([r['initial_response'] for r in bank]).cuda()/scale[None, 153:, None]
    train = np.flatnonzero([r['role'] == 'train' for r in bank])
    # Fixed training-bank normalizers, never noisy minibatch denominators.
    normalizers = []
    for start, stop in [(0, 74), (74, 83), (83, 87), (87, 91)]:
        normalizers.append(physical['response'][train, :, start:stop].square().mean((0, 2)).clamp_min(1e-5))
    return data, x, bank, physical, torch.stack(normalizers)


def response_prediction(model, history, pairs, epsilon, seed, particles=4):
    predictions = []
    for k in range(2):
        plus, minus = history.clone(), history.clone()
        minus[:, -1] = pairs[:, k, 0]; plus[:, -1] = pairs[:, k, 1]
        a = model(plus, 1, particles, seed, dense=True)[..., 153:].mean(1)
        b = model(minus, 1, particles, seed, dense=True)[..., 153:].mean(1)
        predictions.append((a-b)/(2*epsilon))
    return torch.stack(predictions, -1)


def response_loss(predicted, target, normalizers):
    pieces = []
    for g, (start, stop) in enumerate([(0, 74), (74, 83), (83, 87), (87, 91)]):
        pieces.append(((predicted[..., start:stop, :]-target[..., start:stop, :]).square().mean((0, 2))
                       /normalizers[g]).mean())
    return torch.stack(pieces).mean()


def objective(scores, cfg, connected=True):
    value = scores['nll']
    if connected:
        value = value+cfg['path_weight']*scores['path_energy']
        value = value+cfg['increment_weight']*scores['increment_energy']
        value = value+cfg['correlation_weight']*scores['correlation_energy']
    return value.mean()


def select(model, x, indices, weights, increment_scale, c):
    cfg = c['training']; collected = []
    with torch.no_grad():
        for start in range(0, len(indices), 32):
            sample = x[indices[start:start+32]]
            predicted = model(sample[:, :4], 60, 8, c['seed']+start)
            score = predictive_scores(predicted, sample[:, 4:], sample[:, 3],
                    weights, increment_scale, cfg['likelihood_bandwidth'])
            collected.append({k:float(v.mean()) for k,v in score.items()})
    return {k:float(np.mean([r[k] for r in collected])) for k in collected[0]}


def fit(c, p, arm, deadline):
    folder = root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists(): return True
    data, x, bank, physical, normalizers = inputs(c)
    cfg = c['training']; torch.manual_seed(arm['seed'])
    model = DirectProcess(c, data['rate'].cuda()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    weights = feature_weights(data['scale'].cuda()); increment_scale = data['increment_scale'].cuda()
    train = np.flatnonzero([r['role'] == 'train' for r in data['rows']])
    selection = np.flatnonzero([r['role'] == 'selection' for r in data['rows']])
    selection = np.random.default_rng(c['seed']).choice(selection, c['evaluation']['selection_rows'], replace=False)
    teachers = np.flatnonzero([r['role'] == 'train' for r in bank])
    validation_teachers = np.flatnonzero([r['role'] == 'selection' for r in bank])
    identity = digest([p['identity'], arm]); rng = np.random.default_rng(arm['seed'])
    start = 0; history = []; best = float('inf'); selected = 0
    if (folder/'last.pt').exists():
        old = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        if old['identity'] != identity: raise ValueError('Changed direct-model resume identity')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        start = old['update']; history = old['history']; best = old['best']; selected = old['selected']
        rng.bit_generator.state = old['rng']; torch.set_rng_state(old['torch_rng'])
    def save(update):
        atomic_save(folder/'last.pt', dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
            update=update, history=history, best=best, selected=selected, rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), arm=arm))
    with online_training(c['wandb'], run_id='meas-'+identity[:16], name=arm['name'],
            config=dict(c, arm=arm), folder=folder, receipt_path=folder/'wandb.json',
            job_type='predictor', group=c['protocol'], receipt_fields=dict(identity=identity)) as tracking:
        for update in range(start+1, cfg['updates']+1):
            if time.time() > deadline-900: save(update-1); return False
            begin = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
            fraction = min(update/cfg['warmup'], 1.)
            decay = .01+.99*.5*(1+math.cos(math.pi*max(0, update-cfg['warmup'])/(cfg['updates']-cfg['warmup'])))
            for group in optimizer.param_groups: group['lr'] = cfg['lr']*fraction*decay
            indices = rng.choice(train, cfg['batch_size']); observed = x[indices]
            steps = int(rng.choice(cfg['horizons_steps'], p=cfg['horizon_probabilities']))
            seed = int(rng.integers(2**31))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(observed[:, :4], steps, cfg['particles'], seed)
            scores = predictive_scores(prediction, observed[:, 4:4+steps], observed[:, 3],
                    weights, increment_scale, cfg['likelihood_bandwidth'])
            objective(scores, cfg, arm['connected']).backward()
            del prediction
            indices = rng.choice(teachers, cfg['teacher_batch'])
            teacher_seed = int(rng.integers(2**31))
            predicted = model(physical['history'][indices], 1, 4, teacher_seed, dense=True)
            value = .5*((predicted.mean(1)-physical['target'][indices].mean(1)).square()*weights).sum(-1).mean()
            (cfg['teacher_weight']*value).backward(); del predicted
            derivative = value.new_zeros(())
            if arm['response']:
                predicted = response_prediction(model, physical['history'][indices], physical['pairs'][indices],
                                                  cfg['response_epsilon'], teacher_seed)
                derivative = response_loss(predicted, physical['response'][indices], normalizers)
                (cfg['response_weight']*derivative).backward(); del predicted
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            row = dict(update=update, horizon_ps=steps*.1, **{k:float(v.mean().detach()) for k,v in scores.items()},
                       teacher_half_mse=float(value.detach()), response_loss=float(derivative.detach()), gradient_norm=float(norm))
            if update % cfg['selection_every'] == 0 or update == cfg['updates']:
                model.eval(); validation = select(model, x, selection, weights, increment_scale, c)
                # All treatments use this same proper marginal predictive selector.
                selector = validation['nll']
                row.update({'selection_'+k:v for k,v in validation.items()})
                with torch.no_grad():
                    for label, ids in [('train', teachers[:20]), ('selection', validation_teachers)]:
                        predicted = response_prediction(model, physical['history'][ids], physical['pairs'][ids],
                                                        cfg['response_epsilon'], c['seed'])
                        row[label+'_response_loss'] = float(response_loss(predicted, physical['response'][ids], normalizers))
                if selector < best:
                    best = selector; selected = update
                    atomic_save(folder/'best.pt', dict(model=model.state_dict(), update=update, identity=identity, arm=arm))
            history.append(row); tracking.log(row, step=update)
            if update == start+1 or update % cfg['save_every'] == 0: save(update)
            write_json(folder/'progress.json', dict(state='training', **row, total=cfg['updates'],
                         seconds_per_update=time.monotonic()-begin))
            if update == start+1 or update % 25 == 0: print(arm['name'], row, flush=True)
        tracking.summary.update(dict(selected_update=selected, selection_nll=best))
    save(cfg['updates'])
    write_metric_rows(history, folder.parent, family=FAMILY, name='learning',
                      columns=list(dict.fromkeys(k for r in history for k in r)))
    write_json(folder/'complete.json', dict(state='complete', identity=identity, selected_update=selected,
               updates=cfg['updates'], checkpoint_sha256=sha(folder/'best.pt')))
    return True
