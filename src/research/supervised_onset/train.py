"""End-to-end predictive likelihood training; AP is an evaluation diagnostic."""
import copy
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F

from src.models.encoders.graph_bank import GraphBank
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.local_predictability.metrics import source_weights, hazard_loss, cumulative_risk, weighted_scores
from .common import write_json, save_checkpoint, sha
from .data import sampling_distribution
from .model import Model
from .tracking import risk_diagnostics, validation_fields, training_record, population_summary


def encoder_config(study, corpus):
    if study.config.get('initial_encoder'):
        return dict(study.config['encoder'],d0=2.8,n_ref=80.)
    return dict(study.config['encoder'], **{k: corpus.manifest['domains']['hot'][k] for k in ('d0', 'n_ref')})


def make_model(study, corpus, arm, device):
    torch.manual_seed(study.config['seed'])
    model = Model(encoder_config(study, corpus), arm).to(device)
    actual = sum(p.numel() for p in model.encoder.parameters())
    if actual != study.config['parameter_budget']['encoder_actual']:
        raise ValueError(f'Encoder parameter count changed: {actual}, expected {study.config["parameter_budget"]}')
    return model


def make_banks(study, corpus, device):
    model = make_model(study, corpus, study.config['arms'][0], device)
    banks = {d: GraphBank(a, model.encoder, device,
        plan_cache_bytes=study.config['runtime']['plan_cache_MiB']*2**20,
        node_capacity=80*study.config['training']['microbatch']) for d, a in corpus.arrays.items()}
    return banks


def configure_runtime(study, model, banks, corpus):
    if study.config['runtime']['compile']:
        domain = 'hot' if model.input == 'paired' else model.input
        ids = corpus.split['train'][:study.config['training']['microbatch']]
        compile_spatial_encoder(model.encoder, banks[domain].batch(ids))


@torch.no_grad()
def encode(model, banks, ids, chunk):
    model.eval()
    model.prepare_pass(banks, ids, chunk)
    return torch.cat([model(banks, ids[s:s+chunk]) for s in range(0, len(ids), chunk)])


def supervised_step(model, banks, ids, importance, events, config, teacher=None):
    model.train()
    result = dict(nll=0., distillation=0.)
    chunk = config['microbatch']
    for start in range(0, len(ids), chunk):
        rows = ids[start:start+chunk]
        z = model(banks, rows)
        logits = model.logits(z)
        per_row = hazard_loss(logits, events[rows])
        weights = importance[start:start+len(rows)]
        nll = (weights * per_row).sum() / len(ids)
        loss = nll
        if teacher is not None:
            latent = (model.teacher_projection(z) - teacher['features'][rows]).square().mean(1)
            soft = F.binary_cross_entropy_with_logits(logits, teacher['hazards'][rows], reduction='none').mean(1)
            distill = config['teacher_weight'] * (weights * (latent + soft)).sum() / len(ids)
            loss = loss + distill
            result['distillation'] += float(distill.detach())
        loss.backward()
        result['nll'] += float(nll.detach())
    return result


@torch.no_grad()
def validate(model, banks, corpus, chunk):
    ids = corpus.split['selection']
    z = encode(model, banks, ids, chunk)
    logits = model.logits(z)
    risk = cumulative_risk(logits).cpu().numpy()
    pop = corpus.pop
    weights = torch.as_tensor(source_weights(pop['source'][ids]), dtype=logits.dtype, device=logits.device)
    event = torch.as_tensor(pop['event'][ids], device=logits.device)
    result = dict(nll=float(weights @ hazard_loss(logits, event)))
    result.update(risk_diagnostics(pop['event'][ids],risk,pop['source'][ids]))
    if not np.isfinite(list(result.values())).all():
        raise FloatingPointError(f'Nonfinite validation scores: {result}')
    return result


def selection_key(scores):
    """Smaller held-out likelihood loss wins; AP never breaks ties."""
    value = float(scores['nll'])
    if not math.isfinite(value):
        raise FloatingPointError(f'Nonfinite selection NLL: {scores}')
    return value


@torch.no_grad()
def initialize(model, banks, corpus, chunk, *, preserve_encoder=False):
    fit = corpus.split['train']
    domains = ('hot', 'cold') if model.input == 'paired' else (model.input,)
    pooled = torch.cat([torch.cat([model.encoder.pooled_graph(banks[d].batch(fit[s:s+chunk]))
        for s in range(0, len(fit), chunk)]) for d in domains])
    w = torch.as_tensor(np.tile(source_weights(corpus.pop['source'][fit]), len(domains))/len(domains),
                        dtype=pooled.dtype, device=pooled.device)
    mean = w @ pooled
    scale = (w @ (pooled-mean).square()).sqrt().clamp_min(1e-5)
    if not preserve_encoder:
        model.encoder.pooled_mean.copy_(mean)
        model.encoder.pooled_scale.copy_(scale)
    event = corpus.pop['event'][fit]
    weight = source_weights(corpus.pop['source'][fit])
    prior = np.array([(weight @ (event == k)) / (weight @ (event >= k)) for k in range(5)])
    model.hazard[-1].weight.zero_()
    model.hazard[-1].bias.copy_(torch.logit(torch.as_tensor(prior.clip(1e-6, 1-1e-6), device=pooled.device)))


def teacher_targets(study, corpus, banks, device):
    """Freeze the screen teacher; only fit-row features become student targets."""
    path = study.technical / 'teacher-fit.npz'
    checkpoint = study.technical / 'runs' / study.config['teacher_arm'] / 'screen-best.pt'
    checksum = sha(checkpoint)
    if not path.exists():
        model = make_model(study, corpus, study.arm(study.config['teacher_arm']), device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['identity'] != study.identity:
            raise ValueError('Teacher checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        fit = corpus.split['train']
        z = encode(model, banks, fit, study.config['training']['microbatch'])
        w = torch.as_tensor(source_weights(corpus.pop['source'][fit]), dtype=z.dtype, device=device)
        mean = w @ z
        scale = (w @ (z-mean).square()).sqrt().clamp_min(1e-5)
        with torch.no_grad():
            hazards = model.logits(z).sigmoid()
        np.savez(path, ids=fit, features=((z-mean)/scale).cpu().numpy(), hazards=hazards.cpu().numpy(),
                 mean=mean.cpu().numpy(), scale=scale.cpu().numpy())
        write_json(path.with_suffix('.json'), dict(checkpoint_sha256=checksum, rows=len(fit),
            selection='screen minimum NLL teacher; no student test/calibration targets', identity=study.identity,
            targets_sha256=sha(path)))
    receipt = json.loads(path.with_suffix('.json').read_text())
    if receipt['checkpoint_sha256'] != checksum or receipt['targets_sha256'] != sha(path):
        raise ValueError('Frozen teacher targets changed')
    with np.load(path) as a:
        np.testing.assert_array_equal(a['ids'], corpus.split['train'])
        result = {}
        for key in ('features', 'hazards'):
            value = torch.zeros((len(corpus.pop['source']), a[key].shape[1]), device=device)
            value[corpus.split['train']] = torch.as_tensor(a[key], device=device)
            result[key] = value
    return result


def fit(study, corpus, banks, name, *, until, max_updates, device):
    """Continue the optimizer/RNG trajectory and preserve minimum selection NLL."""
    from .tracking import tracked_run
    with tracked_run(study, name) as tracking:
        return _fit(study, corpus, banks, name, until=until, max_updates=max_updates,
                    device=device, tracking=tracking)


def _fit(study, corpus, banks, name, *, until, max_updates, device, tracking):
    arm = study.arm(name)
    c = study.config['training']
    root = study.technical / 'runs' / name
    root.mkdir(parents=True, exist_ok=True)
    model = make_model(study, corpus, arm, device)
    other = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer = torch.optim.AdamW([dict(params=model.encoder.parameters(), lr=c['encoder_lr']),
        dict(params=other, lr=c['head_lr'])], weight_decay=c['weight_decay'])
    rng = np.random.default_rng(study.config['seed'])
    events = torch.as_tensor(corpus.pop['event'], device=device)
    fit_ids = corpus.split['train']
    q, iw = sampling_distribution(corpus.pop['source'][fit_ids], corpus.pop['event'][fit_ids], c['positive_sampling_fraction'])
    teacher = teacher_targets(study, corpus, banks, device) if arm['teacher'] else None
    start, best = 0, None
    checkpoint_path = root / 'last.pt'
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved['identity'] != study.identity or saved['arm'] != arm:
            raise ValueError(f'Cannot resume changed experiment: {name}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        start, best = saved['update'], saved['best']
    else:
        initial=study.config.get('initial_encoder')
        if initial:
            from src.project_runtime.paths import resolve_path
            path=resolve_path(initial['checkpoint'])
            saved=torch.load(path,map_location=device,weights_only=False)
            receipt=json.loads((path.parent/'complete.json').read_text())
            if receipt['identity']!=saved['identity'] or receipt['sha256']!=sha(path):
                raise ValueError('Pretraining completion receipt differs from checkpoint')
            if saved['method']!=initial['method'] or saved['epoch']<12 or saved['release_identity']!=study.config['fixed_dataset']['identity']:
                raise ValueError('Pretraining checkpoint method, epoch or dataset identity differs')
            model.encoder.load_state_dict(saved['encoder'],strict=True)
            write_json(root/'initial-encoder.json',dict(path=str(path),sha256=sha(path),
                method=saved['method'],epoch=saved['epoch'],release_identity=saved['release_identity']))
        initialize(model, banks, corpus, c['microbatch'],preserve_encoder=bool(initial))
    configure_runtime(study, model, banks, corpus)
    population_summary(tracking,corpus,study.config)
    tracking.summary['model/encoder_parameters']=sum(p.numel() for p in model.encoder.parameters())

    def save(update, path):
        save_checkpoint(path, dict(identity=study.identity, branch=study.config['branch'],
            arm=arm, encoder_config=encoder_config(study, corpus), model=model.state_dict(),
            optimizer=optimizer.state_dict(), update=update, best=best,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(), config=study.config))

    def assess(update):
        nonlocal best
        scores = validate(model, banks, corpus, c['microbatch'])
        entry = dict(scores, update=update)
        minimum=c.get('minimum_selection_epoch',0)*math.ceil(len(fit_ids)/c['batch_size'])
        if update>=minimum and (best is None or selection_key(entry) < selection_key(best)):
            best = entry
            save(update, root / 'best.pt')
        with (root / 'selection.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        print(json.dumps(dict(arm=name, stage='selection', **entry)), flush=True)
        tracking.log(dict(optimizer_update=update,**validation_fields(scores)))
        if best is not None:
            tracking.summary['checkpoint/selected_update']=best['update']
            tracking.summary['checkpoint/validation_event_nll']=best['nll']
        return entry

    if best is None:
        assess(start)
        save(start, checkpoint_path)
    started = time.monotonic()
    completed, last_assessed = start, start
    epoch_stream=None
    if 'epochs' in c:
        from src.research.encoder_context.epochs import batches,epoch_weights
        steps_per_epoch=math.ceil(len(fit_ids)/c['batch_size'])
        if max_updates!=steps_per_epoch*c['epochs']:
            raise ValueError('Update budget must contain the declared complete epochs')
        epoch_stream=iter(batches(np.arange(len(fit_ids)),c['batch_size'],c['epochs'],study.config['seed'],start))
        full_weights=epoch_weights(corpus.pop['source'][fit_ids])
        tracking.summary['training/epochs_requested']=c['epochs']
        tracking.summary['training/steps_per_epoch']=steps_per_epoch
        tracking.summary['training/sampling']='each training row once per shuffled epoch; equal-source loss weights'
    for update in range(start, max_updates):
        if time.time() >= until - 90:
            break
        factor = min((update+1)/c['warmup'], 1.) / math.sqrt(max((update+1)/c['screen_updates'], 1.))
        for group, lr in zip(optimizer.param_groups, (c['encoder_lr'], c['head_lr']), strict=True):
            group['lr'] = lr * factor
        optimizer.zero_grad(set_to_none=True)
        if epoch_stream is None:
            drawn=rng.choice(len(fit_ids),size=c['batch_size'],replace=True,p=q)
            importance=iw[drawn]
        else:
            step,drawn=next(epoch_stream)
            if step!=update:raise ValueError('Epoch resume cursor differs')
            importance=full_weights[drawn]
        ids = fit_ids[drawn]
        weights = torch.as_tensor(importance, dtype=torch.float32, device=device)
        record = supervised_step(model, banks, ids, weights, events, c, teacher)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        completed = update+1
        if not np.isfinite(list(record.values())).all():
            raise FloatingPointError(f'Nonfinite training loss: {name}, update={completed}, {record}')
        if completed % c['log_every'] == 0:
            record.update(update=completed, gradient_norm=float(norm),
                seconds=time.monotonic()-started, updates_this_stage=completed-start)
            with (root / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(dict(arm=name, stage='train', **record)), flush=True)
            training_record(tracking,record,optimizer)
            if epoch_stream is not None:tracking.log({'optimizer_update':completed,'train/epoch':completed/steps_per_epoch})
        if completed % c['evaluate_every'] == 0:
            assess(completed)
            last_assessed = completed
        if completed % c['save_every'] == 0:
            save(completed, checkpoint_path)
        if epoch_stream is not None and completed==12*steps_per_epoch:
            save(completed,root/'epoch-012.pt')
    if completed != last_assessed:
        assess(completed)
    save(completed, checkpoint_path)
    result = dict(arm=name, updates=completed, requested_updates=max_updates, best=best,
        state='update_budget_complete' if completed >= max_updates else 'time_budget_checkpointed',
        seconds=time.monotonic()-started, identity=study.identity)
    if epoch_stream is not None:result['completed_epochs']=completed/steps_per_epoch
    write_json(root / 'training-state.json', result)
    tracking.summary['training'] = result
    return result
