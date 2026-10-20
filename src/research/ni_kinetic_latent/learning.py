"""Fixed-budget kinetic representation, decoder and stochastic dynamics fits."""
import math
import os
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.temporal_vamp.linear_vamp import LinearVAMP
from .data import load, root, read, atomic_save, write_json, digest, sha, FAMILY
from .model import Model, mlp, vamp_score


def _linear(model, x, c):
    """Train-only covariance estimators; the same left basis encodes every time."""
    features = x.shape[-1]
    present = x[:, 3:54].reshape(-1, features).numpy()
    if model.kind == 'pca':
        mean = present.mean(0, dtype=np.float64)
        covariance = np.zeros((features, features), dtype=np.float64)
        for start in range(0, len(present), 16384):
            centered = present[start:start+16384].astype(np.float64) - mean
            covariance += centered.T @ centered / len(present)
        values, vectors = np.linalg.eigh(covariance)
        matrix = vectors[:, np.argsort(values)[::-1][:model.dimension]]
        details = dict(eigenvalues=values[::-1].tolist())
    else:
        lags = c['representation']['lags']
        future = np.concatenate([x[:, 3+lag:54+lag].reshape(-1, features).numpy()
                                 for lag in lags])
        estimator = LinearVAMP(regularization=c['representation']['ridge'])
        estimator.fit(np.tile(present, (len(lags), 1)), future)
        mean = estimator.mean0_
        matrix = estimator.whitening0_ @ estimator.left_singular_vectors_[:, :model.dimension]
        details = dict(singular_values=estimator.singular_values_.tolist(),
                       ridge0=estimator.ridge0_, ridge1=estimator.ridge1_)
    model.linear_mean.copy_(torch.as_tensor(mean, device=model.linear_mean.device))
    model.linear_matrix.copy_(torch.as_tensor(matrix, device=model.linear_matrix.device))
    return details


def _sources(bank, train_rows):
    names = np.asarray([bank['rows'][i]['source'] for i in train_rows])
    return names, {name: np.flatnonzero(names == name) for name in np.unique(names)}


def representation_loss(model, x, indices, update, rng, groups, weights, c, *, inputs=None):
    inputs = x if inputs is None else inputs
    rows, frames = indices // 51, indices % 51 + 3
    present = x[rows, frames]
    loss = torch.zeros((), device=x.device)
    reconstruction = loss
    score = loss
    kl = loss
    if model.kind == 'vae':
        mean, logvar = model.posterior(inputs[rows, frames])
        latent = mean + torch.exp(.5 * logvar) * torch.randn_like(mean)
        reconstruction = ((model.decode(latent) - present).square() * weights).sum(-1).mean()
        kl = .5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        loss = reconstruction + c['representation']['beta'] * kl
    else:
        latent = model.encode(inputs[rows, frames])
        lags = c['representation']['lags']
        lag = lags[(update - 1) % len(lags)]
        future_rows, future_frames = rows, frames + lag
        if model.kind == 'shuffle':
            names, by_source = groups
            future_rows = np.empty_like(rows)
            for name in np.unique(names[rows]):
                selected = names[rows] == name
                future_rows[selected] = rng.choice(by_source[name], int(selected.sum()))
            future_frames = rng.integers(3, 54, size=len(rows)) + lag
        future = model.encode(inputs[future_rows, future_frames])
        score = vamp_score(latent, future, c['representation']['ridge'])
        loss = -score
        if model.kind == 'hybrid':
            reconstruction = ((model.decode(latent) - present).square() * weights).sum(-1).mean()
            loss = loss + reconstruction
    return loss, dict(reconstruction=reconstruction, vamp_score=score, kl=kl,
                      latent_variance=latent.var(0, unbiased=False).mean())


@torch.no_grad()
def _latents(model, x):
    shape = x.shape
    result = torch.cat([model.encode(chunk) for chunk in x.reshape(-1, shape[-1]).split(8192)])
    return result.reshape(*shape[:-1], -1)


def fitted(c, arm, *, model_factory=Model):
    folder = root(c) / 'analyses' / arm['name'] / 'technical'
    done = read(folder / 'complete.json')
    if sha(folder / 'final.pt') != done['checkpoint_sha256']:
        raise ValueError(f"Changed final kinetic checkpoint: {arm['name']}")
    saved = torch.load(folder / 'final.pt', map_location='cuda', weights_only=False)
    expected = digest([read(root(c) / 'technical/plan.json')['identity'], arm])
    if saved['identity'] != done['identity'] or saved['identity'] != expected:
        raise ValueError('Kinetic checkpoint identity differs from completion receipt')
    model = model_factory(c, arm['kind']).cuda()
    model.load_state_dict(saved['model'])
    model.eval()
    return model, saved


def fit(c, arm, deadline, *, load_provider=load, model_factory=Model, family=FAMILY,
        input_key='x', run_prefix='nikinetic', input_normalization_key=None):
    folder = root(c) / 'analyses' / arm['name'] / 'technical'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return True
    plan, bank = load_provider(c)
    train_rows = np.asarray(bank['train_rows'])
    x_cpu = bank['x'][train_rows].contiguous()
    input_cpu = x_cpu if input_key == 'x' else bank[input_key][train_rows].contiguous()
    weights = torch.as_tensor(bank['weights'], device='cuda')
    groups = _sources(bank, train_rows)
    identity = digest([plan['identity'], arm])
    torch.manual_seed(arm['seed'])
    model = model_factory(c, arm['kind']).cuda()
    rng = np.random.default_rng(arm['seed'])
    settings = c['training']
    rows = []
    completed = []
    global_update = 0
    previous = None
    if (folder / 'last.pt').exists():
        previous = torch.load(folder / 'last.pt', map_location='cuda', weights_only=False)
        if previous['identity'] != identity:
            raise ValueError(f"Changed resumed kinetic fit: {arm['name']}")
        model.load_state_dict(previous['model'])
        rng.bit_generator.state = previous['rng']
        torch.set_rng_state(previous['torch_rng'].cpu())
        torch.cuda.set_rng_state(previous['cuda_rng'].cpu())
        completed = previous['completed']
        rows, global_update = previous['history'], previous['global_update']
    stage = 'representation'
    stage_update = 0
    optimizer = None

    def save(final=False):
        payload = dict(identity=identity, model=model.state_dict(), arm=arm,
                       optimizer=optimizer.state_dict() if optimizer is not None else None,
                       stage=stage, stage_update=stage_update, update=global_update,
                       global_update=global_update, completed=completed, history=rows,
                       rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state(), data_identity=plan['data_identity'],
                       input_key=input_key, input_dimension=input_cpu.shape[-1],
                       input_normalization=(bank[input_normalization_key]
                                            if input_normalization_key else None),
                       normalization={key: bank[key].cpu() for key in
                                      ('raw_mean', 'raw_scale', 'mean', 'scale', 'weights', 'active')})
        atomic_save(folder / ('final.pt' if final else 'last.pt'), payload)

    write_json(folder / 'execution-segments' / f'{time.time_ns()}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(),
        train_sequences=len(x_cpu), parameters=sum(p.numel() for p in model.parameters()),
        precision='FP32 neural operations; FP64 VAMP covariance/Cholesky',
        resumed_update=global_update, checkpoint_rule='Final fixed epoch budgets only',
        encoder_input_dimension=input_cpu.shape[-1], decoder_target_dimension=x_cpu.shape[-1],
        input='Current measurements/context for encoder; last four raw latent states for propagator',
        arm=arm))
    with online_training(c['wandb'], run_id=run_prefix + '-' + identity[:16],
            name=arm['name'], config=dict(c, arm=arm), folder=folder,
            receipt_path=folder / 'wandb.json', job_type='predictor',
            group=c['protocol'], receipt_fields=dict(identity=identity)) as tracking:
        if arm['kind'] in ('pca', 'linear_vamp') and 'representation' not in completed:
            if time.time() > deadline - 300:
                save()
                return False
            details = _linear(model, input_cpu, c)
            write_json(folder / 'linear-fit.json', details)
            completed.append('representation')
            save()
        x = x_cpu.cuda()
        inputs = x if input_key == 'x' else input_cpu.cuda()
        latent = None
        for stage_index, stage in enumerate(('representation', 'decoder', 'dynamics')):
            if stage in completed:
                continue
            start = (previous['stage_update'] if previous is not None and
                     previous['stage'] == stage else 0)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if stage == 'representation':
                parameters = list(model.encoder.parameters()) + list(model.location.parameters())
                if arm['kind'] == 'vae':
                    parameters += list(model.log_variance.parameters())
                if arm['kind'] in ('vae', 'hybrid'):
                    parameters += list(model.decoder.parameters())
                epoch_count = settings['repr_epochs']
                observations = len(x) * 51
            else:
                if latent is None:
                    latent = _latents(model, inputs)
                if stage == 'decoder':
                    if start == 0:
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            torch.manual_seed(arm['seed'] + 207)
                            model.decoder = mlp(model.dimension, c['model']['width'], 244).cuda()
                    parameters = list(model.decoder.parameters())
                    epoch_count = settings['decoder_epochs']
                    observations = len(x) * 51
                else:
                    if start == 0:
                        flat = latent.reshape(-1, model.dimension)
                        model.transition.location.copy_(flat.mean(0))
                        model.transition.scale.copy_(flat.std(0, unbiased=False).clamp_min(1e-5))
                    parameters = list(model.transition.parameters())
                    epoch_count = settings['dynamics_epochs']
                    observations = len(x) * 60
            for parameter in parameters:
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(parameters, lr=settings['lr'], weight_decay=1e-5)
            if start:
                optimizer.load_state_dict(previous['optimizer'])
            batches = math.ceil(observations / settings['batch_size'])
            total = batches * epoch_count
            warmup = batches * settings['warmup_epochs']
            order, order_epoch = None, -1
            model.train()
            for stage_update in range(start + 1, total + 1):
                if time.time() > deadline - 300:
                    stage_update -= 1
                    save()
                    return False
                begin = time.monotonic()
                epoch, offset = divmod(stage_update - 1, batches)
                if epoch != order_epoch:
                    order = np.random.default_rng(arm['seed'] + 100000 * stage_index + epoch).permutation(observations)
                    order_epoch = epoch
                indices = order[offset * settings['batch_size']:(offset + 1) * settings['batch_size']]
                warm = min(stage_update / warmup, 1.)
                cosine = .01 + .99 * .5 * (1 + math.cos(math.pi * max(0, stage_update - warmup) / (total - warmup)))
                for group in optimizer.param_groups:
                    group['lr'] = settings['lr'] * warm * cosine
                optimizer.zero_grad(set_to_none=True)
                if stage == 'representation':
                    loss, scores = representation_loss(model, x, indices, stage_update, rng, groups, weights, c,
                                                       inputs=inputs)
                elif stage == 'decoder':
                    row_ids, frames = indices // 51, indices % 51 + 3
                    error = model.decode(latent[row_ids, frames]) - x[row_ids, frames]
                    loss = (error.square() * weights).sum(-1).mean()
                    scores = dict(reconstruction=loss)
                else:
                    row_ids, frames = indices // 60, indices % 60 + 3
                    history = latent[row_ids[:, None], frames[:, None] + np.arange(-3, 1)]
                    loss = -model.transition.log_prob(history, latent[row_ids, frames + 1]).mean() / model.dimension
                    scores = dict(nll_per_dimension=loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {stage} objective for {arm['name']} at {stage_update}")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
                optimizer.step()
                global_update += 1
                record = dict(update=global_update, stage=stage, stage_update=stage_update,
                              epoch=epoch + 1, loss=float(loss.detach()),
                              gradient_norm=float(gradient), lr=optimizer.param_groups[0]['lr'],
                              **{key: float(value.detach()) for key, value in scores.items()})
                rows.append(record)
                tracking.log({f'{stage}/{key}': value for key, value in record.items()
                              if key != 'stage'}, step=global_update)
                if stage_update == start + 1 or stage_update % settings['save_every'] == 0:
                    save()
                write_json(folder / 'progress.json', dict(state='training', **record,
                    stage_total=total, completed=completed, seconds_per_update=time.monotonic() - begin))
                if stage_update == start + 1 or stage_update % 100 == 0:
                    print(arm['name'], record, flush=True)
            completed.append(stage)
            save()
            previous = None
        final_latent = _latents(model, inputs).reshape(-1, model.dimension).double()
        centered = final_latent - final_latent.mean(0)
        spectrum = torch.linalg.eigvalsh(centered.T @ centered / len(centered)).clamp_min(0)
        effective_rank = float(spectrum.sum().square() / spectrum.square().sum().clamp_min(1e-30))
        write_json(folder / 'representation-rank.json', dict(
            covariance_eigenvalues=spectrum.cpu().tolist(), participation_rank=effective_rank,
            scope='All 64 observations of training sequences only'))
        tracking.summary.update(dict(final_update=global_update, checkpoint_rule='fixed_final_epochs',
                                     representation=arm['kind'], completed_stages=completed,
                                     train_participation_rank=effective_rank))
    save(final=True)
    save()
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    write_metric_rows(rows, folder.parent, family=family, name='learning', columns=columns)
    write_json(folder / 'complete.json', dict(identity=identity, updates=global_update,
        completed_stages=completed, checkpoint_sha256=sha(folder / 'final.pt')))
    return True


def benchmark(c, *, load_provider=load, model_factory=Model, input_key='x',
              kinds=('pca', 'linear_vamp', 'vae', 'vamp', 'hybrid', 'shuffle'),
              report_name='benchmark'):
    """Train-input integration check and timings; no online scientific run."""
    _, bank = load_provider(c)
    train_rows = np.asarray(bank['train_rows'])
    # Enough independent real sequences for covariance fitting, without the full training budget.
    selected = train_rows[:min(256, len(train_rows))]
    cpu = bank['x'][selected].contiguous()
    x = cpu.cuda()
    input_cpu = cpu if input_key == 'x' else bank[input_key][selected].contiguous()
    inputs = x if input_key == 'x' else input_cpu.cuda()
    weights = torch.as_tensor(bank['weights'], device='cuda')
    groups = _sources(bank, selected)
    report = []
    for kind in kinds:
        torch.manual_seed(c['seed'])
        model = model_factory(c, kind).cuda()
        rng = np.random.default_rng(c['seed'])
        if kind in ('pca', 'linear_vamp'):
            begin = time.monotonic()
            _linear(model, input_cpu, c)
            report.append(dict(kind=kind, stage='covariance', seconds=time.monotonic() - begin))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for stage in (('decoder', 'dynamics') if kind in ('pca', 'linear_vamp')
                      else ('representation', 'decoder', 'dynamics')):
            if stage == 'dynamics':
                flat = _latents(model, inputs).reshape(-1, model.dimension)
                model.transition.location.copy_(flat.mean(0))
                model.transition.scale.copy_(flat.std(0, unbiased=False).clamp_min(1e-5))
            seconds = []
            for repeat in range(2):
                indices = rng.integers(len(x) * 51, size=c['training']['batch_size'])
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                if stage == 'representation':
                    loss, _ = representation_loss(model, x, indices, repeat + 1, rng, groups, weights, c,
                                                  inputs=inputs)
                elif stage == 'decoder':
                    current = x[indices // 51, indices % 51 + 3]
                    encoded = model.encode(inputs[indices // 51, indices % 51 + 3]).detach()
                    loss = ((model.decode(encoded) - current).square() * weights).sum(-1).mean()
                else:
                    rr, tt = indices // 51, indices % 51 + 3
                    with torch.no_grad():
                        history = model.encode(inputs[rr[:, None], tt[:, None] + np.arange(-3, 1)])
                        following = model.encode(inputs[rr, tt + 1])
                    loss = -model.transition.log_prob(history, following).mean() / model.dimension
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                seconds.append(time.monotonic() - begin)
            record = dict(kind=kind, stage=stage, seconds=seconds, loss=float(loss.detach()),
                          gradient_norm=float(gradient), memory_GiB=torch.cuda.max_memory_allocated()/2**30)
            print(record, flush=True)
            report.append(record)
        with torch.no_grad():
            z, decoded = model.rollout(inputs[:2, :4], 60, 4, c['seed'])
            mean_z, _ = model.rollout(inputs[:2, :4], 60, 1, c['seed'], deterministic=True)
            if not torch.isfinite(z).all() or not torch.isfinite(decoded).all() or not torch.isfinite(mean_z).all():
                raise FloatingPointError(f'Nonfinite local rollout integration check: {kind}')
            report.append(dict(kind=kind, rollout_shape=list(z.shape),
                               decoded_shape=list(decoded.shape), particles_differ=bool((z[:, 0]-z[:, 1]).abs().max() > 0)))
        del model, optimizer
        torch.cuda.empty_cache()
    write_json(root(c) / 'technical' / f'{report_name}.json', dict(
        gpu=torch.cuda.get_device_name(), scope='Local train-input integration/timing only',
        encoder_input_dimension=inputs.shape[-1], decoder_target_dimension=x.shape[-1], rows=report))
