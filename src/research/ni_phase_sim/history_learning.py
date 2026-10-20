"""Matched anchor-start budgets for classic and history-attention models."""
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from .data import root, read, normalization, digest, write_json, atomic_save, sha
from .history_data import load, FAMILY
from .history_model import HistoryProcess
from .learning import objective, batch


def loss_for(model, state, tangent, history, ids, length, k, norm, arm, c):
    # Exact observed history always ends at the fixed physical anchor.
    args = batch(state, tangent, ids, np.zeros(len(ids), dtype=int), length, k)
    bound = model.condition(history[ids])
    return objective(bound, *args, norm, arm, c)


def fit(c, arm, deadline, *, model_factory=HistoryProcess, loss_function=loss_for,
        family=FAMILY, run_prefix='phasehist'):
    folder = root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        return True
    p, records, state, tangent, history = load(c, 'train'); norm = normalization(state, tangent)
    settings = c['training']; total = settings['updates']; n = len(state)
    torch.manual_seed(arm['seed']); model = model_factory(c, arm).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=1e-5)
    rng = np.random.default_rng(arm['seed']); identity = digest([p['identity'], arm]); rows = []; start = 0
    if (folder/'last.pt').exists():
        previous = torch.load(folder/'last.pt', map_location='cuda', weights_only=False)
        if previous['identity'] != identity:
            raise ValueError('Changed resumed history fit')
        model.load_state_dict(previous['model']); optimizer.load_state_dict(previous['optimizer'])
        rng.bit_generator.state = previous['rng']; start = previous['update']; rows = previous['history']
    def save(update, final=False):
        atomic_save(folder/('final.pt' if final else 'last.pt'), dict(identity=identity,
            model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
            history=rows, update=update, arm=arm,
            normalization={k: v.detach().cpu() for k, v in norm.items()}))
    write_json(folder/'execution-segments'/f'{time.time_ns()}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), start_update=start,
        parameters=sum(v.numel() for v in model.parameters()), train_states=n,
        train_ancestors=len({r['ancestor'] for r in records}), precision='FP32 neural state, attention, JVP, loss',
        checkpoint_rule='Final fixed8000; no test selection', arm=arm))
    with online_training(c['wandb'], run_id=run_prefix+'-'+identity[:16], name=arm['name'],
            config=dict(c, arm=arm), folder=folder, receipt_path=folder/'wandb.json',
            job_type='predictor', group=c['protocol'], receipt_fields=dict(identity=identity)) as tracking:
        for update in range(start+1, total+1):
            if time.time() > deadline-360:
                save(update-1); return False
            begin = time.monotonic(); optimizer.zero_grad(set_to_none=True)
            model.fit_update = update
            warm = min(update/settings['warmup'], 1.)
            cosine = .01+.99*.5*(1+math.cos(math.pi*max(0, update-settings['warmup'])/(total-settings['warmup'])))
            for group in optimizer.param_groups:
                group['lr'] = settings['lr']*warm*cosine
            length = int(rng.choice(settings['unroll_steps'], p=settings['unroll_probabilities']))
            k = int(rng.integers(2)); norm['selected_power'] = norm['response_power'][:length, :, k]
            size = settings['batch_size']; ids = rng.integers(n, size=size); totals = {}
            for offset in range(0, size, settings['microbatch']):
                chosen = ids[offset:offset+settings['microbatch']]; weight = len(chosen)/size
                loss, scores = loss_function(model, state, tangent, history, chosen, length, k, norm, arm, c)
                (weight*loss).backward()
                for key, value in scores.items():
                    totals[key] = totals.get(key, 0.)+weight*float(value.detach())
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            row = dict(update=update, unroll_steps=length, query=k,
                loss=sum(settings['loss_weights'][k]*v for k, v in totals.items()), gradient_norm=float(grad),
                **{name: totals.get(name, 0.) for name in settings['loss_weights']})
            rows.append(row); tracking.log(row, step=update)
            if update == start+1 or update % settings['save_every'] == 0:
                save(update)
            write_json(folder/'progress.json', dict(state='training', **row, total=total, seconds_per_update=time.monotonic()-begin))
            if update == start+1 or update % 50 == 0:
                print(arm['name'], row, flush=True)
        tracking.summary.update(dict(final_update=total, checkpoint_rule='fixed_final'))
    save(total); save(total, final=True)
    write_metric_rows(rows, folder.parent, family=family, name='learning')
    write_json(folder/'complete.json', dict(identity=identity, updates=total, checkpoint_sha256=sha(folder/'final.pt')))
    return True


def benchmark(c):
    """Real training inputs; integration and timing checks without online runs."""
    from src.research.continuous_motion.data import Source
    from .history_data import fixed_history
    p = read(root(c)/'technical/plan.json'); packets = []; histories = []
    for job in [j for j in p['jobs'] if j['role'] == 'train'][:2]:
        packet = torch.load(Path(c['phase_archive'])/job['key']/'path.pt', map_location='cpu', weights_only=False)
        reader = Source(next(s['source'] for s in p['sources'] if s['key'] == job['source_key']))
        packets.append(packet); histories.append(fixed_history(reader, job['frame'], packet['atom_ids'][None])[0])
    scale = torch.tensor([5., 5., 5., 10., 10., 10.], device='cuda')
    state = torch.stack([v['state'] for v in packets]).cuda().float()/scale
    tangent = torch.stack([v['tangent'] for v in packets]).cuda().float()/scale[:, None]
    history = torch.stack(histories).cuda()/scale
    norm = normalization(state, tangent); report = []
    candidates = [a for a in c['arms'] if a['seed'] == c['seed'] and a['name'].split('-s')[0] in ['classic-full', 'attn1-full', 'attn4-values', 'attn4-full']]
    for arm in candidates:
        torch.manual_seed(arm['seed']); model = HistoryProcess(c, arm).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for length in [4, 20]:
            timings = []
            for repeat in range(2):
                optimizer.zero_grad(set_to_none=True); norm['selected_power'] = norm['response_power'][:length, :, 0]
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin = time.monotonic()
                size = c['training']['batch_size']; micro = c['training']['microbatch']
                for offset in range(0, size, micro):
                    ids = np.arange(min(micro, size-offset)) % 2
                    loss, _ = loss_for(model, state, tangent, history, ids, length, 0, norm, arm, c)
                    (loss*len(ids)/size).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize(); timings.append(time.monotonic()-begin)
            report.append(dict(arm=arm['name'], steps=length, seconds=timings, loss=float(loss.detach()),
                memory_GiB=torch.cuda.max_memory_allocated()/2**30, parameters=sum(v.numel() for v in model.parameters())))
            print(report[-1], flush=True)
        if arm['architecture'] == 'attention':
            gradients = {name: float(torch.stack([v.grad.norm() for v in getattr(model, name).parameters()
                         if v.grad is not None]).norm())
                         for name in ['encoder_attention', 'decoder_attention', 'readout_mix']}
            if any(value == 0 or not math.isfinite(value) for value in gradients.values()):
                raise ValueError(f'History modules have missing/nonfinite gradients: {gradients}')
            report.append(dict(arm=arm['name'], gradient_norms=gradients))
        h = history[:1]; x = state[:1, 0]; u = tangent[:1, 0, ..., 0]
        with torch.no_grad():
            bound = model.condition(h)
            _, jvp = torch.func.jvp(lambda y: bound(y, 4), (x,), (u,))
            secant = (bound(x+.1*u, 4)-bound(x-.1*u, 4))/.2
            error = float((jvp-secant).norm()/jvp.norm().clamp_min(1e-8))
            rotation = torch.linalg.qr(torch.randn(3, 3, device='cuda'))[0]
            rotated = (h.reshape(1, 4, 80, 2, 3) @ rotation).flatten(-2)
            expected = (bound(x, 4).reshape(1, 4, 80, 2, 3) @ rotation).flatten(-2)
            obtained = model.condition(rotated)(rotated[:, -1], 4)
            equivariance = float((expected-obtained).abs().max())
            # History dependence is required for the four-frame architecture.
            changed = h.clone(); changed[:, :-1, :, 3:] *= -1
            dependence = float((bound(x, 4)-model.condition(changed)(x, 4)).abs().max())
        if error > .02 or equivariance > 1e-4:
            raise ValueError(f'History model derivative/equivariance audit: {error}, {equivariance}')
        if arm['history_frames'] == 4 and dependence == 0:
            raise ValueError('Model ignores past observations')
        report.append(dict(arm=arm['name'], jvp_secant_relative_error=error,
            rotation_max_error=equivariance, history_intervention_max_change=dependence))
        del model, optimizer, bound; torch.cuda.empty_cache()
    write_json(root(c)/'technical/benchmark.json', dict(gpu=torch.cuda.get_device_name(), checks=report,
        implementations={str(Path(f).name): sha(Path(f)) for f in [__file__, str(Path(__file__).with_name('history_model.py'))]}))
