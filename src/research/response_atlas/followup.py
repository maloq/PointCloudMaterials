"""Stronger toy controls and fixed-budget atomistic response precision."""
import argparse
import copy
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.experiment_runner.wandb_tracking import DEFAULTS, online_training
from src.project_runtime.paths import resolve_path
from .common import bind, output, read
from .mechanisms import gaussian, exact, exact_response, jacobian
from .reference import AutogradOracle, SmallAtlas, unbiased_gram


def export(c, name, rows):
    return write_metric_rows(rows, output(c) / 'analyses/followup-v1',
                             family='response_atlas_followup', name=name)


def toy(c, index):
    torch.set_num_threads(2)
    seed = c['fit_seeds'][index]
    gen = torch.Generator().manual_seed(c['seed'])
    x = torch.rand(192, 2, generator=gen, dtype=torch.float64) * 2 - 1
    evaluation = torch.rand(c['toy_test_points'], 2,
        generator=torch.Generator().manual_seed(c['toy_test_seed']), dtype=torch.float64) * 2 - 1
    oracle = AutogradOracle(gaussian)
    # Shared higher-precision selection observations, never response-selected.
    began = time.monotonic()
    selection = torch.stack([torch.stack([gaussian(q, 30_000_000 + i * 1000 + b)
        for b in range(c['toy_selection_shots'])]).mean(0) for i, q in enumerate(x[128:])])
    selection_seconds = time.monotonic() - began
    labels = {}; acquisition = {}; responses = None
    for arm, shots in [('values8', 8), ('values32', 32), ('responses8', 8)]:
        began = time.monotonic(); values = []; derivatives = []
        for i, q in enumerate(x[:128]):
            seeds = range(4_000_000 + i * 100, 4_000_000 + i * 100 + shots)
            if arm == 'responses8':
                bundle = oracle.query(q, torch.eye(2, dtype=q.dtype), seeds)
                values.append(bundle.values.mean(0)); derivatives.append(bundle.responses.mean(0))
            else:
                values.append(torch.stack([gaussian(q, s) for s in seeds]).mean(0))
        acquisition[arm] = time.monotonic() - began + selection_seconds
        labels[arm] = torch.stack(values)
        if derivatives:
            responses = torch.stack(derivatives)
    if not torch.equal(labels['values8'], labels['responses8']):
        raise ValueError('Matched eight-shot observations differ')
    scale = labels['values8'].std(0).clamp_min(.1)
    response_scale = responses.square().mean().sqrt().clamp_min(.1)
    root = output(c) / 'technical/toy' / str(seed); root.mkdir(parents=True, exist_ok=True)
    torch.save(dict(train=x[:128], selection=x[128:], evaluation=evaluation, labels=labels,
        responses=responses, selection_labels=selection, scale=scale,
        response_scale=response_scale, acquisition_seconds=acquisition), root / 'data.pt')
    def nll(pred, target):
        return (.5 * ((pred - target) / scale).square() + scale.log() + .5 * math.log(2 * math.pi)).mean()
    rows = []
    arms = ['values8', 'values32', 'responses8']
    arms = arms[index % 3:] + arms[:index % 3]
    for arm in arms:
        location = root / arm; location.mkdir(exist_ok=True)
        if (location / 'complete.json').exists():
            rows.extend(read(location / 'complete.json')['rows']); continue
        torch.manual_seed(seed)
        model = SmallAtlas(2, 2, latent_dim=4, width=32).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
        best = math.inf; best_epoch = 0; snapshots = {}; epoch = 0
        time_marks = list(c['toy_cost_seconds']); epoch_marks = list(c['toy_epoch_checkpoints'])
        if acquisition[arm] >= min(time_marks):
            raise ValueError(f'Acquisition alone exceeds first declared total budget: {arm}, {acquisition[arm]}')
        with online_training(DEFAULTS, run_id=f'resp-followup-{seed}-{arm}',
                name=f'response-followup-{arm}-{seed}', config=dict(c, arm=arm, fit_seed=seed,
                batch_size=128, microbatch=128, capacity='toy MLP width32 latent4; explicit exception',
                selector='shared selection feature Gaussian NLL',
                inputs=['u', 'v'], conditions_as_inputs=[], history=False),
                folder=location, receipt_path=location / 'wandb.json', job_type='predictor', group=c['protocol']) as run:
            start = time.monotonic()
            while epoch_marks or time_marks:
                model.train()
                loss = nll(model(x[:128]), labels[arm])
                if arm == 'responses8':
                    loss = loss + ((jacobian(model, x[:128], True) - responses) / response_scale).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite objective: {arm}, {seed}, {epoch}')
                optimizer.zero_grad(); loss.backward(); optimizer.step(); epoch += 1
                with torch.no_grad():
                    score = float(nll(model(x[128:]), selection))
                if score < best:
                    best = score; best_epoch = epoch; state = copy.deepcopy(model.state_dict())
                if epoch % 25 == 0:
                    run.log(dict(epoch=epoch, train_objective=float(loss.detach()), selection_nll=score))
                elapsed = time.monotonic() - start
                due = [('epochs', v) for v in epoch_marks if epoch >= v]
                due += [('total_seconds', v) for v in time_marks if elapsed + acquisition[arm] >= v]
                for kind, value in due:
                    snapshots[f'{kind}-{value}'] = dict(model=copy.deepcopy(state),
                        arm=arm, seed=seed, budget_kind=kind, budget=value, reached_epoch=epoch,
                        selected_epoch=best_epoch, selection_nll=best, training_seconds=elapsed,
                        acquisition_seconds=acquisition[arm], total_seconds=elapsed + acquisition[arm])
                    (epoch_marks if kind == 'epochs' else time_marks).remove(value)
                if due:
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                                    snapshots=snapshots), location / 'progress.pt')
            # Final points are evaluated only after all choices/checkpoints freeze.
            for key, saved in snapshots.items():
                model.load_state_dict(saved['model']); model.eval()
                pred = model(evaluation).detach(); derivative = jacobian(model, evaluation).detach()
                row = {k: v for k, v in saved.items() if k != 'model'}
                row.update(exact_value_mse=float((pred - exact(evaluation)).square().mean()),
                           exact_response_mse=float((derivative - exact_response(evaluation)).square().mean()))
                rows.append(row)
                torch.save(dict(**saved, predictions=pred, responses=derivative), location / f'{key}.pt')
            arm_rows = [r for r in rows if r['arm'] == arm]
            run.summary.update(dict(checkpoints=arm_rows, selection_shots=c['toy_selection_shots'],
                                    cost_scope='CPU acquisition + optimization/selection; online initialization and final evaluation excluded'))
            write_json(location / 'complete.json', dict(rows=arm_rows))
    export(c, f'toy-seed-{seed}', rows)


def precision(c):
    rows = []; rng = np.random.default_rng(c['seed'] + 702)
    for parent in c['parent_indices']:
        root = resolve_path(c['simulation_archive']) / f'parent-{parent:03d}'
        if sha(root / 'query.pt') != read(root / 'complete.json')['query_sha256']:
            raise ValueError(f'Query checksum changed: {root}')
        bundle = torch.load(root / 'query.pt', map_location='cpu', weights_only=False)
        if bundle['config'] != c:
            raise ValueError('Wrong precision-study protocol')
        for h, horizon in enumerate(bundle['horizons_steps']):
            full = bundle['responses'][:, h * c['rff_features']:(h + 1) * c['rff_features']]
            for budget in c['precision_budgets']:
                # Fixed nested prefixes; no data-dependent early stopping.
                response = full[:budget]
                signal = float(torch.trace(unbiased_gram(response)))
                noise = float(response.var(0, unbiased=True).sum() / budget)
                samples = []
                for _ in range(c['precision_bootstrap']):
                    sample = response[rng.integers(0, budget, budget)]
                    samples.append(float(torch.trace(unbiased_gram(sample))))
                low, high = np.quantile(samples, [.025, .975])
                rows.append(dict(parent=parent, horizon_fs=horizon * c['timestep_fs'], shots=budget,
                    mean_response_squared_corrected=signal, variance_of_mean_trace=noise,
                    bootstrap_low=float(low), bootstrap_high=float(high),
                    scope='conditional shot bootstrap at fixed development parent; unreliable near zero/low B; no acceptance test'))
    export(c, 'atomistic-precision', rows)


def collect(c):
    rows = []
    for seed in c['fit_seeds']:
        for arm in ('values8', 'values32', 'responses8'):
            rows.extend(read(output(c) / 'technical/toy' / str(seed) / arm / 'complete.json')['rows'])
    export(c, 'toy-learning', rows)


def submit(c):
    bind(c)
    for family in ('response_atlas', 'response_atlas_followup'):
        check_metric_docs(family=family)
    root = output(c) / 'technical'
    if (root / 'launch.json').exists():
        raise ValueError('Follow-up already submitted')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, root / 'code', c,
                                   directories=('src', 'docs/metrics', 'configs/simulation'))
    receipt = dict(jobs={}, code=str(bundle.root), scope='toy control strengthening and fixed-parent shot precision')
    queue = SlurmQueue(root, bundle, 'src.research.response_atlas.followup',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             WANDB_MODE='online', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'), root / 'launch.json', receipt, 'RESP2')
    with queue.submission():
        toy_job = queue.submit('toy', [f'--array=0-{len(c["fit_seeds"])-1}%3', '--cpus-per-task=2', '--mem=8G', '--time=01:00:00'])
        queue.submit('collect', ['--cpus-per-task=2', '--mem=8G', '--time=00:15:00'], 'afterok:' + toy_job)
        queue.submit('gpu', ['--gpus=1', '--cpus-per-task=4', '--mem=48G', '--time=12:00:00', '--signal=B:TERM@180'],
                     partition=c['gpu_partition'])
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('submit', 'toy', 'collect', 'gpu', 'precision'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); c = read(args.config)
    if args.stage == 'submit':
        print(submit(c)); return
    def stopping(signum, frame):
        raise RuntimeError(f'Scheduler signal {signum}: preserve partial branches')
    signal.signal(signal.SIGTERM, stopping)
    bind(c)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    with recorded_stage(output(c) / 'technical' / f'{args.stage}-{index}.json', job=os.environ.get('SLURM_JOB_ID')):
        if args.stage == 'toy':
            toy(c, index)
        elif args.stage == 'collect':
            collect(c)
        elif args.stage == 'precision':
            precision(c)
        else:
            from . import atomistic
            atomistic.gate(c)
            atomistic.run(c)
            precision(c)


if __name__ == '__main__':
    main()
