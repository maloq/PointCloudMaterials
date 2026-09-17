"""Measured native workload and the required train-only 32-window fitting gate."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from src.data.predictive_memory.targets import BLOCKS
from src.data.predictive_memory.prepare import write_json
from src.project_runtime.paths import load_json, resolve_path
from .audit_sources import sha256
from .native_data import NativeWindows, conditions, small_set_indices, SourceSampler, HORIZON_FRAMES
from .native_model import PhysicalMeans


def seed_all():
    random.seed(20260919); np.random.seed(20260919); torch.manual_seed(20260919)
    torch.cuda.manual_seed_all(20260919)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_checkpoint(path, model, optimizer, sampler, step, identity, extra):
    temporary = path.with_suffix('.building.pt')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        sampler=sampler.state_dict(), step=step, identity=identity, extra=extra,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate()), temporary)
    temporary.replace(path)


def check_deadline(config, reserve=120):
    if time.time() + reserve >= datetime.fromisoformat(config['training_deadline_utc']).timestamp():
        raise TimeoutError('Training cutoff reached; preserve/export reserve begins')


def fit_update(model, windows, indices, cond, present, future, optimizer, *, microbatch, future_weight=1.):
    optimizer.zero_grad(set_to_none=True)
    aggregate = 0.
    variant = model.encoder.variant
    for start in range(0, len(indices), microbatch):
        selected = indices[start:start + microbatch]
        observations = [windows.observation(i, variant) for i in selected]
        result = model(observations, cond[selected])
        loss = (result['present'] - present[selected]).square().mean()
        if future_weight:
            loss = loss + future_weight * (result['future'] - future[selected]).square().mean()
        scaled = loss * len(selected) / len(indices)
        scaled.backward()
        aggregate += float(scaled.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return aggregate, float(norm)


@torch.no_grad()
def evaluate_present(model, windows, indices, cond, present, microbatch):
    model.eval()
    errors = []
    for start in range(0, len(indices), microbatch):
        selected = indices[start:start + microbatch]
        result = model([windows.observation(i, 'snapshot') for i in selected], cond[selected])
        errors.append((result['present'] - present[selected]).square())
    errors = torch.cat(errors)
    metrics = dict(mse=float(errors.mean()), blocks={name: float(errors[:, a:b].mean()) for name, (a, b) in BLOCKS.items()})
    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['profile', 'smallfit'], required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); config = load_json(args.config)
    torch.set_num_threads(config.get('torch_threads', 4)); seed_all()
    root = resolve_path(config['output']) / 'technical'; root.mkdir(parents=True, exist_ok=True)
    windows = NativeWindows(root / 'cohort.json', root / 'source_audit.json',
        cpu_cache_gib=config['cpu_cache_gib'], gpu_cache_gib=config['gpu_cache_gib'])
    indices = small_set_indices(windows.rows)
    selected_rows = [windows.rows[i] for i in indices]
    identity = dict(cohort_sha256=sha256(root / 'cohort.json'), source_audit_sha256=sha256(root / 'source_audit.json'),
                    row_ids=[row['row_id'] for row in selected_rows], seed=20260919,
                    protocol='local_crystallization_predictability_v1',
                    scaling_population='32 diagnostic training windows, current and six futures; not a core release')
    prepared = root / 'smallfit_inputs.npz'
    identity_path = root / 'smallfit_inputs.json'
    if prepared.exists():
        if json.loads(identity_path.read_text())['identity'] != identity:
            raise ValueError('Existing preflight inputs do not match frozen identities')
        if sha256(prepared) != json.loads(identity_path.read_text())['sha256']:
            raise ValueError('Preflight target artifact changed')
        values = np.load(prepared)['targets']
    else:
        values = np.array([[windows.packet(i, lag) for lag in np.r_[0, HORIZON_FRAMES]] for i in indices])
        np.savez(prepared, targets=values, indices=indices)
        write_json(identity_path, dict(identity=identity, sha256=sha256(prepared)))
    mean = values.astype(np.float64).reshape(-1, 128).mean(0)
    scale = np.maximum(values.astype(np.float64).reshape(-1, 128).std(0), 1e-4)
    identity['normalizer'] = dict(mean=mean.tolist(), scale=scale.tolist())
    identity['implementation_sha256'] = {name: sha256(Path(__file__).parent / name)
                                        for name in ['native_model.py', 'native_data.py', 'native_preflight.py']}
    # Index targets on the complete frozen row index; only training rows are filled.
    present = torch.zeros(len(windows.rows), 128, device='cuda')
    future = torch.zeros(len(windows.rows), 6, 128, device='cuda')
    standardized = torch.tensor((values - mean) / scale, device='cuda', dtype=torch.float32)
    present[indices] = standardized[:, 0]; future[indices] = standardized[:, 1:]
    all_conditions, condition_stats = conditions(windows.rows)
    identity['conditions'] = condition_stats
    cond = torch.tensor(all_conditions, device='cuda')
    write_json(root / 'smallfit_definition.json', identity)
    sampler = SourceSampler(windows.rows, indices)
    if args.stage == 'profile':
        profiles = {}
        for variant in ['snapshot', 'history12', 'repeat12']:
            check_deadline(config)
            seed_all()
            model = PhysicalMeans(variant=variant, activation_checkpoint=config.get('activation_checkpoint', True),
                                  max_spatial_edges=config['max_spatial_edges']).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
            microbatch = config['microbatch'][variant]
            batch = indices[::4]  # One window per source, all eight sources.
            cold = time.monotonic()
            for i in batch:
                windows.observation(i, variant)
            synchronize(); preparation_seconds = time.monotonic() - cold
            torch.cuda.reset_peak_memory_stats()
            timings, losses = [], []
            for update in range(config.get('profile_updates', 4)):
                synchronize(); started = time.monotonic()
                loss, norm = fit_update(model, windows, batch, cond, present, future, optimizer, microbatch=microbatch)
                synchronize(); elapsed = time.monotonic() - started
                timings.append(elapsed); losses.append(loss)
                print(json.dumps(dict(stage='profile', variant=variant, update=update, seconds=elapsed, loss=loss,
                                      peak_vram_gib=torch.cuda.max_memory_allocated() / 1024**3)), flush=True)
            model.eval(); synchronize(); started = time.monotonic()
            with torch.no_grad():
                for start in range(0, len(batch), microbatch):
                    selected = batch[start:start + microbatch]
                    model([windows.observation(i, variant) for i in selected], cond[selected])
            synchronize(); evaluation_seconds = time.monotonic() - started
            profiles[variant] = dict(microbatch=microbatch, effective_batch=8, update_seconds=timings,
                p90_update_seconds=float(np.percentile(timings[1:], 90)),
                validation_seconds_per_window=evaluation_seconds / len(batch),
                cold_preparation_seconds_per_window=preparation_seconds / len(batch),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                losses=losses, cache=windows.statistics())
            write_json(root / 'native_profile.json', dict(identity=identity, profiles=profiles,
                       total_vram_bytes=torch.cuda.get_device_properties(0).total_memory))
            del model, optimizer; torch.cuda.empty_cache()
        return
    model = PhysicalMeans(variant='snapshot', activation_checkpoint=config.get('activation_checkpoint', True),
                          max_spatial_edges=config['max_spatial_edges']).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    microbatch = config['microbatch']['snapshot']
    step = 0; trace = []; best = float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Explicit diagnostic resume identity differs')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler']); step = saved['step']; trace = saved['extra']['trace']; best = saved['extra']['best']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    success = False
    torch.cuda.reset_peak_memory_stats()
    while step < 2000:
        check_deadline(config)
        synchronize(); started = time.monotonic()
        loss, norm = fit_update(model, windows, sampler.batch(), cond, present, future, optimizer,
                               microbatch=microbatch, future_weight=0.)
        step += 1
        if step % 25 == 0 or step == 1:
            metrics = evaluate_present(model, windows, indices, cond, present, microbatch)
            synchronize()
            record = dict(step=step, loss=loss, gradient_norm=norm, **metrics,
                          seconds_last_update_and_evaluation=time.monotonic() - started)
            trace.append(record); print(json.dumps(record), flush=True)
            improved = metrics['mse'] < best
            best = min(best, metrics['mse'])
            extra = dict(trace=trace, best=best)
            save_checkpoint(root / 'smallfit_latest.pt', model, optimizer, sampler, step, identity, extra)
            if improved:
                save_checkpoint(root / 'smallfit_best.pt', model, optimizer, sampler, step, identity, extra)
            success = metrics['mse'] <= .1 and max(metrics['blocks'].values()) <= .25
            write_json(root / 'smallfit_status.json', dict(state='passed' if success else 'running', step=step,
                metrics=metrics, trace=trace, identity=identity, cache=windows.statistics(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            if success:
                break
    if not success:
        status = json.loads((root / 'smallfit_status.json').read_text()); status['state'] = 'failed_fit_gate'
        write_json(root / 'smallfit_status.json', status)
        raise RuntimeError('Small-set fit failed its declared gates; full native group is blocked')


if __name__ == '__main__':
    main()
