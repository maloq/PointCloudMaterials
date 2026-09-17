"""Packet observability fits; future-input diagnostics are never forecasts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from src.data.predictive_memory.prepare import file_hash, write_json
from src.project_runtime.paths import load_json, resolve_path
from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows
from .baselines import check_deadline, hazard_fit


def source_arrays(data, source, anchors, case, plan):
    """Use all states for state recognition, and the frozen risk set for onset."""
    cadence = plan['sampling']['cadence_ps']
    lag = round(case['horizon_ps'] / cadence)
    if not np.isclose(lag * cadence, case['horizon_ps']):
        raise ValueError(f'Horizon is off the recorded time grid: {case}')
    crystal = np.isin(data['labels'], plan['assay']['crystal_structure_types'])
    packet = data['packet']
    if packet.shape != (*crystal.shape, 128):
        raise ValueError(f"Source {source['id']}: incompatible packet/label shapes")
    shape = (len(packet), len(anchors))
    task = case['task']
    if task == 'state':
        offsets = np.array([lag])
        keep = np.ones(shape, bool)
        target = crystal[:, anchors + lag]
    elif task == 'onset_sequence':
        if lag <= 0:
            raise ValueError('Onset requires a positive forecast horizon')
        persistence = plan['assay']['persistence_frames']
        # An onset at the endpoint needs persistence-1 subsequent observations.
        offsets = np.arange(1, lag + persistence)
        onset = first_sustained_onset(crystal, persistence)
        keep = risk_windows(crystal, onset, anchors, plan['assay']['negative_history_frames'])
        delay = onset[:, None] - anchors
        target = (delay > 0) & (delay <= lag)
    else:
        raise ValueError(f'Unknown observability task: {task}')
    frames = anchors[:, None] + offsets
    if frames.min() < 0 or frames.max() >= packet.shape[1]:
        raise ValueError(f"Source {source['id']}: incomplete observation/confirmation for {case}")
    if not np.allclose(data['times_ps'][frames], frames * cadence, rtol=0, atol=1e-6):
        raise ValueError(f"Source {source['id']}: timeline differs from the frozen cadence")
    condition = np.zeros((*shape, 7), np.float32)
    condition[:, :, plan['sampling']['temperatures_K'].index(int(source['temperature_K']))] = 1
    condition[:, :, 5] = anchors[None] * cadence / plan['sampling']['end_ps']
    condition[:, :, 6] = condition[:, :, 5] ** 2
    observed = packet[:, frames].reshape(*shape, -1)
    features = np.concatenate((observed, condition), axis=-1)
    if not np.isfinite(features).all():
        raise ValueError(f"Source {source['id']}: nonfinite observability features")
    arrays = dict(
        x=features, y=np.where(target, 0, 1),
        source=np.full(shape, source['id']),
        split=np.full(shape, source.get('validation_role', source['split'])),
        temperature=np.full(shape, source['temperature_K']),
        center=np.broadcast_to(data['atom_ids'][:, None], shape),
        anchor=np.broadcast_to(anchors, shape),
    )
    return {key: value[keep] for key, value in arrays.items()}


def build_arrays(release, cache, case, plan):
    anchors = np.array(release['native_anchors'])
    parts = []
    for source in release['sources']:
        with np.load(cache / source['shard']) as data:
            parts.append(source_arrays(data, source, anchors, case, plan))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def run(config):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    plan = json.loads(resolve_path(config['plan']).read_text())
    if config['seed'] != plan['sampling']['selection_seed']:
        raise ValueError('This queue uses only the predeclared training seed')
    cache = resolve_path(config['cache'])
    output = resolve_path(config['output'])
    root = output / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    status = root / 'observability_status.json'
    # A new output is required: never silently reuse weights from another task.
    if status.exists():
        raise FileExistsError(f'Observability output already used: {output}')
    completed = []
    current = None
    write_json(status, dict(state='verifying', completed=completed))
    try:
        check_deadline(config)
        release = json.loads((cache / 'release.json').read_text())
        if release['state'] != 'complete':
            raise ValueError('A complete frozen source release is required')
        if release['plan_sha256'] != file_hash(resolve_path(config['plan'])):
            raise ValueError('Plan differs from the frozen release')
        if release['cadence_ps'] != plan['sampling']['cadence_ps']:
            raise ValueError('Release cadence differs from the plan')
        for source in release['sources']:
            if file_hash(cache / source['shard']) != source['shard_sha256']:
                raise ValueError(f"Changed source shard {source['id']}")
        implementation = [Path(__file__), Path(__file__).with_name('baselines.py'),
                          Path(__file__).with_name('metrics.py'),
                          Path('src/research/forecast_crystallization/local_metrics.py')]
        write_json(root / 'protocol.json', dict(
            config=config, release_sha256=file_hash(cache / 'release.json'),
            implementations={str(path): file_hash(path) for path in implementation},
            binary_encoding='event_bin=0 is positive; event_bin=1 is negative; one-bin logistic likelihood',
            population='Frozen native anchors/centers; all states for state tasks; primary risk set for onset',
            future_inputs='Diagnostic oracles only; dense sequence includes two confirmation frames',
            selection='Source-weighted selection-fold binary NLL; train-only feature normalization',
            evaluation='Save predictions for later analysis; no test-based selection or scientific interpretation',
        ))
        for case in config['cases']:
            check_deadline(config)
            print(f"Preparing {case['name']}", flush=True)
            arrays = build_arrays(release, cache, case, plan)
            binary_plan = deepcopy(plan)
            binary_plan['sampling']['horizons_ps'] = [case['horizon_ps']]
            destination = root / case['name']
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / 'task.json', dict(
                **case, rows=len(arrays['x']), features=arrays['x'].shape[1],
                row_counts={str(s): int(np.sum(arrays['split'] == s)) for s in np.unique(arrays['split'])},
            ))
            for kind in config['model_kinds']:
                current = f"{case['name']}/{kind}"
                write_json(status, dict(state='running', completed=completed, current=current))
                print(f"Starting {current}: {len(arrays['x'])} rows, {arrays['x'].shape[1]} features", flush=True)
                probability, _, _ = hazard_fit(arrays, kind, binary_plan, device, config, destination / kind)
                if probability.shape != (len(arrays['x']), 1) or not np.isfinite(probability).all():
                    raise RuntimeError(f'Invalid binary predictions: {current}')
                completed.append(current)
                write_json(status, dict(state='running', completed=completed, current=current))
                print(f'Checkpoint and predictions saved: {current}', flush=True)
            del arrays
            torch.cuda.empty_cache()
        write_json(status, dict(state='complete', completed=completed, analysis='deferred until requested'))
    except Exception as exc:
        write_json(status, dict(state='failed', completed=completed, current=current, error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    run(load_json(args.config))


if __name__ == '__main__':
    main()
