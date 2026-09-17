"""Frozen native onset states with fresh linear and nonlinear hazard readouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.predictive_memory.prepare import file_hash, write_json
from src.project_runtime.paths import load_json, resolve_path
from .baselines import hazard_fit, check_deadline
from .native_queue import configure_file_limit
from .onset_model import OnsetModel
from .supervised import prepare_rows
from .native_runtime import evaluate
from src.models.encoders.mace_backend import with_mace_backend, mace_backend_metadata


def run(config):
    limits = configure_file_limit()
    torch.set_num_threads(config['torch_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = resolve_path(config['output']) / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    status = root / 'readout_status.json'
    if status.exists():
        raise FileExistsError(f'Use a fresh readout output: {root}')
    completed = []
    write_json(status, dict(state='preparing', completed=completed, file_limit=limits))
    try:
        check_deadline(config)
        windows, labels, cond, events, _, _, identity = prepare_rows(config)
        plan = json.loads(resolve_path(config['plan']).read_text())
        indices = np.flatnonzero(labels['risk'])
        arrays = dict(y=labels['event_bin'][indices], source=labels['source_id'][indices],
                      center=labels['center_id'][indices], anchor=labels['anchor'][indices],
                      split=labels['split'][indices],
                      temperature=np.array([windows.rows[i]['temperature_K'] for i in indices]))
        native_root = resolve_path(config['native_output']) / 'technical'
        for variant in config['variants']:
            check_deadline(config)
            checkpoint = native_root / variant / 'best.pt'
            complete = json.loads((native_root / variant / 'complete.json').read_text())
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if complete['identity'] != identity or state['identity'] != identity:
                raise ValueError(f'Frozen native readout identity differs: {variant}')
            destination = root / variant
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / 'provenance.json', dict(identity=identity,
                       checkpoint_sha256=file_hash(checkpoint),
                       native_readouts_sha256=file_hash(Path(__file__)),
                       native_runtime_sha256=file_hash(Path(__file__).with_name('native_runtime.py')),
                       mace_backend=mace_backend_metadata(config['mace_backend']),
                       backend_implementation={name: file_hash(Path(__file__).parents[2]/'models/encoders'/name)
                                               for name in ['mace_backend.py', 'mace_causal.py']},
                       target='First sustained onset; frozen primary at-risk native rows',
                       features='Frozen z128 plus seven known conditions; no history or future added',
                       selection='Full selection fold, train-only input normalization; same seed and descriptor settings'))
            model = OnsetModel(variant, activation_checkpoint=False, max_spatial_edges=config['max_spatial_edges']).cuda()
            model.load_state_dict(state['model'])
            # Verify each trained checkpoint on actual retained observations
            # before exporting frozen states. No optimizer or checkpoint is altered.
            verify_indices = indices[:8]
            verify_observations = [windows.observation(int(i), variant) for i in verify_indices]
            with torch.no_grad():
                reference = model(verify_observations, cond[verify_indices])
                model.encoder = with_mace_backend(model.encoder, config['mace_backend'])
                converted = model(verify_observations, cond[verify_indices])
            errors = {}
            for key in ('state', 'logits'):
                torch.testing.assert_close(converted[key], reference[key], atol=3e-5, rtol=2e-4)
                errors[key+'_max_abs_error'] = float((converted[key]-reference[key]).abs().max())
            write_json(destination / 'backend_validation.json', dict(indices=verify_indices.tolist(), **errors))
            del reference, converted, verify_observations
            logits = []
            states = []
            for start in range(0, len(indices), 512):
                check_deadline(config)
                batch = indices[start:start+512]
                _, block_logits, block_states = evaluate(model, windows, batch, cond, events, 8)
                logits.append(block_logits); states.append(block_states)
                done = start+len(batch)
                print(f'{variant}: exported {done}/{len(indices)} frozen states', flush=True)
                write_json(status, dict(state='extracting', variant=variant, rows=done, total_rows=len(indices), completed=completed))
            z = torch.cat(states).numpy()
            np.savez(destination / 'frozen_states.npz', embeddings=z, native_logits=torch.cat(logits).numpy(),
                     indices=indices, **arrays)
            del model, states, logits
            torch.cuda.empty_cache()
            arrays['x'] = np.concatenate((z, cond[indices].cpu().numpy()), axis=-1)
            for kind in ['linear', 'mlp']:
                check_deadline(config)
                name = f'{variant}/{kind}'
                print(f'Fitting frozen-state readout {name}', flush=True)
                write_json(status, dict(state='fitting', current=name, completed=completed))
                hazard_fit(arrays, kind, plan, torch.device('cuda'), config, destination / kind)
                completed.append(name)
            # Keep metadata common; do not accidentally save another variant's features.
            del arrays['x']
            torch.cuda.empty_cache()
        write_json(status, dict(state='complete', completed=completed, analysis='deferred until requested'))
    except Exception as exc:
        write_json(status, dict(state='failed', completed=completed, error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(load_json(args.config))


if __name__ == '__main__':
    main()
