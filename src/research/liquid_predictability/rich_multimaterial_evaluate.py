"""Evaluate a stopped rich-descriptor fit using its frozen scientific producer."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


MODULE = 'src.research.liquid_predictability.rich_multimaterial_evaluate'
RELATIVE_FILE = Path('src/research/liquid_predictability/rich_multimaterial_evaluate.py')


def evaluate(root, batch, chunk):
    import numpy as np
    import torch
    from src.data.fixed_cohort.protocol import digest, sha, write_json
    from src.experiment_runner.metric_docs import check_metric_docs
    from src.project_runtime.paths import resolve_path
    from src.research.liquid_predictability.rich_multimaterial_data import CachedPatches
    from src.research.liquid_predictability.rich_multimaterial_train import RichPatchMACE, export
    from src.research.supervised_onset.tracking import update_training_summary

    tech = root / 'technical'
    c = json.loads((tech / 'code/config.json').read_text())
    original = json.loads((tech / 'identity.json').read_text())
    identity = digest(original)
    plan = json.loads((tech / 'batch-plan.json').read_text())
    if digest(c) != plan['config_sha256'] or c != original['config']:
        raise ValueError('Evaluation recipe differs from the frozen training recipe')
    implementation = check_metric_docs(family=c['metric_family'])[c['metric_family']]['files']
    binding = dict(config=c, dataset=plan['dataset_identity'],
                   batch_plan_sha256=sha(tech / 'batch-plan.json'), implementation=implementation)
    if binding != original:
        amendment = json.loads((tech / 'protocol-amendment.json').read_text())
        if amendment['base_identity'] != identity or amendment['binding'] != binding:
            raise ValueError('Frozen implementation has no matching recorded amendment')
    state_path = tech / 'detached-evaluation.json'
    record = dict(identity=identity, state='starting', batch_size=batch, patch_chunk=chunk,
                  scientific_config_changed=False, created_online_runs=0,
                  checkpoint_selector='minimum recorded validation descriptor NLL',
                  job=os.environ.get('SLURM_JOB_ID'), started_at=time.time(),
                  driver_sha256=sha(Path(__file__)))
    data = None
    with (tech / 'fit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (tech / 'worker-lease.json').exists():
                raise RuntimeError('Training lease remains; inspect it before evaluation')
            write_json(state_path, record)
            print(json.dumps(record), flush=True)
            selected = min((json.loads(line) for line in (tech / 'validation.jsonl').read_text().splitlines()),
                           key=lambda row: row['validation/descriptor_nll'])
            best = torch.load(tech / 'best.pt', map_location='cpu', weights_only=False)
            if best['identity'] != identity or best['epoch'] != selected['epoch']:
                raise ValueError('Selected checkpoint does not match the recorded validation selector')
            record.update(selected_epoch=best['epoch'], selected_update=best['update'],
                          checkpoint_sha256=sha(tech / 'best.pt'),
                          validation_nll=selected['validation/descriptor_nll'])
            if sha(tech / 'training-pool-row-ids.npy') != plan['subset_sha256']:
                raise ValueError('Fitting rows changed')
            if sha(tech / 'target-standardization.npz') != plan['transform_sha256']:
                raise ValueError('Fitting target transform changed')
            torch.set_num_threads(1)
            torch.set_float32_matmul_precision('high')
            device = torch.device('cuda', 0)
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(c['batch_search']['memory_fraction'], device)
            kernel_cache = resolve_path(c['runtime']['kernel_cache'])
            kernel_cache.mkdir(parents=True, exist_ok=True)
            os.environ['CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR'] = str(kernel_cache)
            record.update(state='loading_frozen_data', gpu=torch.cuda.get_device_name(device))
            write_json(state_path, record)
            data = CachedPatches(c, 'train')
            if data.identity != plan['dataset_identity']:
                raise ValueError('Dataset identity changed')
            data.select(np.load(tech / 'training-pool-row-ids.npy'))
            with np.load(tech / 'target-standardization.npz') as transform:
                for name in ('mean', 'scale', 'active', 'loss_weight'):
                    if not np.array_equal(getattr(data, name), transform[name]):
                        raise ValueError(f'Fitting transform changed: {name}')
            execution = copy.deepcopy(c)
            execution['patch_chunk'] = chunk
            model = RichPatchMACE(execution, len(data.mean)).to(device)
            model.load_state_dict(best['model'], strict=True)
            del best
            study = SimpleNamespace(root=root, technical=tech, identity=identity, config=c)
            complete = root / 'analyses/descriptor-v1/technical/complete.json'
            if complete.exists():
                result = json.loads(complete.read_text())
                if result['identity'] != identity or result['checkpoint_sha256'] != record['checkpoint_sha256']:
                    raise ValueError('Existing evaluation belongs to a different checkpoint')
                fields = result['summary']
                record['reused_completed_export'] = True
            else:
                record['state'] = 'exporting'
                write_json(state_path, record)
                print(json.dumps(record), flush=True)
                fields = export(model, data, c, study, batch, device)
            if sha(tech / 'best.pt') != record['checkpoint_sha256']:
                raise RuntimeError('Checkpoint changed during evaluation')
            stop = json.loads((tech / 'training-stop.json').read_text())
            fields.update({'training/stopped_by_user': True,
                           'training/completed_epochs': stop['epoch'],
                           'training/stopped_update': stop['update']})
            record['state'] = 'updating_recorded_training_summary'
            write_json(state_path, record)
            update_training_summary(study, 'fit', fields, evaluation='multimaterial-local-rich-descriptors')
            record.update(state='complete', finished_at=time.time(), output=str(complete.parent.parent))
            write_json(state_path, record)
            print(json.dumps(record), flush=True)
        except BaseException:
            record.update(state='failed', traceback=traceback.format_exc(), finished_at=time.time())
            write_json(state_path, record)
            raise
        finally:
            if data is not None:
                data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--patch-chunk', type=int, default=256)
    args = parser.parse_args()
    if min(args.batch_size, args.patch_chunk) <= 0 or args.batch_size % args.patch_chunk:
        parser.error('Require positive batch/chunk sizes with batch divisible by chunk')
    root = args.run.resolve()
    code = root / 'technical/code'
    if Path(__file__).resolve().parents[3] != code:
        # Copy only this orchestration entry point. Scientific model, data and
        # metric implementations remain those captured by the original run.
        target = code / RELATIVE_FILE
        if target.exists() and target.read_bytes() != Path(__file__).read_bytes():
            raise ValueError(f'Evaluation driver changed after staging: {target}')
        shutil.copyfile(__file__, target)
        command = [sys.executable, '-u', '-m', MODULE, '--run', str(root),
                   '--batch-size', str(args.batch_size), '--patch-chunk', str(args.patch_chunk)]
        subprocess.run(command, cwd=code, env=os.environ.copy(), check=True)
        return
    evaluate(root, args.batch_size, args.patch_chunk)


if __name__ == '__main__':
    main()
