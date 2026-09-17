"""Matched H100 backbone comparisons and GATr repeats of completed onset fits.

This driver reuses the v2 trainer. It lives outside that trainer's frozen source
set so adding a downstream comparison cannot invalidate running v2 experiments.
"""
import argparse
from contextlib import ExitStack
import csv
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from src.data.predictive_memory.prepare import file_hash, write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import load_json, resolve_path
from src.research.local_predictability.backbone_data import prepare_data
from src.research.local_predictability.backbone_v2 import fit, export, identity, gate_path, verify_gate
from src.research.local_predictability.native_data import SourceSampler
from src.research.local_predictability.native_queue import configure_file_limit


def table(root, name, rows):
    snapshot_metric_docs(root, 'backbone_comparison')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path = root/'tables'/f'{name}.csv'
    temporary = path.with_suffix('.building.csv')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def compare_profiles(mace, gatr):
    for key in ('gpu', 'variant', 'precision', 'effective_batch', 'microbatch', 'rows', 'observation_shapes'):
        if mace[key] != gatr[key]:
            raise ValueError(f'Unmatched speed comparison: {key}')
    for key in ('data', 'config', 'implementation'):
        if mace['identity'][key] != gatr['identity'][key]:
            raise ValueError(f'Unmatched speed identity: {key}')
    row = dict(variant=mace['variant'], gpu=mace['gpu'], precision=mace['precision'],
        effective_batch=mace['effective_batch'], microbatch=mace['microbatch'])
    for name, value in [('mace', mace), ('gatr', gatr)]:
        durations = np.asarray(value['update_seconds'])
        if len(durations) < 4 or not np.isfinite(durations).all() or not (durations > 0).all():
            raise ValueError('Need at least four finite positive synchronized profile timings')
        row.update({f'{name}_update_seconds': float(durations.mean()),
            f'{name}_windows_per_second': value['effective_batch']/float(durations.mean()),
            f'{name}_validation_seconds': value['validation_seconds'],
            f'{name}_cold_preparation_seconds': value['cold_preparation_seconds'],
            f'{name}_peak_allocated_gib': value['peak_allocated_bytes']/1024**3})
    row['gatr_training_speedup'] = row['mace_update_seconds']/row['gatr_update_seconds']
    row['gatr_validation_speedup'] = row['mace_validation_seconds']/row['gatr_validation_seconds']
    return row


def assert_paired(reference, candidate):
    for key in ('indices', 'source', 'center', 'anchor'):
        np.testing.assert_array_equal(reference[key], candidate[key], err_msg=f'Unmatched prediction rows: {key}')


def compare_screen(plan):
    root = resolve_path(plan['output']); screen = resolve_path(plan['screen_output'])/'technical'
    rows = []
    for variant in ('snapshot', 'history12'):
        profiles = [load_json(screen/kind/'profile'/f'{variant}.json') for kind in ('mace', 'axial_gatr')]
        rows.append(compare_profiles(*profiles))
    table(root, 'h100_speed', rows)
    completed = load_json(screen/'screen_status.json')
    if completed['state'] != 'complete' or completed['matched_updates'] != 2048:
        raise ValueError('Physical comparison requires both complete matched 2048-update fits')
    prediction_roots = [screen/kind/'physical_means/snapshot' for kind in ('mace', 'axial_gatr')]
    metrics = []
    for split in ('selection', 'calibration', 'test'):
        arrays, receipts = [], []
        for directory in prediction_roots:
            receipt = load_json(directory/f'{split}_export.json')
            path = directory/f'{split}_predictions.npz'
            if file_hash(path) != receipt['sha256'] or file_hash(directory/'best.pt') != receipt['checkpoint_sha256']:
                raise ValueError(f'Changed physical prediction export: {path}')
            with np.load(path) as data:
                arrays.append(dict(data))
            receipts.append(receipt)
        assert_paired(*arrays)
        np.testing.assert_array_equal(arrays[0]['targets'], arrays[1]['targets'])
        for kind, receipt in zip(('mace', 'axial_gatr'), receipts, strict=True):
            values = receipt['metrics']
            metrics.append(dict(encoder=kind, split=split, rows=receipt['rows'], selected_step=receipt['selected_step'],
                present_mse=values['present_mse'], future_mse=values['future_mse'],
                **{f'future_{lag}_ps_mse': error for lag, error in values['future_by_horizon'].items()}))
    table(root, 'physical_snapshot', metrics)
    write_json(root/'technical/screen_comparison.json', dict(speed=rows, physical=metrics,
        inference='Single-seed point estimates; profiles use repeated resident batches and exclude cold preparation'))


def validate_reference(config, data, reference_root):
    """Bind the repeats to the actually completed MACE populations and budget."""
    sampler = SourceSampler(data.windows.rows, data.splits('onset')['train'])
    for _ in range(config['training_updates']):
        sampler.batch()
    files = {}
    for stage in ('parent', 'snapshot', 'history12', 'repeat12'):
        directory = reference_root/'technical'/stage
        complete = load_json(directory/'complete.json')
        previous = complete['identity']
        if complete['state'] != 'complete' or complete['steps'] != config['training_updates']:
            raise ValueError(f'Incomplete or differently budgeted reference: {directory}')
        for key in ('cohort_sha256', 'release_sha256', 'labels_sha256', 'conditions_sha256'):
            if previous[key] != data.identity[key]:
                raise ValueError(f'MACE/GATr population mismatch: {stage}/{key}')
        if previous['seed'] != config['seed'] or previous['selection_indices'] != data.selection('onset'):
            raise ValueError(f'MACE/GATr seed or selection mismatch: {stage}')
        checkpoint = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
        if checkpoint['step'] != config['training_updates'] or checkpoint['identity'] != previous:
            raise ValueError(f'MACE reference checkpoint does not verify its completion: {stage}')
        if checkpoint['sampler'] != sampler.state_dict():
            raise ValueError(f'MACE/GATr training draw sequence differs: {stage}')
        files[str(directory/'latest.pt')] = file_hash(directory/'latest.pt')
        if stage != 'parent':
            files[str(directory/'predictions.npz')] = file_hash(directory/'predictions.npz')
            files[str(directory/'result.json')] = file_hash(directory/'result.json')
    return files


def copy_verified_gate(parent, child, data):
    """Relocate the SAME gate evidence for a child differing only in output path."""
    expected = identity(child, data, 'axial_gatr', 'physical_means', 'snapshot')
    if identity(parent, data, 'axial_gatr', 'physical_means', 'snapshot') != expected:
        raise ValueError('Parent and child must have identical scientific configuration')
    verify_gate(parent, data, 'axial_gatr')
    source = gate_path(parent, 'axial_gatr'); target = gate_path(child, 'axial_gatr')
    receipt = load_json(source)
    if file_hash(source.parent/'best.pt') != receipt['checkpoint_sha256']:
        raise ValueError('Passing parent gate checkpoint changed')
    target.parent.mkdir(parents=True, exist_ok=True)
    for name in ('receipt.json', 'best.pt'):
        original, destination = source.parent/name, target.parent/name
        if destination.exists():
            if file_hash(destination) != file_hash(original):
                raise ValueError(f'Existing child gate differs: {destination}')
        else:
            temporary = destination.with_suffix('.copying')
            shutil.copyfile(original, temporary); temporary.replace(destination)
    verify_gate(child, data, 'axial_gatr')
    write_json(target.parent/'reuse.json', dict(source=str(source), sha256=file_hash(source),
        reason='Identical data/code/configuration; only the output directory differs. No weights or receipt rewritten.'))


def fit_or_resume(config, data, variant, *, parent=None, resume=False):
    directory = resolve_path(config['output'])/'technical/axial_gatr/onset'/variant
    status = directory/'status.json'
    if status.exists() and load_json(status)['state'] == 'complete':
        if not resume:
            raise FileExistsError(f'Repeats require explicit --resume: {directory}')
        saved = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
        expected = identity(config, data, 'axial_gatr', 'onset', variant)
        expected['gate_receipt_sha256'] = verify_gate(config, data, 'axial_gatr')
        if saved['identity'] != expected or saved['step'] != config['training_updates']:
            raise ValueError('Completed continuation identity changed')
        return
    fit(config, data, 'axial_gatr', 'onset', variant, parent=parent, resume=resume)


def compare_onset(plan, variant):
    root = resolve_path(plan['output'])
    reference = resolve_path(plan['mace_reference'])/'technical'/variant
    controls = load_json(resolve_path(plan['controls_config']))
    candidate = resolve_path(controls['output'])/'technical/axial_gatr/onset'/variant
    with np.load(reference/'predictions.npz') as original, np.load(candidate/'test_predictions.npz') as repeated:
        test = original['split'] == 'test'
        assert_paired({key: original[key][test] for key in ('indices', 'source', 'center', 'anchor')}, repeated)
        np.testing.assert_array_equal(original['event_bin'][test], repeated['event_bin'])
    rows = []
    for kind, path in [('mace_v1', reference/'result.json'), ('axial_gatr_v2', candidate/'onset_assay.json')]:
        result = load_json(path)
        for row in result['population']:
            rows.append(dict(encoder=kind, variant=variant, seed=20260919, parent_updates=4096,
                continuation_updates=4096, joint_event_nll=result['joint_event_nll'][row['split']], **row))
    table(root, f'onset_{variant}', rows)


def run(plan, resume=False):
    root = resolve_path(plan['output']); technical = root/'technical'; technical.mkdir(parents=True, exist_ok=True)
    parent = load_json(resolve_path(plan['parent_config'])); child = load_json(resolve_path(plan['controls_config']))
    configure_file_limit(); torch.set_num_threads(parent['torch_threads'])
    identity_path = technical/'repeat_identity.json'
    expected = dict(plan=plan, implementation_sha256=file_hash(Path(__file__)),
        configs={key: file_hash(resolve_path(plan[key])) for key in ('parent_config', 'controls_config')})
    if identity_path.exists():
        if not resume or load_json(identity_path) != expected:
            raise ValueError('Repeat driver needs explicit resume with unchanged plan and implementation')
    with ExitStack() as stack:
        for destination in (root, resolve_path(parent['output']), resolve_path(child['output'])):
            (destination/'technical').mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((destination/'technical/worker.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(identity_path, expected)
        compare_screen(plan)
        data = prepare_data(parent, resolve_path(parent['output'])/'technical')
        references = validate_reference(parent, data, resolve_path(plan['mace_reference']))
        reference_receipt = technical/'matched_mace_reference.json'
        if reference_receipt.exists() and load_json(reference_receipt) != references:
            raise ValueError('Previously matched MACE reference artifacts changed')
        write_json(reference_receipt, references)
        receipt = gate_path(parent, 'axial_gatr')
        write_json(technical/'repeat_status.json', dict(state='running', current='gate'))
        if receipt.exists():
            verify_gate(parent, data, 'axial_gatr')
        elif fit(parent, data, 'axial_gatr', 'physical_means', 'snapshot', diagnostic=True, resume=resume) != 'passed':
            raise RuntimeError('GATr repeat fitting gate failed; onset repeats are blocked')
        write_json(technical/'repeat_status.json', dict(state='running', current='onset_parent'))
        fit_or_resume(parent, data, 'snapshot', resume=resume)
        copy_verified_gate(parent, child, data)
        checkpoint = resolve_path(parent['output'])/'technical/axial_gatr/onset/snapshot/best.pt'
        completed = []
        for variant in ('snapshot', 'history12', 'repeat12'):
            write_json(technical/'repeat_status.json', dict(state='running', current=variant, completed=completed))
            fit_or_resume(child, data, variant, parent=checkpoint, resume=resume)
            export(child, data, 'axial_gatr', 'onset', variant)
            compare_onset(plan, variant)
            completed.append(variant)
        write_json(technical/'repeat_status.json', dict(state='complete', completed=completed,
            interpretation='One seed; physical snapshot screen and matched onset controls. No additional variants selected from test outcomes.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('run', 'compare-screen'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); plan = load_json(args.config)
    if args.stage == 'run':
        run(plan, args.resume)
    else:
        compare_screen(plan)


if __name__ == '__main__':
    main()
