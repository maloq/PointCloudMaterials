"""Full-radius paired observed/relaxed structural audit; no new simulation."""
import json
from pathlib import Path

import numpy as np
import torch

from src.project_runtime.paths import resolve_path, dataset_path
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from src.research.bcr_pilot.data import patches_from_snapshot
from src.training_methods.bcr.data import pack
from src.training_methods.bcr.probes import descriptors, error_metrics
from .common import file_hash, write_json, remaining
from .readouts import fit_residual_probe, predict_probe, target_groups


def freeze_selection(study):
    cfg = study.config['relaxed']; path = study.technical/'relaxed-selection.json'
    if path.exists():
        selected = json.loads(path.read_text())
        if selected['identity'] != study.identity: raise ValueError('Relaxed selection identity changed')
        return selected
    plan = json.loads(resolve_path(cfg['plan']).read_text())
    cache = resolve_path(plan['config']['cache'])
    excluded = {r['root'] for r in study.records if r['split'] == 'development'}
    bcr_seen = {r['root'] for r in study.records if r['split'] == 'train'}
    rng = np.random.default_rng(study.config['seed']); sources = []; cells = []
    for temperature in (400, 450, 500, 510, 520):
        for role, count in [('train', cfg['train_roots_per_temperature']),
                            ('selection', cfg['development_roots_per_temperature'])]:
            candidates = sorted([s for s in plan['sources'] if s['temperature_K'] == temperature
                and s.get('validation_role', s['split']) == role and s['lineage'] not in excluded
                and all((cache/'cells'/f'{s["id"]}-{f}'/'complete.json').exists() for f in cfg['frames'])], key=lambda s: s['id'])
            if len(candidates) < count:
                raise ValueError(f'Need {count} fully paired {temperature}K/{role} sources; have {len(candidates)}')
            for ordinal, index in enumerate(rng.permutation(len(candidates))[:count]):
                source = candidates[index]
                split = ('tune' if ordinal == count-1 else 'fit') if role == 'train' else 'development'
                if split == 'development' and source['lineage'] in bcr_seen:
                    raise ValueError('Transfer development root was used for BCR training')
                source = dict(source, audit_split=split)
                sources.append(source)
                for frame in cfg['frames']:
                    receipt = cache/'cells'/f'{source["id"]}-{frame}'/'complete.json'
                    record = json.loads(receipt.read_text())
                    if record['identity'] != plan['identity']: raise ValueError(f'Relaxed plan identity mismatch: {receipt}')
                    cells.append(dict(source=source['id'], frame=frame, receipt=str(receipt),
                                      receipt_sha256=file_hash(receipt), archive=record['archive']))
    roots = [s['lineage'] for s in sources]
    if len(set(roots)) != len(roots): raise ValueError('Duplicate transfer root ancestry')
    selected = dict(identity=study.identity, sources=sources, cells=cells,
                    original_plan_sha256=file_hash(resolve_path(cfg['plan'])),
                    no_test_or_calibration_sources=True, selection='Outcome-blind seeded selection of completed four-frame cells',
                    coordinates='Archived float16 full cells, decoded before local subtraction; not high-precision denoising data')
    write_json(path, selected)
    return selected


def archive_cell(source, cell, expected_potential):
    receipt = Path(cell['receipt'])
    if file_hash(receipt) != cell['receipt_sha256']: raise ValueError(f'Changed relaxed receipt: {receipt}')
    archive = Path(cell['archive']); meta = json.loads((archive/'metadata.json').read_text())
    if meta['state'] != 'relaxed' or meta['source_frame'] != cell['frame'] or meta['source_manifest_sha256'] != source['manifest_sha256']:
        raise ValueError(f'Relaxed archive source mismatch: {archive}')
    if meta['fmax_eV_per_A'] > .01 or meta['settings']['minimizer'] != 'fire':
        raise ValueError(f'Relaxation protocol mismatch: {archive}')
    if [meta['potential_checksums'][p] for p in meta['settings']['potential_files']] != expected_potential:
        raise ValueError(f'Relaxation potential mismatch: {archive}')
    cold = TemporalLAMMPSBinaryTrajectory.load(archive/'relaxed_binary_float16')
    cold.verify_checksums()
    conversion = json.loads((archive/'conversion.json').read_text())
    if conversion['source_sha256'] != cold.manifest['source']['sha256'] or int(cold.timesteps[0]) != meta['source_timestep']:
        raise ValueError(f'Relaxed conversion/timestep mismatch: {archive}')
    return cold, dict(metadata_sha256=file_hash(archive/'metadata.json'),
                      conversion_sha256=file_hash(archive/'conversion.json'),
                      binary_manifest_sha256=file_hash(cold.root/'manifest.json'), quantization=conversion['quantization'])


def prepare(study, deadline=None, max_cells=None):
    cfg = study.config['relaxed']; root = resolve_path(cfg['cache'])
    root.mkdir(parents=True, exist_ok=True)
    selected = freeze_selection(study); sources = {s['id']: s for s in selected['sources']}
    plan = json.loads(resolve_path(cfg['plan']).read_text())
    completed = []
    for cell in selected['cells']:
        if max_cells is not None and len(completed) >= max_cells:
            return completed
        remaining(deadline); key = f'{cell["source"]}-{cell["frame"]}'; out = root/'cells'/key
        if (out/'complete.json').exists():
            done = json.loads((out/'complete.json').read_text())
            if done['identity'] != study.identity or file_hash(out/'patches.npz') != done['patches_sha256']:
                raise ValueError(f'Relaxed patch cache changed: {out}')
            completed.append(done); continue
        source = sources[cell['source']]
        raw_path = dataset_path(source['dataset'])/source['relative_trajectory_path']
        if file_hash(raw_path/'manifest.json') != source['manifest_sha256']: raise ValueError(f'Observed source changed: {raw_path}')
        raw = ShootingBinaryTrajectory.load(raw_path)
        cold, provenance = archive_cell(source, cell, plan['config']['potential_sha256'])
        np.testing.assert_array_equal(raw.atom_ids, cold.atom_ids)
        frame = cell['frame']; box = cold.box_high[0].astype(float)-cold.box_low[0].astype(float)
        np.testing.assert_allclose(raw.box_high[frame].astype(float)-raw.box_low[frame].astype(float), box, atol=1e-6, rtol=0)
        if int(raw.timesteps[frame]) != int(cold.timesteps[0]): raise ValueError(f'Observed/relaxed timestep differs: {key}')
        ids = source['center_atom_ids'][:cfg['centers_per_cell']]
        centers = np.searchsorted(raw.atom_ids, ids)
        np.testing.assert_array_equal(raw.atom_ids[centers], ids)
        values = {}
        for domain, positions in [('observed', raw.positions[frame]), ('relaxed', cold.positions[0])]:
            patches, neighbors, _ = patches_from_snapshot(positions.astype(float), np.diag(box), raw.atom_ids, centers,
                                                          study.manifest['radius_A'], study.pilot['max_atoms'])
            values[domain+'_positions'] = np.concatenate(patches)
            values[domain+'_offsets'] = np.r_[0, np.cumsum([len(p) for p in patches])]
            values[domain+'_atom_ids'] = np.concatenate(neighbors)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out/'patches.npz', **values, center_atom_ids=np.array(ids))
        done = dict(identity=study.identity, key=key, source=source['id'], root=source['lineage'],
            split=source['audit_split'], temperature_K=source['temperature_K'], frame=frame,
            center_atom_ids=ids, patches_sha256=file_hash(out/'patches.npz'), observed_manifest_sha256=source['manifest_sha256'],
            observed_positions_dtype=str(raw.positions.dtype), relaxed_positions_dtype=str(cold.positions.dtype), **provenance)
        write_json(out/'complete.json', done); completed.append(done)
        if len(completed) % 20 == 0: print(f'Relaxed audit prepared {len(completed)}/{len(selected["cells"])} cells', flush=True)
    write_json(root/'manifest.json', dict(identity=study.identity, state='complete', cells=completed,
        radius_A=study.manifest['radius_A'], original_plan_sha256=selected['original_plan_sha256'],
        potential_sha256=plan['config']['potential_sha256'], new_simulations=0,
        support='Separate complete 8 A neighborhoods in each domain around identical tracked center IDs',
        precision='Archived full-box float16; structural audit only; no weak-noise corruption experiment'))
    return completed


def load(study):
    root = resolve_path(study.config['relaxed']['cache'])
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['identity'] != study.identity or manifest['state'] != 'complete': raise ValueError('Incomplete/changed relaxed audit cache')
    patches = dict(observed=[], relaxed=[]); records = []
    for cell in manifest['cells']:
        path = root/'cells'/cell['key']/'patches.npz'
        if file_hash(path) != cell['patches_sha256']: raise ValueError(f'Changed paired patches: {path}')
        with np.load(path) as data:
            for domain in patches:
                x, offsets = data[domain+'_positions'], data[domain+'_offsets']
                patches[domain].extend([x[offsets[i]:offsets[i+1]].copy() for i in range(len(offsets)-1)])
            records.extend([dict(root=cell['root'], source=cell['source'], frame=cell['frame'], center_atom_id=int(i),
                                 split=cell['split'], temperature_K=cell['temperature_K']) for i in data['center_atom_ids']])
    return patches, records


def run(study, device, deadline=None):
    prepare(study, deadline)
    patches, records = load(study); root = study.technical/'relaxed'; root.mkdir(exist_ok=True)
    split = {role: np.array([i for i, r in enumerate(records) if r['split'] == role]) for role in ('fit', 'tune', 'development')}
    chosen = split['development']; chosen_records = [records[i] for i in chosen]
    targets = {}
    for domain in patches:
        path = root/f'{domain}-targets.npz'
        if not path.exists():
            remaining(deadline); target, cov = descriptors(patches[domain], study.manifest['radius_A'])
            np.savez(path, **target, covariates=cov)
        with np.load(path) as data: targets[domain] = {k: data[k] for k in ('radial', 'angular', 'rich', 'covariates')}
    for step in study.config['relaxed']['steps']:
        features = {}
        model = study.model(step, device)
        for domain in patches:
            path = root/f'{step:06d}-{domain}-features.npz'
            if not path.exists():
                pooled, exported = [], []
                with torch.no_grad():
                    for start in range(0, len(records), 16):
                        remaining(deadline)
                        p = model.encoder.pooled(pack(patches[domain][start:start+16], device))
                        pooled.append(p.cpu().numpy()); exported.append(model.encoder.readout(p).cpu().numpy())
                np.savez(path, pooled=np.concatenate(pooled), exported=np.concatenate(exported))
            with np.load(path) as data: features[domain] = {k: data[k] for k in ('pooled', 'exported')}
        for input_domain, target_domain in [('observed', 'observed'), ('relaxed', 'relaxed'), ('observed', 'relaxed')]:
            for representation in ('pooled', 'exported'):
                for family in ('radial', 'angular', 'rich'):
                    destination = root/'probes'/str(step)/f'{input_domain}_to_{target_domain}'/representation/family
                    if (destination/'complete.json').exists(): continue
                    remaining(deadline)
                    x, y = features[input_domain][representation], targets[target_domain][family]
                    fitted = fit_residual_probe(x, y, split['fit'], split['tune'], study.config['probes'],
                                                 study.config['seed'], device, deadline)
                    predictions = predict_probe(fitted, x[chosen], device)
                    actual = (y[chosen]-fitted['target_mean'])/fitted['target_scale']
                    liquid = targets['observed']['covariates'][chosen, 2] < .35
                    metrics = {name: dict(all=error_metrics(actual, prediction, chosen_records),
                        liquid=error_metrics(actual[liquid], prediction[liquid], [r for r, v in zip(chosen_records, liquid) if v]),
                        groups={k: error_metrics(actual[:, ids], prediction[:, ids], chosen_records) for k, ids in target_groups(family).items()})
                        for name, prediction in zip(('ridge', 'residual'), predictions, strict=True)}
                    destination.mkdir(parents=True, exist_ok=True)
                    np.savez(destination/'predictions.npz', target=actual, ridge=predictions[0], residual=predictions[1],
                             indices=chosen, roots=np.array([r['root'] for r in chosen_records]), liquid=liquid)
                    torch.save({k: v for k, v in fitted.items() if k != 'model'}, destination/'probe.pt')
                    write_json(destination/'complete.json', dict(identity=study.identity, encoder_step=step,
                        input_domain=input_domain, target_domain=target_domain, representation=representation, family=family,
                        selected_step=fitted['selected_step'], ridge_tuning_mse=fitted['ridge_tuning_mse'],
                        selected_tuning_mse=fitted['selected_tuning_mse'], metrics=metrics, development_roots=15,
                        q6_population='Observed-domain q6<0.35; same mask for both input/target domains'))
                    print(f'Relaxed probe: {step}/{input_domain}->{target_domain}/{representation}/{family}', flush=True)
    write_json(root/'complete.json', dict(identity=study.identity, state='complete',
        rows={role: len(ids) for role, ids in split.items()}, roots={role: len({records[i]['root'] for i in ids}) for role, ids in split.items()},
        encoder_steps=study.config['relaxed']['steps'], encoder_training=False, new_simulations=0))
