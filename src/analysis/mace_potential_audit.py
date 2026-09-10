"""Matched-input target-potential sensitivity, after the mixed-data comparison."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import permutation_test
import torch

from src.analysis.liquid_structure import persistence_image
from src.data_utils.conversion.relaxation import read_relaxed
from src.data_utils.mace_denoising import signature
from src.data_utils.mace_history import history_clouds
from src.data_utils.mace_relaxed import paired_clouds
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.temporal_campaign import write_json
from src.models.encoders.mace_denoising import PretrainedMACEDenoisingEncoder
from src.simulation.relaxation import relax_frame, sha256
from src.training_methods.mace_denoising import BLOCKS, Predictor, raw_prediction


def paired_statistics(reference, candidate, temperatures, seed):
    """One error mean per independent source, with temperature-stratified intervals."""
    test = permutation_test((reference, candidate), lambda a, b: np.mean(b-a),
        permutation_type='samples', n_resamples=np.inf, vectorized=False, alternative='two-sided')
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(np.flatnonzero(temperatures==t),
        size=(4000, int(np.sum(temperatures==t))), replace=True) for t in np.unique(temperatures)], axis=1)
    delta = candidate-reference
    relative = (candidate[draws].mean(1)-reference[draws].mean(1))/reference[draws].mean(1)
    return dict(source_count=len(reference), reference_mean=float(reference.mean()), candidate_mean=float(candidate.mean()),
        mean_difference=float(delta.mean()), relative_difference=float(delta.mean()/reference.mean()),
        exact_two_sided_p=float(test.pvalue), permutation_count=len(test.null_distribution),
        difference_bootstrap_95_interval=np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
        relative_bootstrap_95_interval=np.quantile(relative, [.025, .975]).tolist(),
        per_source_difference=delta.tolist())


def audit_existing_minimizers(cfg, scaling, directory):
    records = []
    for pair in cfg['relaxation_pairs']:
        cg, fire = (Path(pair[k]).parent for k in ('cg_manifest', 'fire_manifest'))
        for name in ('histories.npy', 'centers.npy', 'neighbor_ids.npy'):
            np.testing.assert_array_equal(np.load(cg/name), np.load(fire/name))
        a, b = np.load(cg/'targets.npy'), np.load(fire/'targets.npy')
        mse = [float(np.mean((a[:, block]-b[:, block])**2)) for block in BLOCKS]
        original = json.loads(Path(pair['cg_manifest']).read_text())
        records.append(dict(name=cg.name, source_index=original['source_index'],
            block_target_mse=mse, balanced_target_distance=float(np.mean(np.array(mse)/scaling['block_scale']**2))))
    write_json(directory/'cg_fire_target_sensitivity.json', dict(pairs=records,
        source_count=len({r['source_index'] for r in records}),
        interpretation='Same MEAM potential and thermal inputs, different converged minimizers. Descriptor differences only; two sources are insufficient for a reliable general significance claim.'))


@torch.no_grad()
def run_audit(cfg, data, exports):
    from src.analysis.mace_denoising import score
    audit = json.loads(Path(cfg['potential_audit_config']).read_text())
    root = Path(audit['output'])
    root.mkdir(parents=True, exist_ok=True)
    build = json.loads((Path(cfg['output'])/'potential_build.json').read_text())
    backend = json.loads((Path(cfg['output'])/'potential_backend_check.json').read_text())
    if build['state'] != 'built' or backend['state'] != 'passed' or backend['binary_sha256'] != sha256(build['binary']):
        raise RuntimeError('The shared EAM/MEAM GPU binary must pass its CPU force comparison before the potential audit.')
    export = Path(exports[audit['predictor']])
    payload = torch.load(export, map_location='cuda', weights_only=False)
    encoder = PretrainedMACEDenoisingEncoder(**payload['encoder_kwargs']).cuda().eval()
    encoder.load_state_dict(payload['encoder'], strict=True)
    variant = next(v for v in cfg['variants'] if v['name']==audit['predictor'])
    model = Predictor(cfg, variant).cuda().eval()
    checkpoint = torch.load(Path(cfg['output'])/'runs'/f"{audit['predictor']}_seed{payload['seed']}"/'best.pt',
                            map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    used = {r['lineage'] for r in data.records}
    assert all('independent_melt_'+str(s['preparation_seed']) not in used for s in audit['sources'])
    offsets = np.array(cfg['frame_offsets_ps'])
    frame = audit['frame']
    cases = []
    with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=get_context('spawn')) as pool:
        for index, source in enumerate(audit['sources']):
            directory = root/source['name']
            directory.mkdir(exist_ok=True)
            sig = signature(dict(audit=audit, source=source, encoder_sha256=sha256(export),
                binary_sha256=build['binary_sha256'], implementation_sha256=sha256(__file__)))
            completion = directory/'case.json'
            if completion.exists():
                saved = json.loads(completion.read_text())
                if saved['signature'] != sig:
                    raise ValueError(f'Potential-audit configuration changed: {completion}')
                for name, digest in saved['checksums'].items():
                    if sha256(directory/name) != digest:
                        raise ValueError(f'Changed potential-audit artifact: {directory/name}')
                cases.append(saved)
                continue
            write_json(Path(cfg['output'])/'status.json', dict(state='paired_potential_audit',
                completed_sources=len(cases), total_sources=len(audit['sources']), source=source['name']))
            trajectory = ShootingBinaryTrajectory.load(source['path'])
            trajectory.verify_checksums()
            steps = np.rint(offsets/source['cadence_ps']).astype(np.int64)
            np.testing.assert_allclose(steps*source['cadence_ps'], offsets, rtol=0, atol=1e-9)
            rng = np.random.default_rng(np.random.SeedSequence([audit['seed'], index, frame]))
            centers = rng.choice(trajectory.atom_count, audit['centers'], replace=False)
            low = trajectory.box_low[frame].astype(np.float64)
            lengths = trajectory.box_high[frame].astype(np.float64)-low
            hot = trajectory.positions[frame].astype(np.float64)-low
            targets, relaxations = {}, {}
            reference_ids = None
            input_digest = None
            for potential, settings in audit['potentials'].items():
                work = directory/potential
                if not (work/'metadata.json').exists():
                    relax_frame(trajectory, frame, work, settings)
                relaxed, metadata = read_relaxed(work)
                if metadata['settings'] != settings:
                    raise ValueError(f'Potential-audit relaxation settings changed: {work}')
                anchor, quenched, neighbors, errors = paired_clouds(hot, relaxed-low, lengths, centers)
                identities = trajectory.atom_ids[neighbors]
                if reference_ids is None:
                    reference_ids, input_digest = identities, metadata['input_sha256']
                else:
                    np.testing.assert_array_equal(identities, reference_ids)
                    assert metadata['input_sha256']==input_digest, 'Potential comparison used different initial cells.'
                targets[potential] = np.stack(list(pool.map(persistence_image, quenched.astype(np.float32), chunksize=32)))
                np.save(directory/f'{potential}_targets.npy', targets[potential])
                np.save(directory/f'{potential}_clouds.npy', quenched)
                if not (work/'conversion.json').exists():
                    with (work/'conversion_stdout.log').open('w') as log:
                        subprocess.run([sys.executable, 'scripts/convert_trajectory.py', 'relaxation', str(work)],
                                       stdout=log, check=True)
                relaxations[potential] = dict(fmax_eV_per_A=metadata['fmax_eV_per_A'], seconds=metadata['seconds'],
                    local_quantization_max_A=errors, potential_checksums=metadata['potential_checksums'])
            history, error = history_clouds(trajectory, centers, reference_ids, frame, steps)
            np.testing.assert_array_equal(history[:, -1], anchor)
            predictions = []
            for start in range(0, len(history), cfg['atom_batch_size']):
                points = torch.from_numpy(history[start:start+cfg['atom_batch_size']].astype(np.float32)).cuda()
                times = torch.tensor(offsets, dtype=torch.float32, device='cuda').repeat(len(points), 1)
                z = encoder(points, torch.zeros(len(points), device='cuda', dtype=torch.long), times)
                hidden = model.trunk(z)
                predictions.append(torch.cat([head(hidden) for head in model.heads], -1).cpu().numpy())
            prediction = raw_prediction(np.concatenate(predictions), data.scaling, 'blocks')
            metrics = {potential: score(prediction, target, np.zeros(len(centers), np.int64),
                np.full(len(centers), source['temperature_K']), data.scaling)[0] for potential, target in targets.items()}
            delta = targets['Al1_EAM_FS']-targets['Lee2003_MEAM']
            distance = float(np.mean([np.mean(delta[:, block]**2)/data.scaling['block_scale'][d]**2 for d, block in enumerate(BLOCKS)]))
            np.savez_compressed(directory/'observations.npz', histories=history, centers=centers, neighbor_ids=reference_ids,
                                frame_offsets_ps=offsets, predictions=prediction)
            saved = dict(state='complete', signature=sig, source=source, metrics=metrics, relaxations=relaxations,
                input_sha256=input_digest, balanced_target_distance=distance, history_quantization_max_A=error,
                checksums={p.name:sha256(p) for p in directory.iterdir() if p.suffix in ('.npy','.npz')})
            write_json(completion, saved)
            cases.append(saved)
            print('POTENTIAL_PAIR_COMPLETE', source['name'], distance, flush=True)
    temperatures = np.array([c['source']['temperature_K'] for c in cases])
    a = np.array([c['metrics']['Lee2003_MEAM']['balanced_mse'] for c in cases])
    b = np.array([c['metrics']['Al1_EAM_FS']['balanced_mse'] for c in cases])
    primary = paired_statistics(a, b, temperatures, audit['seed'])
    secondary = {}
    for d in range(3):
        a_block = np.array([c['metrics']['Lee2003_MEAM']['blocks'][f'H{d}']['scaled_mse'] for c in cases])
        b_block = np.array([c['metrics']['Al1_EAM_FS']['blocks'][f'H{d}']['scaled_mse'] for c in cases])
        secondary[f'H{d}'] = paired_statistics(a_block, b_block, temperatures, audit['seed'])
    pvalues = np.array([secondary[f'H{d}']['exact_two_sided_p'] for d in range(3)])
    order = np.argsort(pvalues)
    adjusted = np.empty(3)
    adjusted[order] = np.minimum(1., np.maximum.accumulate((3-np.arange(3))*pvalues[order]))
    for d in range(3):
        secondary[f'H{d}']['holm_p'] = float(adjusted[d])
    write_json(root/'metrics.json', dict(state='complete', primary=primary, homology=secondary, cases=cases,
        mean_balanced_target_distance=float(np.mean([c['balanced_target_distance'] for c in cases])),
        predictor=audit['predictor'], selected_seed=payload['seed'], scope=audit['scope'],
        inference='Exact paired swaps require label exchangeability under the null. Bootstrap resamples independent sources within temperature. Lack of significance is not evidence of equivalence.'))
    low, high = primary['relative_bootstrap_95_interval']
    (root/'RESULTS.md').write_text('\n'.join(['# Matched potential audit', '', audit['scope'], '',
        f"Nine independent sources; {audit['centers']} identical neighborhoods per source; same FIRE procedure for both potentials.", '',
        f"Predictor: {audit['predictor']}, seed {payload['seed']} selected on the earlier validation split.", '',
        f"MEAM-target balanced MSE: {primary['reference_mean']:.6f}; EAM-target MSE: {primary['candidate_mean']:.6f}.",
        f"EAM minus MEAM relative error difference: {100*primary['relative_difference']:.2f}% (source bootstrap 95% interval [{100*low:.2f}%, {100*high:.2f}%]).",
        f"Exact paired two-sided p = {primary['exact_two_sided_p']:.6g}, using {primary['permutation_count']} assignments. Primary alpha: 0.05; prespecified practical effect: 5% relative error.", '',
        'The test unit is an independent melt source. Neighborhoods are averaged within each source, not treated as independent replications. Secondary H0/H1/H2 tests use Holm correction.', '',
        'This tests sensitivity of the relaxed target and prediction error to the potential on fixed observed inputs. It does not test how the potential changes the distribution of generated trajectories. Nonsignificance does not establish equivalence.', '',
        '[Full metrics](metrics.json). [Paired permutation method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).'])+'\n')
