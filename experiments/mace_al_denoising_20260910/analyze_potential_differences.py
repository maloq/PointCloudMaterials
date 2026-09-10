"""Explain the nine completed matched-potential cases; no new MD or encoder training.

Experiment recipe: pointnet Python, from the repository root, PYTHONPATH=.
Reads potential_audit.json and the completed audit artifacts. Derived results go
under potential_audit/deeper_analysis; original targets and predictions are read only.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

import gudhi
import numpy as np
from scipy.spatial import cKDTree

from src.analysis.liquid_structure import ORDER_NAMES, bond_order, persistence_image
from src.data_utils.conversion.relaxation import read_relaxed
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.simulation.relaxation import sha256


TOPIC = Path('experiments/mace_al_denoising_20260910')
AUDIT = json.loads((TOPIC / 'potential_audit.json').read_text())
ROOT = Path(AUDIT['output'])
OUT = ROOT / 'deeper_analysis'
POTENTIALS = ('Lee2003_MEAM', 'Al1_EAM_FS')
BLOCKS = (slice(0, 16), slice(16, 80), slice(80, 144))
RADIAL_EDGES = np.linspace(0, 6., 121)


def verify_potential_identity():
    import hashlib
    url = 'https://raw.githubusercontent.com/lammps/lammps/patch_2Sep2026/potentials/Al_mm.eam.fs'
    remote = urlopen(url, timeout=30).read()
    local = Path(AUDIT['potentials']['Al1_EAM_FS']['potential_files'][0]).read_bytes()
    assert ' '.join(remote.decode().splitlines()[3:]).split() == ' '.join(local.decode().splitlines()[3:]).split(), \
        'Local Al1 potential differs from official Al_mm beyond its comment header.'
    (OUT/'potential_identity.json').write_text(json.dumps(dict(source_url=url,
        remote_sha256=hashlib.sha256(remote).hexdigest(), local_sha256=hashlib.sha256(local).hexdigest(),
        same_noncomment_tokens=True),indent=2)+'\n')


def minimum_image(vectors, lengths):
    return vectors - lengths * np.round(vectors / lengths)


def ptm(positions, lengths):
    from ovito.data import DataCollection
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    from ovito.pipeline import Pipeline, StaticSource
    data = DataCollection()
    data.create_particles(count=len(positions)).create_property('Position', data=positions)
    data.create_cell(np.column_stack((np.diag(lengths), np.zeros(3))), pbc=(True, True, True))
    pipeline = Pipeline(source=StaticSource(data=data))
    modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.1)
    modifier.structures[4].enabled = True  # Include local icosahedral motifs.
    pipeline.modifiers.append(modifier)
    return np.asarray(pipeline.compute().particles['Structure Type']).copy()


def structure(positions, lengths, centers):
    tree = cKDTree(positions, boxsize=lengths)
    distances, indices = tree.query(positions[centers], k=160, workers=1)
    assert distances[:, -1].min() > 6., 'The RDF query did not reach 6 Angstrom.'
    _, neighbor_indices = tree.query(positions[indices[:, :13]], k=13, workers=1)
    vectors = minimum_image(positions[neighbor_indices[:, :, 1:]] - positions[indices[:, :13], None], lengths)
    order, counts = bond_order(vectors, 3.5)
    shell_volumes = 4*np.pi/3 * np.diff(RADIAL_EDGES**3)
    rdf = np.histogram(distances[:, 1:], bins=RADIAL_EDGES)[0] / (len(centers)*len(positions)/np.prod(lengths)*shell_volumes)
    labels = ptm(positions, lengths)
    return dict(order=order, coherent_bonds=counts[:, 1], neighbors12=indices[:, 1:13],
                neighbors79=indices[:, 1:80], distance12=distances[:, 1:13].mean(1),
                distance12_std=distances[:, 1:13].std(1),
                nearest=distances[:, 1], coordination35=(distances[:, 1:] < 3.5).sum(1),
                rdf=rdf, ptm=labels)


def diagrams(points):
    tree = gudhi.AlphaComplex(points=points, precision='safe').create_simplex_tree()
    tree.compute_persistence(homology_coeff_field=2, min_persistence=0.)
    stats = []
    for dim in range(3):
        pairs = tree.persistence_intervals_in_dimension(dim)
        pairs = np.sqrt(np.maximum(pairs[np.isfinite(pairs[:, 1])], 0.))
        excluded = np.sum(pairs[:, 1] > 3.5)
        pairs = pairs[pairs[:, 1] <= 3.5]
        life = pairs[:, 1] - pairs[:, 0]
        stats.append([len(pairs), pairs[:, 0].mean(), pairs[:, 1].mean(), life.mean(),
                      life.sum(), np.sum(life > .05), np.sum(life > .1), excluded])
    return stats


def case_analysis(source):
    directory = ROOT / source['name']
    saved = json.loads((directory / 'case.json').read_text())
    for name, digest in saved['checksums'].items():
        assert sha256(directory / name) == digest, f'Changed audit input: {directory/name}'
    with np.load(directory / 'observations.npz') as observations:
        centers = observations['centers']
        identities = observations['neighbor_ids'] - 1
        prediction = observations['predictions'].astype(float)
        hot_clouds = observations['histories'][:, -1].astype(float)
    trajectory = ShootingBinaryTrajectory.load(source['path'])
    low = trajectory.box_low[AUDIT['frame']].astype(float)
    lengths = trajectory.box_high[AUDIT['frame']].astype(float) - low
    positions = [np.mod(trajectory.positions[AUDIT['frame']].astype(float)-low, lengths)]
    metadata = []
    for potential in POTENTIALS:
        x, meta = read_relaxed(directory / potential)
        positions.append(np.mod(x-low, lengths))
        metadata.append(meta)
    np.testing.assert_array_equal(identities[:, 0], centers)
    states = [structure(x, lengths, centers) for x in positions]
    clouds = [np.load(directory / f'{p}_clouds.npy').astype(float) for p in POTENTIALS]
    targets = [np.load(directory / f'{p}_targets.npy').astype(float) for p in POTENTIALS]
    normalized_targets, diagram_stats, quantization_targets = [], [], []
    for index, (x, cloud) in enumerate(zip(positions[1:], clouds)):
        normalized = cloud * (2.86 / states[index+1]['distance12'])[:, None, None]
        normalized_targets.append(np.stack([persistence_image(c) for c in normalized]))
        diagram_stats.append(np.array([diagrams(c) for c in cloud]))
        exact_cloud = minimum_image(x[identities[:16]] - x[centers[:16], None], lengths)
        quantization_targets.append(np.stack([persistence_image(c) for c in exact_cloud]))
    a, b = clouds
    ac, bc = a-a.mean(1, keepdims=True), b-b.mean(1, keepdims=True)
    u, _, vh = np.linalg.svd(ac.transpose(0, 2, 1) @ bc)
    reflection = np.ones((len(a), 3)); reflection[:, -1] = np.linalg.det(u @ vh)
    rotation = (u * reflection[:, None]) @ vh
    rotated = ac @ rotation
    scale = (rotated*bc).sum((1,2)) / (rotated**2).sum((1,2))
    residual = bc - scale[:, None, None]*rotated
    # Match EAM's first-shell mean to MEAM without moving the MEAM descriptor
    # on its fixed Angstrom grid. This complements common-distance normalization.
    matched_clouds = b * (states[1]['distance12']/states[2]['distance12'])[:, None, None]
    matched_targets = np.stack([persistence_image(c) for c in matched_clouds])
    overlap12 = np.array([len(set(a)&set(b))/12 for a,b in zip(states[1]['neighbors12'], states[2]['neighbors12'])])
    overlap79 = np.array([len(set(a)&set(b))/79 for a,b in zip(states[1]['neighbors79'], states[2]['neighbors79'])])
    displacements = np.stack([np.linalg.norm(minimum_image(x-y, lengths), axis=1)
                             for x,y in ((positions[1],positions[0]), (positions[2],positions[0]), (positions[2],positions[1]))])
    result = dict(name=source['name'], temperature_K=source['temperature_K'],
        atom_count=len(positions[0]), box_lengths_A=lengths.tolist(),
        force_max=[m['fmax_eV_per_A'] for m in metadata],
        displacement_mean_A=displacements.mean(1).tolist(),
        displacement_quantiles_A=np.quantile(displacements, [.5,.9,.99], axis=1).T.tolist(),
        patch_similarity_rmsd_A=float(np.sqrt((residual**2).sum(2).mean())),
        patch_similarity_scale_mean=float(scale.mean()),
        patch_similarity_scale_std=float(scale.std()),
        neighbor12_overlap=float(overlap12.mean()), neighbor79_overlap=float(overlap79.mean()),
        ptm_fraction=[(np.bincount(s['ptm'], minlength=5)/len(positions[0])).tolist() for s in states],
        ptm_label_agreement=float((states[1]['ptm']==states[2]['ptm']).mean()),
        q6_paired_correlation=float(np.corrcoef(states[1]['order'][:,1],states[2]['order'][:,1])[0,1]),
        order_mean=[s['order'].mean(0).tolist() for s in states],
        nearest_mean_A=[float(s['nearest'].mean()) for s in states],
        first12_mean_A=[float(s['distance12'].mean()) for s in states],
        first12_std_mean_A=[float(s['distance12_std'].mean()) for s in states],
        coordination35_mean=[float(s['coordination35'].mean()) for s in states])
    np.savez_compressed(OUT / f"{source['name']}.npz", targets=np.stack(targets), prediction=prediction,
        normalized_targets=np.stack(normalized_targets), diagram_stats=np.stack(diagram_stats),
        matched_distance_EAM_targets=matched_targets,
        unquantized_targets=np.stack(quantization_targets), rdf=np.stack([s['rdf'] for s in states]),
        orders=np.stack([s['order'] for s in states]), hot_clouds=hot_clouds,
        patch_similarity_scale=scale, patch_similarity_rmsd=np.sqrt((residual**2).sum(2).mean(1)),
        neighbor12_overlap=overlap12, neighbor79_overlap=overlap79)
    (OUT / f"{source['name']}.json").write_text(json.dumps(result, indent=2)+'\n')
    print('ANALYZED', source['name'], 'PTM', result['ptm_fraction'], flush=True)
    return result


def mse(errors, scales):
    return np.array([np.mean(errors[..., block]**2) / scales[d]**2 for d, block in enumerate(BLOCKS)])


def summary(cases):
    arrays = [dict(np.load(OUT / f"{c['name']}.npz")) for c in cases]
    scales = np.load(ROOT.parent / 'scaling.npz')['block_scale']
    targets = np.stack([a['targets'] for a in arrays])
    predictions = np.stack([a['prediction'] for a in arrays])
    delta = targets[:, 1] - targets[:, 0]
    result = dict(cases=cases, state_order=['hot', *POTENTIALS],
        displacement_order=['hot_to_MEAM','hot_to_EAM','MEAM_to_EAM'],
        ptm_order=['Other','FCC','HCP','BCC','ICO'], order_names=ORDER_NAMES,
        diagram_stat_order=['count','birth_mean_A','death_mean_A','lifetime_mean_A','total_lifetime_A',
                            'count_lifetime_gt_0.05A','count_lifetime_gt_0.1A','finite_death_gt_3.5A_excluded'],
        balanced_target_mse=float(mse(delta,scales).mean()),
        block_target_mse=mse(delta, scales).tolist(),
        global_mean_shift_fraction=(mse(delta.mean((0,1)), scales)/mse(delta,scales)).tolist(),
        source_mean_shift_fraction=(mse(delta.mean(1), scales)/mse(delta,scales)).tolist(),
        normalized_block_target_mse=mse(np.stack([a['normalized_targets'][1]-a['normalized_targets'][0] for a in arrays]),scales).tolist(),
        matched_distance_block_target_mse=mse(np.stack([a['matched_distance_EAM_targets']-a['targets'][0] for a in arrays]),scales).tolist(),
        quantization_block_mse=mse(np.stack([a['unquantized_targets']-a['targets'][:,:16] for a in arrays]),scales).tolist(),
        diagram_means=np.stack([a['diagram_stats'] for a in arrays]).mean((0,2)).tolist(),
        within_source_correlations=[], leave_one_source_out=[], by_temperature={})
    for d, block in enumerate(BLOCKS):
        x,y = targets[:,0,:,block],targets[:,1,:,block]
        xc,yc = x-x.mean(1,keepdims=True), y-y.mean(1,keepdims=True)
        result['within_source_correlations'].append(float(np.sum(xc*yc)/np.sqrt(np.sum(xc**2)*np.sum(yc**2))))
    for i, case in enumerate(cases):
        train = np.arange(len(cases)) != i
        shift = delta[train].mean((0,1))
        result['leave_one_source_out'].append(dict(name=case['name'], temperature_K=case['temperature_K'],
            raw_target_mse=float(mse(delta[i],scales).mean()),
            offset_corrected_target_mse=float(mse(delta[i]-shift,scales).mean()),
            predictor_MEAM_mse=float(mse(predictions[i]-targets[i,0],scales).mean()),
            predictor_EAM_mse=float(mse(predictions[i]-targets[i,1],scales).mean()),
            shifted_predictor_EAM_mse=float(mse(predictions[i]+shift-targets[i,1],scales).mean()),
            held_out_constant_MEAM_mse=float(mse(targets[i,0]-targets[train,0].mean((0,1)),scales).mean()),
            held_out_constant_EAM_mse=float(mse(targets[i,1]-targets[train,1].mean((0,1)),scales).mean()),
            source_constant_EAM_mse=float(mse(targets[i,1]-targets[i,1].mean(0),scales).mean())))
    temperatures = np.array([c['temperature_K'] for c in cases])
    for t in np.unique(temperatures):
        mask = temperatures==t
        within = targets[mask] - targets[mask].mean(2, keepdims=True)
        result['by_temperature'][str(int(t))] = dict(
            predictor_mse=[float(mse(predictions[mask]-targets[mask,p],scales).mean()) for p in range(2)],
            target_distance=float(mse(delta[mask],scales).mean()),
            within_source_target_variance=[float(mse(within[:,p],scales).mean()) for p in range(2)],
            ptm_fraction=np.array([c['ptm_fraction'] for c,m in zip(cases,mask) if m]).mean(0).tolist())
    result['provenance'] = dict(recipe_sha256=sha256(__file__), audit_config_sha256=sha256(TOPIC/'potential_audit.json'),
        audit_metrics_sha256=sha256(ROOT/'metrics.json'), ptm_rmsd_cutoff=0.1, ptm_enabled=['FCC','HCP','BCC','ICO'],
        scale_normalization='Each relaxed patch is rescaled to first-12-neighbor mean distance 2.86 A; membership unchanged.',
        shift_validation='Fit per-pixel EAM-minus-MEAM mean on eight sources, evaluate ninth; post-hoc descriptive cross-validation.',
        quantization='First 16 recorded centers per source: TDA of exact relaxed.dump offsets versus saved float16 local offsets.',
        uncertainty='Nine independent source cells at one early time. No new significance tests on post-hoc statistics.')
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result, arrays, targets, delta, scales)


def evaluate_forces():
    """Evaluate both force fields on all three states, without moving atoms."""
    work_root = OUT/'forces'
    work_root.mkdir(exist_ok=True)
    results = []
    for source in AUDIT['sources']:
        directory = ROOT/source['name']
        force_fields, thermodynamics = [], []
        for potential in POTENTIALS:
            settings = AUDIT['potentials'][potential]
            work = work_root/source['name']/potential
            work.mkdir(parents=True,exist_ok=True)
            commands = ['units metal', 'atom_style atomic', 'boundary p p p',
                f"read_data {(directory/'Lee2003_MEAM/input.data').resolve()}",
                f"mass 1 {settings['mass']}", *settings['pair_commands'],
                'neighbor 2.0 bin', 'neigh_modify delay 0 every 1 check yes',
                'thermo_style custom step pe press fmax fnorm', 'thermo_modify format float %.17g',
                'variable energy equal pe/atoms', 'variable pressure equal press']
            for state in ('hot', *POTENTIALS):
                if state != 'hot':
                    meta = json.loads((directory/state/'metadata.json').read_text())
                    commands.append(f"read_dump {(directory/state/'relaxed.dump').resolve()} {meta['source_timestep']} x y z box yes replace yes")
                commands.extend(['run 0', f'print "STATE {state} ${{energy}} ${{pressure}}"',
                    f'write_dump all custom {state}_forces.dump id fx fy fz modify sort id format line "%d %.17g %.17g %.17g"'])
            (work/'in.lammps').write_text('\n'.join(commands)+'\n')
            with (work/'stdout.log').open('w') as log:
                subprocess.run([*settings['lammps_command'],'-in','in.lammps'],cwd=work,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True,timeout=180)
            table = [line.split() for line in (work/'stdout.log').read_text().splitlines() if line.startswith('STATE ')]
            assert len(table)==3, f'Incomplete force evaluation in {work}'
            thermodynamics.append([[float(row[2]),float(row[3])/10000] for row in table])
            forces = []
            for state in ('hot', *POTENTIALS):
                rows = np.loadtxt(work/f'{state}_forces.dump',skiprows=9)
                np.testing.assert_array_equal(rows[:,0],np.arange(1,70305))
                forces.append(rows[:,1:])
            force_fields.append(np.stack(forces))
        forces = np.stack(force_fields)
        np.savez_compressed(work_root/f"{source['name']}.npz",forces=forces)
        hot_a,hot_b = forces[:,0]
        direction_cosines = np.sum(hot_a*hot_b,axis=1)/(np.linalg.norm(hot_a,axis=1)*np.linalg.norm(hot_b,axis=1))
        record = dict(name=source['name'],temperature_K=source['temperature_K'],
            energy_eV_atom_and_pressure_GPa=thermodynamics,
            force_vector_rms_eV_A=np.sqrt((forces**2).sum(3).mean(2)).tolist(),
            force_component_max_eV_A=np.abs(forces).max((2,3)).tolist(),
            hot_force_cosine_mean=float(direction_cosines.mean()),
            hot_force_negative_cosine_fraction=float((direction_cosines<0).mean()))
        for p in range(2):
            assert record['force_component_max_eV_A'][p][p+1] <= .010001, 'Own-potential force changed.'
        results.append(record)
        print('FORCES',source['name'],record['hot_force_cosine_mean'],flush=True)
    (OUT/'force_metrics.json').write_text(json.dumps(dict(cases=results,
        potential_order=POTENTIALS,state_order=['hot',*POTENTIALS],
        protocol='Single-point evaluation of exact saved coordinates, zero velocities, no minimization or integration.',
        recipe_sha256=sha256(__file__)),indent=2)+'\n')


def plot(result, arrays, targets, delta, scales):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(15,8),layout='constrained')
    colors = ['#777777','#2878b5','#d65f00']
    labels = ['Hot MEAM input','MEAM relaxation','EAM relaxation']
    rdf = np.stack([a['rdf'] for a in arrays]).mean(0)
    for i in range(3):
        axes[0,0].plot((RADIAL_EDGES[:-1]+RADIAL_EDGES[1:])/2,rdf[i],color=colors[i],label=labels[i])
    axes[0,0].set(xlim=(2,6),xlabel='Distance (Å)',ylabel='g(r)',title='Same density; different local packing')
    axes[0,0].legend(fontsize=8)
    ptm_fractions = np.array([c['ptm_fraction'] for c in result['cases']]).mean(0)
    for i in range(3):
        axes[0,1].bar(np.arange(4)+(i-1)*.25,100*ptm_fractions[i,1:],width=.25,color=colors[i])
    axes[0,1].set(xticks=np.arange(4),xticklabels=result['ptm_order'][1:],ylabel='Atoms (%)',title='PTM ordered motifs (cutoff 0.1)')
    others = ptm_fractions[:,0]*100
    axes[0,1].text(.02,.98,f'Other: hot {others[0]:.2f}%, MEAM {others[1]:.2f}%, EAM {others[2]:.2f}%',
                   transform=axes[0,1].transAxes,va='top',fontsize=8)
    axes[0,1].set_ylim(0,2.65)
    for p in range(2):
        axes[0,2].plot(np.linspace(.7,2.5,16),targets[:,p,:,:16].mean((0,1)),label=labels[p+1],color=colors[p+1])
    axes[0,2].set(xlim=(1,1.8),xlabel='Component merge radius (Å)',ylabel='H0 descriptor',title='Connectivity length scales shift')
    axes[0,2].legend(fontsize=8)
    x=np.arange(3)
    raw=np.array(result['block_target_mse'])
    source_fraction=np.array(result['source_mean_shift_fraction'])
    axes[1,0].bar(x,raw*source_fraction,label='Source mean shift',color='#d65f00')
    axes[1,0].bar(x,raw*(1-source_fraction),bottom=raw*source_fraction,label='Neighborhood-dependent residual',color='#2878b5')
    axes[1,0].set(xticks=x,xticklabels=['H0','H1','H2'],ylabel='Scaled squared target difference',title='Most descriptor difference is systematic')
    axes[1,0].legend(fontsize=8)
    corr=result['within_source_correlations']
    axes[1,1].bar(x,corr,color='#2878b5')
    axes[1,1].set(xticks=x,xticklabels=['H0','H1','H2'],ylim=(0,1),ylabel='Centered paired correlation',title='Local variation transfers only partly')
    methods=['Target gap','After offset correction','After distance normalization','Float16 error']
    values=[raw.mean(),np.mean([r['offset_corrected_target_mse'] for r in result['leave_one_source_out']]),
            np.mean(result['normalized_block_target_mse']),np.mean(result['quantization_block_mse'])]
    axes[1,2].barh(methods,values,color=['#d65f00','#2878b5','#8c6bb1','#777777'])
    axes[1,2].set(xscale='log',xlabel='Balanced squared descriptor difference',title='Controls: offset, scale, quantization')
    fig.suptitle('Al potential audit: 9 matched 70,304-atom cells, 2,304 identical 80-atom patches',fontsize=14)
    fig.savefig(OUT/'comparison.png',dpi=180)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    verify_potential_identity()
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn')) as pool:
        cases = list(pool.map(case_analysis, AUDIT['sources']))
    summary(cases)
    evaluate_forces()
    print('COMPLETE', OUT, flush=True)
