"""Read-only comparison of completed dense GPU Al with recorded CPU descendants."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.al_replay import structure
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import result_folders, write_json
from src.experiment_runner.metric_docs import fingerprint, write_metric_rows
from src.project_runtime.paths import load_json

FAMILY = 'meam_dense_comparison'
ATOMS = 70304
THERMO = ['temperature_K', 'pressure_bar', 'volume_A3_per_atom', 'energy_eV_per_atom']


def read_json(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    actual = fingerprint(path)
    if actual != expected:
        raise ValueError(f'{path}: SHA256 differs: {actual} != {expected}')


def thermo_log(path):
    """Native run boundaries repeat steps; retain the last force reevaluation.

    GPU force reductions can differ slightly at run boundaries. Record the
    largest duplicate discrepancy rather than treating repeats as extra time.
    """
    rows, active = {}, False
    duplicate_max = np.zeros(4)
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields == ['Step', 'Temp', 'Press', 'Volume', 'PotEng']:
            active = True
            continue
        # The recorded recovery concatenates a restart input after an interrupted
        # thermo block, which has no Loop-time footer.
        if line.startswith(('Loop time of', 'read_restart ')):
            active = False
        if not active or not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f'{path}: unexpected thermodynamic row {line!r}')
        step = int(fields[0])
        values = np.array([float(x) for x in fields[1:]])
        if step in rows:
            duplicate_max = np.maximum(duplicate_max, np.abs(values - rows[step]))
        rows[step] = values
    if not rows:
        raise ValueError(f'{path}: no thermodynamic samples')
    values = np.array([rows[s] for s in sorted(rows)])
    values[:, 2:] /= ATOMS
    frame = pd.DataFrame(values, columns=THERMO)
    frame.insert(0, 'time_ps', np.array(sorted(rows)) * .002)
    return frame, duplicate_max.tolist()


def historical(record, inputs):
    d = Path(record['parent_directory'])
    checked(d/'outcome.json', record['parent_input_sha256']['outcome.json'])
    o = read_json(d/'outcome.json')
    for filename, field in [('crystallization_progress.npz', 'progress_artifact'),
                            ('thermodynamics.npz', 'thermodynamics_artifact')]:
        checked(d/filename, o[field]['sha256'])
        inputs[str(d/filename)] = o[field]['sha256']
    with np.load(d/'crystallization_progress.npz') as z:
        if not np.array_equal(z['time_ps'], np.arange(801)*.75):
            raise ValueError(f'{d}: historical timeline changed')
        curve = pd.DataFrame(dict(time_ps=z['time_ps'][::20],
            crystal_fraction=z['crystalline_fraction'][::20].astype(float),
            largest_cluster_atoms=z['largest_crystalline_cluster_atoms'][::20]))
    with np.load(d/'thermodynamics.npz') as z:
        if not np.array_equal(z['step'], np.arange(801)*250):
            raise ValueError(f'{d}: historical thermo steps changed')
        thermo = pd.DataFrame(dict(time_ps=z['step']*.003,
            temperature_K=z['temperature_K'].astype(float),
            pressure_bar=z['pressure_GPa'].astype(float)*10000,
            volume_A3_per_atom=z['volume_A3']/ATOMS,
            energy_eV_per_atom=z['potential_energy_eV_per_atom']))
    return curve, thermo


def dense(record, directory, cached, cache_hashes, inputs):
    metadata = read_json(directory/'input_metadata.json')
    status = read_json(directory/'status.json')
    gpu = metadata.get('execution_backend') == 'torch_meam_lammps_iso_npt_v1'
    if metadata['atom_count'] != ATOMS or metadata['timestep_ps'] != .002:
        raise ValueError(f'{directory}: unexpected native integration contract')
    for name in ['input_metadata.json', 'status.json', 'trajectory_binary_float16/manifest.json']:
        inputs[str(directory/name)] = fingerprint(directory/name)
    monitor = directory/'technical/stop-monitor/history.json'
    if monitor.exists():
        inputs[str(monitor)] = fingerprint(monitor)
        rows = read_json(monitor)
        curve = pd.DataFrame([dict(time_ps=r['time_ps'], crystal_fraction=r['crystal_fraction'],
            largest_cluster_atoms=r['largest_crystal_cluster']) for r in rows])
        outcome = read_json(directory/'technical/stop-monitor/outcome.json')
        inputs[str(directory/'technical/stop-monitor/outcome.json')] = fingerprint(directory/'technical/stop-monitor/outcome.json')
        if curve.time_ps.iloc[-1] != outcome['measurement_ps']:
            curve.loc[len(curve)] = [outcome['measurement_ps'], outcome['final_crystal_fraction'],
                                    outcome['final_largest_crystal_cluster']]
        origin = 'native monitoring snapshot, float32 PTM consumer'
    elif record['source_id'] in cache_hashes:
        checked(directory/'trajectory_binary_float16/manifest.json',
                cache_hashes[record['source_id']]['manifest_sha256'])
        curve = cached[cached.source_id == record['source_id']][
            ['time_ps', 'crystal_fraction', 'largest_cluster_fraction']].copy()
        curve['largest_cluster_atoms'] = np.rint(curve.pop('largest_cluster_fraction')*ATOMS).astype(int)
        origin = 'reused verified float16 PTM observations'
    else:
        traj = ShootingBinaryTrajectory.load(directory/'trajectory_binary_float16')
        if not np.array_equal(traj.timesteps, np.arange(6001)*50):
            raise ValueError(f'{directory}: expected complete 600 ps trajectory')
        rows = []
        for t in range(0, 601, 15):
            i = t*10
            _, values = structure(traj.positions[i].astype(float),
                                 (traj.box_high[i]-traj.box_low[i]).astype(float))
            rows.append(dict(time_ps=float(t), crystal_fraction=values['crystal_fraction'],
                             largest_cluster_atoms=values['largest_crystal_cluster']))
        curve = pd.DataFrame(rows)
        origin = 'new PTM observations on verified float16 trajectory'
        print(f'PTM completed source {record["source_id"]}', flush=True)
    if curve.time_ps.duplicated().any() or np.any(np.diff(curve.time_ps) <= 0):
        raise ValueError(f'{directory}: duplicate or unordered structural times')
    # A restart-only publication can lack the original thermodynamic prefix.
    # Keep those missing observations explicit; do not reconstruct from positions.
    frames, log_audit = [], []
    for name in ['measurement.lammps.log', 'measurement.resume.lammps.log']:
        path = directory/name
        if path.exists():
            frame, duplicates = thermo_log(path)
            frames.append(frame)
            inputs[str(path)] = fingerprint(path)
            log_audit.append(dict(path=str(path), duplicate_max_native_units=duplicates))
    if not frames:
        raise ValueError(f'{directory}: no native measurement logs')
    thermo = pd.concat(frames).drop_duplicates('time_ps', keep='last').sort_values('time_ps')
    if gpu and not np.array_equal(np.rint(thermo.time_ps*500).astype(int),
                                 np.arange(status['frame_count'])*50):
        raise ValueError(f'{directory}: GPU thermodynamic timeline incomplete')
    return curve, thermo, dict(structure_origin=origin, log_audit=log_audit)


def summarize(record, engine, curve, thermo):
    regular = curve[np.isclose(np.mod(curve.time_ps, 15), 0)]
    times = regular.time_ps.to_numpy()
    f = regular.crystal_fraction.to_numpy()
    hits = np.flatnonzero((f[:-1] >= .5) & (f[1:] >= .5) & (np.diff(times) == 15))
    t50 = float(times[hits[0]]) if len(hits) else None
    row = dict(source_id=record['source_id'], temperature_target_K=record['temperature_K'],
        root_lineage=record['root_lineage'], engine=engine, t50_ps=t50,
        no_confirmed_half=t50 is None, observed_endpoint_ps=float(curve.time_ps.iloc[-1]),
        final_crystal_fraction=float(curve.crystal_fraction.iloc[-1]),
        early_max_crystal_fraction=float(curve[curve.time_ps <= 15].crystal_fraction.max()))
    early = thermo[(thermo.time_ps > 0) & (thermo.time_ps <= 15)]
    expected = 20 if engine == 'CPU_3fs' else 150
    row['early_samples'] = len(early)
    row['early_complete'] = len(early) == expected
    for name in THERMO:
        row['early_mean_'+name] = float(early[name].mean()) if row['early_complete'] else None
    row['logged_mean_temperature_K'] = float(thermo.temperature_K.mean())
    row['logged_mean_pressure_bar'] = float(thermo.pressure_bar.mean())
    row['logged_time_start_ps'] = float(thermo.time_ps.iloc[0])
    row['logged_time_end_ps'] = float(thermo.time_ps.iloc[-1])
    return row


def plots(root, curves, sources, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = dict(CPU_3fs='#a5aab0', CPU_2fs='#2478b0', GPU_2fs='#d44932')
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2), sharey=True, constrained_layout=True)
    for ax, (temp, group) in zip(axes, curves.groupby('temperature_target_K'), strict=True):
        sid = int(pairs[pairs.temperature_target_K == temp].source_id.iloc[0])
        for engine in ['CPU_3fs', 'CPU_2fs', 'GPU_2fs']:
            for source, frame in group[group.engine == engine].groupby('source_id'):
                same = engine == 'CPU_3fs' and source == sid
                color = '#282c34' if same else colors[engine]
                ax.plot(frame.time_ps, frame.crystal_fraction*100, color=color,
                        linewidth=2.5 if engine == 'GPU_2fs' or same else .8,
                        alpha=1 if engine == 'GPU_2fs' or same else .45,
                        linestyle='--' if same else '-')
        ax.axhline(50, color='gray', ls=':', lw=.8)
        ax.set(title=f'{temp:g} K', xlabel='Production time (ps)', xlim=(0, 600), ylim=(0, 100))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Crystalline atoms (%)')
    fig.legend(handles=[Line2D([], [], color=colors['GPU_2fs'], lw=2.5, label='GPU: 2 fs, one run per T'),
        Line2D([], [], color='#282c34', ls='--', label='CPU: 3 fs, same starting melt'),
        Line2D([], [], color=colors['CPU_3fs'], label='CPU: 3 fs, 30 runs per T'),
        Line2D([], [], color=colors['CPU_2fs'], label='CPU: 2 fs, completed dense runs')],
        loc='outside upper center', ncol=2, frameon=False)
    fig.savefig(root/'plots/crystallization.png', dpi=170)
    fig.savefig(root/'plots/crystallization.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    for j, (temp, group) in enumerate(curves.groupby('temperature_target_K')):
        for engine in ['CPU_3fs', 'CPU_2fs', 'GPU_2fs']:
            for _, frame in group[group.engine == engine].groupby('source_id'):
                for i, field in enumerate(['energy_eV_per_atom', 'volume_A3_per_atom']):
                    valid = frame.dropna(subset=[field])
                    axes[i, j].plot(valid.crystal_fraction*100, valid[field], '.-',
                        color=colors[engine], lw=2 if engine == 'GPU_2fs' else .5,
                        markersize=3 if engine == 'GPU_2fs' else 1,
                        alpha=1 if engine == 'GPU_2fs' else .2)
        axes[0, j].set_title(f'{temp:g} K')
        axes[1, j].set_xlabel('Crystalline atoms (%)')
        for ax in axes[:, j]:
            ax.grid(alpha=.15)
    axes[0, 0].set_ylabel('Potential energy (eV/atom)')
    axes[1, 0].set_ylabel('Volume (angstrom³/atom)')
    fig.suptitle('Thermodynamics at observed crystallinity: red GPU, blue 2 fs CPU, gray 3 fs CPU\n'
                 'Instantaneous logged values at exact structural times; missing restart prefixes omitted', fontsize=12)
    fig.savefig(root/'plots/thermodynamics.png', dpi=170)
    fig.savefig(root/'plots/thermodynamics.pdf')
    plt.close(fig)


def run(config):
    os.environ['OVITO_THREAD_COUNT'] = '1'
    root = result_folders(config['output'])
    if (root/'technical/completion.json').exists():
        raise FileExistsError(f'{root}: completed analysis exists')
    main = read_json(Path(config['cpu_launch'])/'manifest.json')
    launch = Path(config['gpu_launch'])
    gpu = read_json(launch/'manifest.json')
    records = main['runs'][:150]
    if len({r['root_lineage'] for r in records}) != 150:
        raise ValueError('Expected 150 main independent ancestors, excluding supplemental descendants')
    old = Path(config['cached_cpu_analysis'])
    cached_path = old/'tables/observations.csv'
    checked(cached_path, read_json(old/'technical/table-contracts/observations.json')['sha256'])
    cached = pd.read_csv(cached_path)
    cached = cached[cached.protocol == 'dense_2fs_010ps']
    cache_hashes = {r['source_id']: r['dense'] for r in read_json(old/'technical/input_hashes.json') if 'dense' in r}
    inputs = {str(cached_path): fingerprint(cached_path)}
    sources, curves, audits, coverage = [], [], [], []
    by_id = {r['source_id']: r for r in records}
    for record in records:
        sid = record['source_id']
        curve, thermo = historical(record, inputs)
        groups = [('CPU_3fs', curve, thermo)]
        directory = Path(main['root'])/record['run_dir']
        state = read_json(directory/'status.json')['state'] if (directory/'status.json').exists() else 'unstarted'
        coverage.append(dict(source_id=sid, dense_state=state, directory=str(directory.resolve())))
        if state == 'complete':
            curve, thermo, audit = dense(record, directory, cached, cache_hashes, inputs)
            engine = 'GPU_2fs' if sid in config['gpu_source_ids'] else 'CPU_2fs'
            actual_gpu = read_json(directory/'input_metadata.json').get('execution_backend') == 'torch_meam_lammps_iso_npt_v1'
            if actual_gpu != (engine == 'GPU_2fs'):
                raise ValueError(f'Unrecorded backend source {sid}')
            audits.append(dict(source_id=sid, **audit))
            groups.append((engine, curve, thermo))
        for engine, curve, thermo in groups:
            sources.append(summarize(record, engine, curve, thermo))
            # Exact observed times only; merge preserves missing thermodynamics.
            frame = curve.merge(thermo, on='time_ps', how='left', validate='one_to_one')
            frame['source_id'], frame['temperature_target_K'], frame['engine'] = sid, record['temperature_K'], engine
            curves.extend(frame.to_dict('records'))
        print(f'Audited {sid}', flush=True)
    source_frame, curve_frame = pd.DataFrame(sources), pd.DataFrame(curves)
    pairs = []
    for sid in config['gpu_source_ids']:
        record = by_id[sid]
        # Verify that paired descendants truly share native preparation and seeds.
        d = Path(main['root'])/record['run_dir']
        parent = Path(record['parent_directory'])
        for name in ['prepared_liquid.lammps.data', 'melt_final.restart.bin']:
            expected = record['parent_input_sha256'][name]
            checked(d/name, expected)
            checked(parent/name, expected)
            inputs[str(d/name)] = expected
        if read_json(parent/'outcome.json')['velocity_seed'] != record['velocity_seed']:
            raise ValueError(f'Source {sid}: velocity seed mismatch')
        a = source_frame[(source_frame.source_id == sid) & (source_frame.engine == 'GPU_2fs')].iloc[0]
        b = source_frame[(source_frame.source_id == sid) & (source_frame.engine == 'CPU_3fs')].iloc[0]
        pairs.append(dict(source_id=sid, temperature_target_K=record['temperature_K'],
            gpu_t50_ps=a.t50_ps, cpu_parent_t50_ps=b.t50_ps,
            early_delta_temperature_K=a.early_mean_temperature_K-b.early_mean_temperature_K,
            early_delta_pressure_bar=a.early_mean_pressure_bar-b.early_mean_pressure_bar,
            early_delta_energy_meV_per_atom=1000*(a.early_mean_energy_eV_per_atom-b.early_mean_energy_eV_per_atom),
            early_delta_volume_percent=100*(a.early_mean_volume_A3_per_atom/b.early_mean_volume_A3_per_atom-1)))
    pair_frame = pd.DataFrame(pairs)
    cohort = []
    for (engine, temp), group in source_frame.groupby(['engine', 'temperature_target_K']):
        events = group.t50_ps.dropna()
        cohort.append(dict(engine=engine, temperature_target_K=temp, sources=len(group),
            confirmed_half=len(events), no_confirmed_half=int(group.no_confirmed_half.sum()),
            t50_conditional_min_ps=events.min(), t50_conditional_median_ps=events.median(),
            t50_conditional_max_ps=events.max(), early_complete=int(group.early_complete.sum())))
    cohort_frame = pd.DataFrame(cohort)
    for name, frame in [('sources', source_frame), ('observations', curve_frame),
                        ('paired_gpu_cpu', pair_frame), ('cohorts', cohort_frame)]:
        rows = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
        write_metric_rows(rows, root, family=FAMILY, name=name)
    write_json(root/'technical/provenance.json', dict(config=config, inputs=inputs, dense_audits=audits,
        coverage=coverage, captured_at=datetime.now(timezone.utc).isoformat(),
        cpu_manifest_sha256=fingerprint(Path(config['cpu_launch'])/'manifest.json'),
        gpu_manifest_sha256=fingerprint(launch/'manifest.json'),
        gpu_launch_source_ids=[r['source_id'] for r in gpu['runs']]))
    plots(root, curve_frame, source_frame, pair_frame)
    write_json(root/'technical/completion.json', dict(state='complete', historical_sources=150,
        dense_cpu_sources=int((source_frame.engine == 'CPU_2fs').sum()),
        dense_gpu_sources=int((source_frame.engine == 'GPU_2fs').sum()),
        completed_at=datetime.now(timezone.utc).isoformat(), simulations_modified=False))
    print(pair_frame.to_string(index=False))
    print(cohort_frame.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(load_json(parser.parse_args().config))
