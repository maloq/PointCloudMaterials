"""Audit Al transformation duration without changing the simulation campaign."""
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
from src.simulation.campaigns.common import _read_thermodynamic_log

FAMILY = 'al_duration'
ATOMS = 70304


def checked_hash(path, expected):
    observed = fingerprint(path)
    if observed != expected:
        raise ValueError(f'{path}: expected SHA256 {expected}, observed {observed}')
    return observed


def historical(record, grid):
    directory = Path(record['parent_directory'])
    outcome_path = directory / 'outcome.json'
    checked_hash(outcome_path, record['parent_input_sha256']['outcome.json'])
    outcome = json.loads(outcome_path.read_text())
    inputs = {}
    for name, field in (('crystallization_progress.npz', 'progress_artifact'),
                        ('thermodynamics.npz', 'thermodynamics_artifact')):
        inputs[name] = checked_hash(directory / name, outcome[field]['sha256'])
    inputs['manifest.json'] = checked_hash(directory / 'trajectory_binary_float16/manifest.json',
                                          record['parent_manifest_sha256'])
    with np.load(directory / 'crystallization_progress.npz') as z:
        times = z['time_ps'].copy()
        if not np.array_equal(times, np.arange(801) * .75):
            raise ValueError(f'{directory}: historical timeline changed')
        if not np.array_equal(z['step'], np.arange(801) * 250):
            raise ValueError(f'{directory}: historical integration steps changed')
        if z['structure_names'].tolist() != ['other', 'fcc', 'hcp', 'bcc', 'ico']:
            raise ValueError(f'{directory}: historical structure names changed')
        fraction = z['crystalline_fraction'].astype(float)
        cluster = z['largest_crystalline_cluster_atoms'].astype(float) / ATOMS
        fractions = z['structure_fractions'].astype(float)
    with np.load(directory / 'thermodynamics.npz') as z:
        if not np.array_equal(z['step'], np.arange(801) * 250):
            raise ValueError(f'{directory}: thermodynamic timeline changed')
        energy = z['potential_energy_eV_per_atom'].copy()
        volume = z['volume_A3'].copy() / ATOMS
    indices = np.searchsorted(times, grid)
    if not np.array_equal(times[indices], grid):
        raise ValueError(f'{directory}: observation times are not exact')
    frame = pd.DataFrame(dict(time_ps=grid, crystal_fraction=fraction[indices],
                              largest_cluster_fraction=cluster[indices]))
    for column, values in (('energy_eV_per_atom', energy), ('volume_A3_per_atom', volume)):
        frame[column] = [values[(times > t - 15) & (times <= t)].mean() for t in grid]
    for i, name in enumerate(('other', 'fcc', 'hcp', 'bcc', 'ico')):
        frame[name + '_fraction'] = fractions[indices, i]
    return frame, inputs, pd.DataFrame(dict(time_ps=times, crystal_fraction=fraction))


def dense(record, directory, grid, reuse):
    trajectory = ShootingBinaryTrajectory.load(directory / 'trajectory_binary_float16')
    if trajectory.atom_count != ATOMS or not np.array_equal(trajectory.timesteps, np.arange(6001) * 50):
        raise ValueError(f'{directory}: exact 0.1-ps source contract changed')
    manifest_hash = fingerprint(trajectory.root / 'manifest.json')
    if record['source_id'] in reuse:
        saved = reuse[record['source_id']]
        if saved['new_manifest_sha256'] != manifest_hash:
            raise ValueError(f'{directory}: saved replay input changed')
        rows = []
        for row in saved['structure']:
            if row['time_ps'] in grid:
                rows.append(dict(time_ps=row['time_ps'],
                    crystal_fraction=row['crystal_fraction_new'],
                    largest_cluster_fraction=row['largest_crystal_cluster_new'] / ATOMS,
                    **{n + '_fraction': row[n + '_fraction_new']
                       for n in ('other', 'fcc', 'hcp', 'bcc', 'ico')}))
    else:
        os.environ['OVITO_THREAD_COUNT'] = '1'
        rows = []
        for t in grid:
            index = int(round(t * 10))
            if int(trajectory.timesteps[index]) * 2 != int(round(t * 1000)):
                raise ValueError(f'{directory}: dense time mismatch at {t}')
            _, values = structure(trajectory.positions[index].astype(float),
                                  (trajectory.box_high[index] - trajectory.box_low[index]).astype(float))
            cluster = values.pop('largest_crystal_cluster')
            rows.append(dict(time_ps=t, largest_cluster_fraction=cluster / ATOMS, **values))
            print(f'PTM source={record["source_id"]} time={t:g} ps', flush=True)
    frame = pd.DataFrame(rows).sort_values('time_ps').reset_index(drop=True)
    if not np.array_equal(frame.time_ps.to_numpy(), grid):
        raise ValueError(f'{directory}: dense structural grid incomplete')
    log = _read_thermodynamic_log(directory / 'measurement.lammps.log')
    steps = np.array(sorted(log))
    if not np.array_equal(steps, np.arange(6001) * 50):
        raise ValueError(f'{directory}: dense thermodynamics timeline incomplete')
    thermo = np.array([log[int(s)] for s in steps])
    times = steps * .002
    for column, values in (('energy_eV_per_atom', thermo[:, 3] / ATOMS),
                           ('volume_A3_per_atom', thermo[:, 2] / ATOMS)):
        frame[column] = [values[(times > t - 15) & (times <= t)].mean() for t in grid]
    inputs = dict(manifest_sha256=manifest_hash,
                  measurement_log_sha256=fingerprint(directory / 'measurement.lammps.log'),
                  structure_origin='saved paired replay' if record['source_id'] in reuse else 'fresh PTM',
                  directory=str(directory.resolve()))
    return frame, inputs


def first_confirmed(times, values, threshold):
    hits = (values[:-1] >= threshold) & (values[1:] >= threshold)
    indices = np.flatnonzero(hits)
    return float(times[indices[0]]) if len(indices) else None


def observables(frame):
    return frame[['crystal_fraction', 'energy_eV_per_atom', 'volume_A3_per_atom']].to_numpy()


def reference(frame):
    return observables(frame)[frame.time_ps.to_numpy() >= 540].mean(axis=0)


def deviation(values, ref, config):
    tolerance = np.array([config['future_fraction_tolerance'],
                          config['future_energy_tolerance_eV_per_atom'],
                          config['future_volume_relative_tolerance'] * ref[2]])
    return np.any(np.abs(values - ref) > tolerance, axis=1)


def settled_time(frame, config):
    """Retrospective state consistency, requiring at least 60 ps future evidence."""
    t = frame.time_ps.to_numpy()
    f = frame.crystal_fraction.to_numpy()
    connected = frame.largest_cluster_fraction.to_numpy() / np.maximum(f, 1 / ATOMS)
    bad = deviation(observables(frame), reference(frame), config)
    future_bad = np.maximum.accumulate(bad[::-1])[::-1]
    eligible = (f >= config['retrospective_crystal_floor']) & (
        connected >= config['connected_share_floor']) & ~future_bad & (t <= 540)
    indices = np.flatnonzero(eligible)
    return float(t[indices[0]]) if len(indices) else None


def hypothetical_stop(frame, floor, config):
    """A causal 60 ps plateau plus 60 ps retained stable tail; audit later change."""
    times = frame.time_ps.to_numpy()
    values = observables(frame)
    fraction = values[:, 0]
    connected = frame.largest_cluster_fraction.to_numpy() / np.maximum(fraction, 1 / ATOMS)
    for i, t in enumerate(times):
        if t < config['plateau_window_ps'] + config['stable_tail_ps'] or t >= 600:
            continue
        use = (times >= t - config['plateau_window_ps'] - config['stable_tail_ps']) & (times <= t)
        window = values[use]
        tolerances = np.array([config['plateau_fraction_range'],
                               config['plateau_energy_range_eV_per_atom'],
                               config['plateau_volume_relative_range'] * window[:, 2].mean()])
        if np.min(window[:, 0]) < floor or np.min(connected[use]) < config['connected_share_floor']:
            continue
        if np.any(np.ptp(window, axis=0) > tolerances):
            continue
        ref = values[(times >= t - config['stable_tail_ps']) & (times <= t)].mean(axis=0)
        future = values[i + 1:]
        return dict(stop_time_ps=float(t), saved_ps=float(600 - t),
                    later_change=bool(deviation(future, ref, config).any()),
                    later_fraction_increase=float(np.max(future[:, 0] - ref[0])),
                    later_fraction_deviation=float(np.max(np.abs(future[:, 0] - ref[0]))),
                    later_energy_deviation_meV=float(np.max(np.abs(future[:, 1] - ref[1])) * 1000))
    return dict(stop_time_ps=None, saved_ps=0., later_change=False,
                later_fraction_increase=None, later_fraction_deviation=None,
                later_energy_deviation_meV=None)


def source_summary(frame, record, protocol, config):
    times = frame.time_ps.to_numpy()
    f = frame.crystal_fraction.to_numpy()
    end = reference(frame)
    last = frame[frame.time_ps >= 540]
    summary = dict(source_id=record['source_id'], temperature_K=record['temperature_K'],
        role=record['split'], root_lineage=record['root_lineage'], protocol=protocol,
        structure_interval_ps=15., final_crystal_fraction=float(f[-1]),
        terminal_mean_crystal_fraction=float(end[0]),
        terminal_mean_other_fraction=float(last.other_fraction.mean()),
        terminal_mean_hcp_fraction=float(last.hcp_fraction.mean()),
        terminal_crystal_range=float(last.crystal_fraction.max() - last.crystal_fraction.min()),
        terminal_energy_range_meV=float(np.ptp(last.energy_eV_per_atom) * 1000),
        settled_time_ps=settled_time(frame, config))
    for threshold in config['crossing_thresholds']:
        summary[f't{round(threshold * 100)}_ps'] = first_confirmed(times, f, threshold)
    return summary


def report(root, config, histories):
    sources, cuts, stops, curves = [], [], [], []
    for history in histories:
        record, protocol = history['record'], history['protocol']
        frame = pd.DataFrame(history['observations'])
        summary = source_summary(frame, record, protocol, config)
        sources.append(summary)
        ref = reference(frame)
        for cut in config['cutoffs_ps']:
            point = frame[frame.time_ps == cut].iloc[0]
            future = frame[frame.time_ps > cut]
            cuts.append(dict(source_id=record['source_id'], protocol=protocol,
                temperature_K=record['temperature_K'], cutoff_ps=cut,
                crystal_fraction_at_cutoff=float(point.crystal_fraction),
                terminal_fraction_gain=float(ref[0] - point.crystal_fraction),
                maximum_later_fraction_gain=float(future.crystal_fraction.max() - point.crystal_fraction),
                terminal_energy_change_meV=float((ref[1] - point.energy_eV_per_atom) * 1000),
                confirmed_settled_by_cutoff=summary['settled_time_ps'] is not None and summary['settled_time_ps'] <= cut))
        for floor in config['causal_crystal_floors']:
            stops.append(dict(source_id=record['source_id'], protocol=protocol,
                temperature_K=record['temperature_K'], crystal_floor=floor,
                **hypothetical_stop(frame, floor, config)))
        for row in history['observations']:
            curves.append(dict(source_id=record['source_id'], protocol=protocol,
                               temperature_K=record['temperature_K'], **row))
    source_frame, cut_frame, stop_frame, curve_frame = map(pd.DataFrame, (sources, cuts, stops, curves))
    temperatures = []
    for (protocol, temperature), group in source_frame.groupby(['protocol', 'temperature_K'], sort=True):
        settled = group.settled_time_ps.dropna()
        row = dict(protocol=protocol, temperature_K=float(temperature), source_count=len(group),
            majority_count=int((group.terminal_mean_crystal_fraction >= .5).sum()),
            near90_count=int((group.terminal_mean_crystal_fraction >= .9).sum()),
            terminal_crystal_median=float(group.terminal_mean_crystal_fraction.median()),
            terminal_crystal_min=float(group.terminal_mean_crystal_fraction.min()),
            terminal_crystal_max=float(group.terminal_mean_crystal_fraction.max()),
            t50_confirmed_count=int(group.t50_ps.notna().sum()),
            t50_conditional_median_ps=float(group.t50_ps.median()),
            t50_latest_confirmed_ps=float(group.t50_ps.max()),
            settled_confirmed_count=len(settled),
            settled_conditional_median_ps=float(settled.median()) if len(settled) else None,
            still_moving_last60_count=int((group.terminal_crystal_range > config['plateau_fraction_range']).sum()))
        for cut in config['cutoffs_ps']:
            c = cut_frame[(cut_frame.protocol == protocol) & (cut_frame.temperature_K == temperature) & (cut_frame.cutoff_ps == cut)]
            row[f'settled_by_{cut}_count'] = int(c.confirmed_settled_by_cutoff.sum())
            row[f'gain5pp_after_{cut}_count'] = int((c.terminal_fraction_gain > .05).sum())
            row[f'terminal_gain_after_{cut}_median'] = float(c.terminal_fraction_gain.median())
        temperatures.append(row)
    causal = []
    for (protocol, floor), group in stop_frame.groupby(['protocol', 'crystal_floor'], sort=True):
        eligible = group[group.stop_time_ps.notna()]
        causal.append(dict(protocol=protocol, crystal_floor=float(floor), source_count=len(group),
            early_stops=len(eligible), later_change_count=int(eligible.later_change.sum()),
            saved_ps=float(group.saved_ps.sum()), saved_fraction=float(group.saved_ps.sum() / (len(group) * 600)),
            max_later_fraction_increase=float(eligible.later_fraction_increase.max()) if len(eligible) else None,
            max_later_energy_deviation_meV=float(eligible.later_energy_deviation_meV.max()) if len(eligible) else None))
    for name, rows in (('sources', sources), ('temperature_summary', temperatures), ('cutoffs', cuts),
                       ('hypothetical_stops', stops), ('stop_summary', causal), ('observations', curves)):
        write_metric_rows(rows, root, family=FAMILY, name=name)
    plot(root, source_frame, curve_frame, cut_frame)
    summary = dict(temperatures=temperatures, hypothetical_stops=causal,
                   note='Descriptive duration audit. No simulation durations, jobs, or training splits changed.')
    write_json(root / 'technical/summary.json', summary)
    return summary


def plot(root, sources, curves, cuts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex='row', sharey='row', constrained_layout=True)
    for j, temperature in enumerate((400, 450, 500, 510, 520)):
        old = curves[(curves.protocol == 'original_3fs_075ps') & (curves.temperature_K == temperature)]
        for _, group in old.groupby('source_id'):
            axes[0, j].plot(group.time_ps, group.crystal_fraction, color='#3366aa', alpha=.2, lw=.7)
        median = old.groupby('time_ps').crystal_fraction.median()
        axes[0, j].plot(median.index, median, color='#163c70', lw=2, label='Original median')
        new = curves[(curves.protocol == 'dense_2fs_010ps') & (curves.temperature_K == temperature)]
        if len(new):
            median = new.groupby('time_ps').crystal_fraction.median()
            axes[0, j].plot(median.index, median, color='#d36b21', lw=2, label='Dense median (21)')
            axes[0, j].legend(fontsize=8, loc='upper left')
        axes[0, j].set_title(f'{temperature} K — 30 original runs')
        axes[0, j].set_ylim(0, 1)
        old_sources = sources[(sources.protocol == 'original_3fs_075ps') & (sources.temperature_K == temperature)]
        times = np.arange(0, 601, 15)
        confirmed = np.array([(old_sources.settled_time_ps <= t).sum() for t in times]) / 30
        axes[1, j].step(times, confirmed, where='post', color='#246a45', lw=2)
        axes[1, j].set_xlabel('Measurement time (ps)')
        axes[1, j].set_ylim(0, 1.03)
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
            ax.axvline(300, color='gray', ls=':', lw=.8)
            ax.axvline(450, color='gray', ls=':', lw=.8)
    axes[0, 0].set_ylabel('PTM crystal fraction (FCC + HCP + BCC)')
    axes[1, 0].set_ylabel('Fraction of runs with confirmed bulk plateau')
    fig.suptitle('Al duration audit: substantial growth often continues after 300–450 ps\n'
                 'Plateau: ≥70% crystal, connected, within 3 percentage points / 5 meV / 0.5% volume of terminal state; ≥60 ps follow-up', fontsize=12)
    fig.savefig(root / 'plots/duration_by_temperature.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    old = cuts[cuts.protocol == 'original_3fs_075ps']
    for temperature in (400, 450, 500, 510, 520):
        group = old[old.temperature_K == temperature]
        result = group.groupby('cutoff_ps').terminal_fraction_gain.apply(lambda x: np.mean(x > .05))
        axes[0].plot(result.index, result, 'o-', label=f'{temperature} K')
    axes[0].set(xlabel='Hypothetical cutoff (ps)', ylabel='Fraction missing >5 percentage points of later growth', ylim=(0, 1.03))
    axes[0].legend(fontsize=8)
    old_sources = sources[sources.protocol == 'original_3fs_075ps']
    axes[1].boxplot([old_sources[old_sources.temperature_K == t].terminal_mean_crystal_fraction for t in (400, 450, 500, 510, 520)],
                    tick_labels=['400', '450', '500', '510', '520'], showmeans=True)
    axes[1].set(xlabel='Temperature (K)', ylabel='Mean crystal fraction during 540–600 ps', ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root / 'plots/cutoff_risk.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    config = load_json(args.config)
    root = result_folders(config['output'])
    if (root / 'technical/summary.json').exists():
        raise FileExistsError(f'{root}: completed audit exists; use a new revision')
    manifest_path = Path(config['campaign']) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = manifest['runs'][:manifest['main_source_count']]
    if len(records) != 150 or len({r['root_lineage'] for r in records}) != 150:
        raise ValueError('Main cohort must contain exactly 150 independent ancestors')
    if pd.Series([r['temperature_K'] for r in records]).value_counts().to_dict() != dict.fromkeys((400., 450., 500., 510., 520.), 30):
        raise ValueError('Main cohort temperature counts changed')
    dense_directories = {}
    for record in records:
        directory = Path(manifest['root']) / record['run_dir']
        outcome = directory / 'outcome.json'
        if outcome.is_file() and json.loads(outcome.read_text())['state'] == 'complete':
            dense_directories[record['source_id']] = directory
    if sorted(dense_directories) != config['completed_dense_source_ids']:
        raise ValueError(f'Completed dense coverage changed: {sorted(dense_directories)}; freeze a new config')
    replay = Path(config['saved_replay'])
    reuse = {int(p.stem): json.loads(p.read_text()) for p in sorted((replay / 'technical/pairs').glob('*.json'))}
    grid = np.arange(0, 601, 15, dtype=float)
    protocol = dict(captured_at=datetime.now(timezone.utc).isoformat(), config=config,
        campaign_manifest_sha256=fingerprint(manifest_path), records=records,
        saved_replay_contract=json.loads((replay / 'technical/metric-contract.json').read_text()),
        reused_pair_hashes={str(s): fingerprint(replay / 'technical/pairs' / f'{s}.json') for s in reuse},
        status='computing', analysis_inputs='geometry-derived physical diagnostics and logged thermodynamics; no fitted models',
        simulation_changes='none', dense_coverage='completion-limited 520 K subset')
    write_json(root / 'technical/protocol.json', protocol)
    histories = []
    fine_curves = []
    input_hashes = []
    for record in records:
        frame, hashes, fine = historical(record, grid)
        histories.append(dict(record=record, protocol='original_3fs_075ps', observations=frame.to_dict('records')))
        fine_curves.extend(dict(source_id=record['source_id'], temperature_K=record['temperature_K'], **r) for r in fine.to_dict('records'))
        input_hashes.append(dict(source_id=record['source_id'], historical=hashes))
        if record['source_id'] in dense_directories:
            frame, hashes = dense(record, dense_directories[record['source_id']], grid, reuse)
            histories.append(dict(record=record, protocol='dense_2fs_010ps', observations=frame.to_dict('records')))
            input_hashes[-1]['dense'] = hashes
        print(f'AUDITED source={record["source_id"]} temperature={record["temperature_K"]:g}', flush=True)
    write_json(root / 'technical/input_hashes.json', input_hashes)
    write_json(root / 'technical/histories.json', histories)
    write_metric_rows(fine_curves, root, family=FAMILY, name='original_fine_crystallinity')
    summary = report(root, config, histories)
    protocol['status'] = 'complete'
    write_json(root / 'technical/protocol.json', protocol)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
