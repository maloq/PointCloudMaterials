"""Counterfactual 50%-crystal termination using frozen Al duration observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.al_duration import first_confirmed
from src.experiment_runner.artifacts import result_folders, write_json
from src.experiment_runner.metric_docs import fingerprint, write_metric_rows
from src.project_runtime.paths import load_json

FAMILY = 'al_half_transform'


def analyse(root, config):
    source = Path(config['observations_from'])
    if (root / 'technical/summary.json').exists():
        raise FileExistsError(f'{root}: completed calculation exists; use a new revision')
    contract = json.loads((source / 'technical/metric-contract.json').read_text())
    inputs = {}
    for name in ('observations', 'sources', 'original_fine_crystallinity'):
        path = source / f'tables/{name}.csv'
        binding = json.loads((source / f'technical/table-contracts/{name}.json').read_text())
        checksum = fingerprint(path)
        if checksum != binding['sha256'] or binding['family'] != 'al_duration':
            raise ValueError(f'{path}: frozen input table binding changed')
        inputs[name] = dict(path=str(path), sha256=checksum)
    curves = pd.read_csv(inputs['observations']['path'])
    prior = pd.read_csv(inputs['sources']['path'])
    fine = pd.read_csv(inputs['original_fine_crystallinity']['path'])
    rows = []
    for (protocol, sid), frame in curves.groupby(['protocol', 'source_id'], sort=True):
        frame = frame.sort_values('time_ps')
        times = frame.time_ps.to_numpy()
        if not np.array_equal(times, np.arange(0, 601, 15)):
            raise ValueError(f'{protocol}/{sid}: exact 15-ps audit grid changed')
        t50 = first_confirmed(times, frame.crystal_fraction.to_numpy(), .5)
        original = prior[(prior.protocol == protocol) & (prior.source_id == sid)].iloc[0]
        saved_t50 = None if pd.isna(original.t50_ps) else float(original.t50_ps)
        if saved_t50 != t50:
            raise ValueError(f'{protocol}/{sid}: confirmed crossing differs from frozen audit')
        stop = min(600., t50 + 15 + config['prediction_tail_ps']) if t50 is not None else 600.
        following = frame[frame.time_ps >= (t50 if t50 is not None else 600)]
        observed_return = bool(t50 is not None and (following.crystal_fraction < .5).any())
        fine_t50 = None
        fine_return = None
        if protocol == 'original_3fs_075ps':
            finer = fine[fine.source_id == sid].sort_values('time_ps')
            if not np.array_equal(finer.time_ps.to_numpy(), np.arange(801) * .75):
                raise ValueError(f'{sid}: original fine timeline changed')
            fine_t50 = first_confirmed(finer.time_ps.to_numpy(), finer.crystal_fraction.to_numpy(), .5)
            fine_return = bool(t50 is not None and
                (finer[finer.time_ps >= t50].crystal_fraction < .5).any())
        row = dict(protocol=protocol, source_id=int(sid), temperature_K=float(original.temperature_K),
            role=original.role, root_lineage=original.root_lineage,
            t50_observed_ps=t50, confirmation_time_ps=t50 + 15 if t50 is not None else None,
            proposed_measurement_stop_ps=stop, proposed_total_dynamics_ps=stop + 15,
            saved_measurement_ps=600 - stop, early_stop=stop < 600,
            no_confirmed_half_crossing=t50 is None,
            later_observed_return_below_half=observed_return,
            original_fine_t50_ps=fine_t50, original_fine_return_below_half=fine_return)
        for cutoff in config['fixed_cutoffs_ps']:
            row[f'ready_to_stop_by_{cutoff}'] = stop <= cutoff
        rows.append(row)
    sources = pd.DataFrame(rows)
    summary = []
    for (protocol, temperature), group in sources.groupby(['protocol', 'temperature_K'], sort=True):
        events = group[group.t50_observed_ps.notna()]
        row = dict(protocol=protocol, temperature_K=float(temperature), source_count=len(group),
            confirmed_half_count=len(events), no_half_count=int(group.no_confirmed_half_crossing.sum()),
            half_time_conditional_median_ps=float(events.t50_observed_ps.median()),
            stop_time_conditional_median_ps=float(events.proposed_measurement_stop_ps.median()),
            latest_confirmed_half_ps=float(events.t50_observed_ps.max()),
            latest_event_stop_ps=float(events.proposed_measurement_stop_ps.max()),
            mean_stop_all_sources_ps=float(group.proposed_measurement_stop_ps.mean()),
            saved_fraction=float(group.saved_measurement_ps.sum() / (600 * len(group))),
            observed_returns_below_half=int(group.later_observed_return_below_half.sum()))
        for cutoff in config['fixed_cutoffs_ps']:
            row[f'ready_to_stop_by_{cutoff}_count'] = int(group[f'ready_to_stop_by_{cutoff}'].sum())
        summary.append(row)
    totals = []
    for protocol, group in sources.groupby('protocol', sort=True):
        totals.append(dict(protocol=protocol, source_count=len(group),
            early_stop_count=int(group.early_stop.sum()), no_half_count=int(group.no_confirmed_half_crossing.sum()),
            saved_measurement_ps=float(group.saved_measurement_ps.sum()),
            saved_measurement_fraction=float(group.saved_measurement_ps.sum() / (600 * len(group))),
            saved_integration_fraction_including_equilibration=float(group.saved_measurement_ps.sum() / (615 * len(group))),
            mean_measurement_stop_ps=float(group.proposed_measurement_stop_ps.mean()),
            later_observed_returns_below_half=int(group.later_observed_return_below_half.sum())))
    for name, data in (('sources', rows), ('temperature_summary', summary), ('totals', totals)):
        write_metric_rows(data, root, family=FAMILY, name=name)
    peer_summary, capped_sources = peer_caps(sources, config)
    write_metric_rows(peer_summary, root, family=FAMILY, name='peer_cutoffs')
    write_metric_rows(capped_sources, root, family=FAMILY, name='peer_capped_sources')
    write_json(root / 'technical/protocol.json', dict(config=config, input_tables=inputs,
        original_observation_contract=contract, source_protocol_sha256=fingerprint(source / 'technical/protocol.json'),
        policy='Counterfactual analysis only. Existing 600-ps jobs/data unchanged. 15-ps physical checks; one subsequent confirming check; 6-ps prediction tail.',
        coverage='All 150 historical ancestors; 21 completion-limited dense descendants at 520 K.'))
    result = dict(temperatures=summary, totals=totals, peer_cutoffs=peer_summary)
    write_json(root / 'technical/summary.json', result)
    plot(root, sources)
    return result


def peer_caps(sources, config):
    """Include all peers in the quorum, and preserve unobserved events as censored."""
    summary, rows = [], []
    for protocol, population in sources.groupby('protocol', sort=True):
        groups = [('temperature', float(t), group) for t, group in population.groupby('temperature_K', sort=True)]
        groups.append(('all_temperatures', None, population))
        for scope, temperature, group in groups:
            quorum = int(np.ceil(config['peer_fraction'] * len(group)))
            events = group[~group.no_confirmed_half_crossing].sort_values('proposed_measurement_stop_ps')
            cap_observed = len(events) >= quorum
            cap = float(events.iloc[quorum - 1].proposed_measurement_stop_ps) if cap_observed else 600.
            capped = np.minimum(group.proposed_measurement_stop_ps, cap)
            reached = ~group.no_confirmed_half_crossing & (group.proposed_measurement_stop_ps <= cap)
            summary.append(dict(protocol=protocol, scope=scope, temperature_K=temperature,
                source_count=len(group), required_half_count=quorum, peer_fraction=config['peer_fraction'],
                peer90_time_observed=cap_observed, peer90_stop_time_ps=cap if cap_observed else None,
                effective_maximum_ps=cap, half_ready_at_cutoff_count=int(reached.sum()),
                right_censored_at_cutoff_count=int((~reached).sum()),
                known_later_half_by600_count=int((~reached & ~group.no_confirmed_half_crossing).sum()),
                no_half_by600_count=int(group.no_confirmed_half_crossing.sum()),
                mean_stop_with_peer_cap_ps=float(capped.mean()),
                saved_measurement_fraction_with_peer_cap=float((600 - capped).sum() / (600 * len(group)))))
            for index, row in group.iterrows():
                rows.append(dict(protocol=protocol, scope=scope, peer_temperature_K=temperature,
                    source_id=int(row.source_id), temperature_K=float(row.temperature_K),
                    proposed_measurement_stop_ps=float(min(row.proposed_measurement_stop_ps, cap)),
                    right_censored_half_event=bool(not reached.loc[index]),
                    known_later_half_by600=bool(not reached.loc[index] and not row.no_confirmed_half_crossing),
                    no_confirmed_half_by600=bool(row.no_confirmed_half_crossing)))
    return summary, rows


def plot(root, sources):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    original = sources[sources.protocol == 'original_3fs_075ps']
    grid = np.arange(0, 601, 3)
    for temperature, group in original.groupby('temperature_K', sort=True):
        fractions = [(group.proposed_measurement_stop_ps <= t).mean() for t in grid]
        # Capped non-crossers are never successful half-crystal stops.
        fractions[-1] = group.early_stop.mean()
        axes[0].step(grid, fractions, where='post', label=f'{temperature:g} K')
    axes[0].set(xlabel='Measurement time (ps)', ylabel='Fraction ready for a confirmed 50%-crystal stop', ylim=(0, 1.03))
    axes[0].legend(fontsize=9)
    temperatures = [400, 450, 500, 510, 520]
    means = [original[original.temperature_K == t].proposed_measurement_stop_ps.mean() for t in temperatures]
    axes[1].bar([str(t) for t in temperatures], means, color='#386a99')
    axes[1].axhline(600, color='black', ls='--', lw=1, label='Current 600 ps')
    axes[1].set(xlabel='Temperature (K)', ylabel='Mean retained measurement time, including non-crossers (ps)', ylim=(0, 640))
    axes[1].legend()
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Stop after confirmed 50% crystallinity + 6 ps prediction tail\n15 ps checks; runs without a confirmed crossing keep the 600 ps limit')
    fig.savefig(root / 'plots/half_crystal_cutoff.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    config = load_json(args.config)
    print(json.dumps(analyse(result_folders(config['output']), config), indent=2))


if __name__ == '__main__':
    main()
