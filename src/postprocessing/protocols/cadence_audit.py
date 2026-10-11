"""Assess the original Al 0.75-ps birth campaign from frozen search evidence."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.experiment_runner.artifacts import file_hash, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import load_json, resolve_path

FAMILY = 'al075_birth_quality'


def run(config):
    output = resolve_path(config['output'])
    catalogue = resolve_path(config['catalogue'])
    plan_path = resolve_path(config['source_plan'])
    plan = json.loads(plan_path.read_text())
    sources = {s['id']: s for s in plan['sources']}
    receipts = {str(plan_path): file_hash(plan_path)}

    def table(name):
        binding_path = catalogue/'technical/table-contracts'/f'{name}.json'
        binding = json.loads(binding_path.read_text())
        path = catalogue/binding['table']
        if file_hash(path) != binding['sha256']:
            raise ValueError(f'Changed original table: {path}')
        for item in (path, binding_path, catalogue/binding['contract'], catalogue/binding['definitions']):
            receipts[str(item)] = file_hash(item)
        return pd.read_csv(path, dtype={'cadence': str, 'left_cadence': str, 'right_cadence': str})

    growth = table('growth-audit')
    task_counts = table('source-results')
    comparison = table('paired-comparisons')
    matched = table('matched-events')
    old = {k: s for k, s in sources.items() if s['origin'] == 'historical_fixed_cohort'}
    if len(old) != 150 or len({s['ancestry_group'] for s in old.values()}) != 150:
        raise ValueError('Unexpected original Al campaign population')
    inventory = []
    for sid, s in old.items():
        if s['native_cadence_ps'] != .75 or len(s['parts']) != 1:
            raise ValueError(f'Original source is not native 0.75 ps: {sid}')
        part, = s['parts']; root = Path(part['path'])
        if file_hash(root/'manifest.json') != part['manifest_sha256']:
            raise ValueError(f'Original observation manifest changed: {sid}')
        manifest = json.loads((root/'manifest.json').read_text())
        steps = np.load(root/manifest['arrays']['timesteps']['file'], allow_pickle=False)
        dt = np.diff(steps)*part['timestep_fs']/1000
        if len(steps) != 801 or not np.allclose(dt, .75, rtol=0, atol=1e-12):
            raise ValueError(f'Unexpected actual timeline: {sid}')
        deck_path = root.parent/'source.in.lammps'; deck = deck_path.read_text()
        equil = re.findall(r'^run (\d+)\s*\nreset_timestep 0', deck, re.M)
        if len(equil) != 1:
            raise ValueError(f'Unknown initial observation gap: {deck_path}')
        receipts[str(deck_path)] = file_hash(deck_path)
        velocities = 'velocities' in manifest['arrays'] and (root/manifest['arrays']['velocities']['file']).is_file()
        inventory.append(dict(source=sid, ancestor=s['ancestry_group'], role=s['role'],
            temperature_K=s['temperature_K'], frames=len(steps),
            duration_ps=float((steps[-1]-steps[0])*part['timestep_fs']/1000),
            initial_unobserved_ps=int(equil[0])*part['timestep_fs']/1000,
            has_saved_velocities=velocities, native_sampling_ps=.75,
            observation_manifest=str(root/'manifest.json'), observation_manifest_sha256=part['manifest_sha256']))
    original = growth[growth.source.isin(old) & (growth.cadence == '075')].copy()
    if len(original) != 600 or original[['source', 'detector']].duplicated().any():
        raise ValueError('Original campaign does not have one result per source/detector')
    original['temperature_K'] = original.source.map(lambda sid: old[sid]['temperature_K'])
    ptm = original[original.detector == 'ptm']
    detector_rows = []
    for detector, group in original.groupby('detector'):
        detector_rows.append(dict(detector=detector, sources=len(group),
            embryo_appearances=int(group.liquid_emergences.sum()),
            established_liquid_origins=int(group.established_liquid_emergences.sum()),
            sources_with_established_origin=int((group.established_liquid_emergences > 0).sum()),
            initial_embryo_records=int(group.initially_present_records.sum()),
            initially_embryonic_sources=int((group.initial_largest_cluster >= 8).sum()),
            initially_established_sources=int((group.initial_largest_cluster >= 64).sum()),
            established_composites=int(group.established_composites_from_subcritical.sum())))
    temperatures = []
    for t, group in ptm.groupby('temperature_K'):
        temperatures.append(dict(temperature_K=t, sources=len(group),
            sources_with_established_birth=int((group.established_liquid_emergences > 0).sum()),
            established_births=int(group.established_liquid_emergences.sum()),
            embryo_appearances=int(group.liquid_emergences.sum()),
            initially_embryonic_sources=int((group.initial_largest_cluster >= 8).sum())))

    # Compare the same observed trajectories, not different native-0.75 and
    # native-0.10 MD runs. Exclude historical partial copies from paired counts.
    paired_groups = {
        'uniform_birth_22': [s['id'] for s in sources.values() if
            s['dataset'].startswith('al-birth-uniform-20260926-source') and '075' in s['views']],
        'dense_001_complete_2': [s['id'] for s in sources.values() if
            s['native_cadence_ps'] == .01 and s['origin'] == 'original' and '075' in s['views']],
        'memory_0075_3': [s['id'] for s in sources.values() if
            s['native_cadence_ps'] == .075 and s['origin'] == 'original' and '075' in s['views']],
    }
    if [len(paired_groups[k]) for k in paired_groups] != [22, 2, 3]:
        raise ValueError('Matched sensitivity populations changed')
    paired = []; paired_sources = []
    for name, ids in paired_groups.items():
        group = growth[growth.source.isin(ids) & (growth.detector == 'ptm')]
        for cadence, rows in group.groupby('cadence'):
            if len(rows) != len(ids):
                raise ValueError(f'Missing matched cadence rows: {name}/{cadence}')
            paired.append(dict(population=name, sources=len(rows),
                ancestors=len({sources[s]['ancestry_group'] for s in ids}),
                cadence_ps=plan['config']['cadences'][cadence],
                embryo_appearances=int(rows.liquid_emergences.sum()),
                established_liquid_origins=int(rows.established_liquid_emergences.sum())))
            paired_sources.extend(dict(population=name, **r) for r in rows.to_dict('records'))
    pairs = comparison[comparison.source.isin(old) & (comparison.left_cadence == '075') &
        (comparison.left_detector == 'ptm') & (comparison.right_detector == 'coherent')]
    mm = matched[matched.source.isin(old) & (matched.left_cadence == '075') &
        (matched.left_detector == 'ptm') & (matched.right_detector == 'coherent')]
    agreement = dict(scope='All root appearances, including initial/uncertain origins; not established-only',
        ptm_root_records=int(pairs.left_events.sum()), bond_order_root_records=int(pairs.right_events.sum()),
        matched=int(pairs.matched.sum()), matched_origin_agreement=float(mm.origin_agrees.mean()),
        matched_fate_agreement=float(mm.fate_agrees.mean()),
        matched_median_absolute_onset_difference_ps=float(mm.onset_shift_ps.abs().median()))
    if agreement['matched'] != len(mm):
        raise ValueError('Matched-event and source-level comparison totals disagree')
    historical = resolve_path(config['historical_prediction_coverage'])/'tables/split-coverage.csv'
    receipts[str(historical)] = file_hash(historical)
    coverage = pd.read_csv(historical)
    incomplete = [s for s in old if not (resolve_path(config['catalogue']).parents[1]/
        'technical/sources'/s/'075/ptm/events.json').is_file()]
    summary = dict(original_sources=len(old), total_observed_ps=sum(r['duration_ps'] for r in inventory),
        sources_with_saved_velocities=sum(r['has_saved_velocities'] for r in inventory),
        detector_rows=detector_rows, temperatures=temperatures, matched_cadence=paired,
        detector_agreement=agreement, historical_strict_prediction_coverage=coverage.to_dict('records'),
        missing_original_v3_event_shards=len(incomplete),
        limitations='Counts are operational lineage evidence, not ground-truth precision/recall. Missing v3 per-event 0.75 shards prevent a new established-only event match without regeneration. Original MD and exported source summaries remain available.')
    for name, rows in [('sources', inventory), ('detectors', detector_rows), ('temperatures', temperatures),
                       ('matched-cadences', paired), ('matched-cadence-sources', paired_sources),
                       ('source-detectors', original.to_dict('records')),
                       ('historical-prediction-coverage', coverage.to_dict('records'))]:
        write_metric_rows(rows, output, family=FAMILY, name=name)
    write_json(output/'technical/input-receipts.json', receipts)
    write_json(output/'technical/summary.json', summary)
    write_json(output/'technical/config.json', config)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = output/'plots'; plot.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    ax[0].bar([str(int(r['temperature_K'])) for r in temperatures], [r['established_births'] for r in temperatures], color='#286f9c')
    ax[0].set(xlabel='Temperature (K; audit metadata)', ylabel='Established liquid-origin lineages', title='Original 150 trajectories')
    uniform = sorted((r for r in paired if r['population']=='uniform_birth_22'), key=lambda r:r['cadence_ps'])
    for a, field, title in [(ax[1], 'embryo_appearances', 'Embryo appearances'), (ax[2], 'established_liquid_origins', 'Established lineages')]:
        values = [r[field] for r in uniform]
        a.bar([str(r['cadence_ps']) for r in uniform], values, color=['#329f85', '#dd8d45'])
        a.set(xlabel='Observation interval (ps)', ylabel='Detected records', title=title)
        a.set_ylim(0, max(values)*1.18)
        for i, value in enumerate(values): a.text(i, value, str(value), ha='center', va='bottom')
    fig.suptitle('0.75-ps Al birth audit — right panels reuse the same 22 trajectories')
    fig.savefig(plot/'campaign-and-cadence.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); run(load_json(args.config))


if __name__ == '__main__':
    main()
