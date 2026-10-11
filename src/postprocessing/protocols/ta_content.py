"""Read-only crystal-content and exclusion audit of the completed Ta birth search."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols.shooting_data import read

FAMILY = 'ta_crystal_content'
PARENTS = ('model_1m', '2.7ns', '2.8ns', '2.9ns', '3.0ns', '3.60ns')


def strong_origins(graph):
    first = graph['frame'].copy()
    parents = defaultdict(list)
    for a, b, _, strong, _ in graph['edges']:
        if strong:
            parents[int(b)].append(int(a))
    for node in range(1, len(first)):
        if any(a >= node for a in parents[node]):
            raise ValueError('Temporal graph is not topologically ordered')
        first[node] = min([int(first[node])] + [int(first[a]) for a in parents[node]])
    return first


def run(c):
    check_metric_docs(family=FAMILY)
    root = resolve_path(c['output'])
    p = read(root / 'technical/plan.json')
    if (p['config'] != c or digest({k:v for k,v in p.items() if k != 'identity'}) != p['identity']
            or read(root / 'technical/complete.json')['identity'] != p['identity']):
        raise ValueError('Require the complete unchanged Ta search')
    frames, endpoints, evidence, sensitivity, filtered = [], [], [], [], []
    counts = Counter()
    for s in p['sources']:
        folder = root / 'technical/sources' / s['id']
        receipt = read(folder / 'complete.json')
        if receipt['identity'] != digest(dict(plan=p['identity'], source=s['id'])):
            raise ValueError('Changed Ta result identity')
        for name in ('graph.npz', 'events.json', 'transient-episodes.json'):
            if sha(folder / name) != receipt['files'][name]:
                raise ValueError(f'Changed Ta source evidence: {folder/name}')
        evidence.append(dict(source=s['id'], files={n:receipt['files'][n] for n in
            ('graph.npz','events.json','transient-episodes.json')}))
        with np.load(folder / 'graph.npz') as stored:
            g = {k:stored[k] for k in stored.files}
        series = g['crystalline_atoms']
        if series.shape != (len(s['times_ps']),) or np.any(series > s['atom_count']):
            raise ValueError('Crystalline count/timeline mismatch')
        common = dict(source=s['id'], parent=s['parent_id'], branch=s['branch_id'],
                      cohort='historical' if s['legacy'] else 'shooting', atoms=s['atom_count'])
        for k, time in enumerate(s['times_ps']):
            sizes = g['size'][g['frame'] == k]
            frames.append(dict(**common, time_ps=time, crystalline_atoms=int(series[k]),
                crystalline_percent=float(100*series[k]/s['atom_count']),
                largest_cluster_atoms=int(sizes.max()) if len(sizes) else 0))
        endpoints.append(dict(**common, start_percent=float(100*series[0]/s['atom_count']),
            end_percent=float(100*series[-1]/s['atom_count']),
            change_percentage_points=float(100*(series[-1]-series[0])/s['atom_count']),
            net_crystalline_atoms=int(series[-1]-series[0])))
        if s['legacy']:
            continue
        origin = strong_origins(g)
        for e in read(folder / 'events.json'):
            counts[e['kind']] += 1
            if e['kind'] == 'parent_present_embryo_growth':
                robust = bool(origin[e['birth_node']] == 0)
                counts['strong_parent_path' if robust else 'weak_link_dependent_parent_path'] += 1
                sensitivity.append(dict(**common, event=e['event_id'],
                    frozen_kind=e['kind'], spatial_establishment_kind=e['establishment_kind'],
                    birth_time_ps=e['birth_time_ps'], birth_atoms=e['birth_size'],
                    earliest_strong_ancestor_ps=s['times_ps'][int(origin[e['birth_node']])],
                    strong_path_to_start=robust))
        transient = read(folder / 'transient-episodes.json')
        stages = [('unestablished_episodes', lambda e: True),
                  ('peak_at_least_8', lambda e: e['peak_atoms'] >= 8),
                  ('verified_dissolved', lambda e: e['terminal_status'] == 'dissolved'),
                  ('isolated_origin', lambda e: e['origin'] == 'isolated'),
                  ('single_root', lambda e: e['root_count'] == 1),
                  ('strong_lineage', lambda e: e['strong_lineage']),
                  ('sustained_size', lambda e: e['primary_failed_candidate'])]
        for stage, keep in stages:
            transient = [e for e in transient if keep(e)]
            filtered.append(dict(**common, stage=stage, count=len(transient)))
    parent_table = []
    for parent in PARENTS:
        shots = [s for s in endpoints if s['parent'] == parent and s['cohort'] == 'shooting']
        if len(shots) != 4 or len({s['start_percent'] for s in shots}) != 1:
            raise ValueError('Expected four identical-position shooting starts')
        parent_table.append(dict(parent=parent, shots=len(shots), atoms=shots[0]['atoms'],
            start_percent=shots[0]['start_percent'], end_percent_mean=float(np.mean([s['end_percent'] for s in shots])),
            end_percent_min=min(s['end_percent'] for s in shots), end_percent_max=max(s['end_percent'] for s in shots),
            change_percentage_points=float(np.mean([s['change_percentage_points'] for s in shots])),
            net_crystalline_atoms_mean=float(np.mean([s['net_crystalline_atoms'] for s in shots]))))
    dest = root / 'analyses/crystal-content-v1'
    for name, values in [('time-series',frames), ('endpoints',endpoints), ('shooting-parent-summary',parent_table),
                         ('parent-ancestry-sensitivity',sensitivity), ('dissolution-filter',filtered),
                         ('shooting-event-counts',[dict(category=k,count=v) for k,v in sorted(counts.items())])]:
        write_metric_rows(values, dest, family=FAMILY, name=name)
    plot(frames, dest)
    total = Counter()
    for r in filtered: total[r['stage']] += r['count']
    lines = ['# Ta crystal content and scarcity of new births', '',
        'Reanalysis of the completed search; no new MD, PTM, relaxation or changed event labels.',
        'The main table covers 24 shooting branches, four from each parent, over24ps at1900K. Six earlier trajectories remain separate reference curves.',
        'Crystalline means the frozen PTM FCC/HCP/BCC labels at RMSD≤0.1. Percentages count all labeled atoms, including small clusters; they are not a thermodynamic phase fraction.', '',
        '| Parent | Initial crystal % | Final mean % | Final range % | Change (percentage points) |',
        '|---|---:|---:|---:|---:|']
    for r in parent_table:
        lines.append(f'| {r["parent"]} | {r["start_percent"]:.3f} | {r["end_percent_mean"]:.3f} | '
            f'{r["end_percent_min"]:.3f}–{r["end_percent_max"]:.3f} | {r["change_percentage_points"]:+.3f} |')
    lines += ['', '## Why the new birth count is low', '',
        f'The shooting branches have {counts["left_censored_existing"]} establishments already present at the initial observation, '
        f'{counts["parent_present_embryo_growth"]} later establishments connected to starting embryos, '
        f'{counts["interface_associated"]} interface-associated establishment, and '
        f'{counts["isolated_establishment"]} newly isolated establishments under the frozen rule.',
        'These are branch-local counts, including repeated observations of related structures across velocity replicas.', '',
        f'Of the parent-connected establishments, {counts["strong_parent_path"]} have a directed strong-overlap path to the start. '
        f'The other {counts["weak_link_dependent_parent_path"]} depend on at least one weak overlap. '
        'All eight weak-link-dependent cases have a spatially isolated establishment label before the parent-ancestry override. '
        'They are ambiguous candidates for closer inspection, not eight demonstrated independent births. '
        'Thus zero accepted new births must not be interpreted as proof that no new nucleation occurred.', '',
        'Sequential dissolution filters in the24 shooting branches:', '',
        '| Filter | Episodes remaining |', '|---|---:|']
    lines += [f'| {k} | {total[k]} |' for k in ('unestablished_episodes','peak_at_least_8','verified_dissolved',
                                                'isolated_origin','single_root','strong_lineage','sustained_size')]
    lines += ['', 'The four previously reported brief candidates combine three new shooting episodes with one earlier trajectory episode. '
        'The three new retained episodes peak at only eight atoms. No retained candidate meets sustained≥8 atoms for1ps.', '',
        'The observed crystal growth and pre-existing components support growth of inherited structures as a major signal in this collection. '
        'However, only24ps and0.5ps sampling, the≥64 atoms/1.5ps establishment threshold, conservative ancestry links, '
        'strict dissolution verification and PTM phase/RMSD choices all constrain what the audit can count. '
        'Coordinate quantization and classifier sensitivity were not re-evaluated here. No intrinsic Ta nucleation rate is inferred.', '',
        'The next targeted check is the eight weak-link-dependent establishments, using their saved atom-overlap histories and native0.1ps frames. '
        'That check has not been run by this summary, and the frozen harvest labels remain unchanged.', '',
        '![Crystalline fraction along all Ta trajectories](plots/crystalline-fraction.png)', '',
        '[Parent summary](tables/shooting-parent-summary.csv) · [Ambiguous lineage cases](tables/parent-ancestry-sensitivity.csv) · '
        '[Definitions](tables/METRICS.md)', '']
    (dest / 'README.md').write_text('\n'.join(lines))
    write_json(dest / 'technical/provenance.json', dict(plan=p['identity'], inputs=evidence,
        scope=dict(shooting=24,historical=6), frozen_labels_unchanged=True))
    print('\n'.join(lines[:16]), flush=True)
    print(dict(counts=counts, filters=total, report=str(dest / 'README.md')), flush=True)


def plot(frames, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    for parent, ax in zip(PARENTS, axes.flat):
        rows = [r for r in frames if r['parent'] == parent]
        for source in sorted({r['source'] for r in rows}):
            series = sorted([r for r in rows if r['source'] == source], key=lambda r:r['time_ps'])
            old = series[0]['cohort'] == 'historical'
            ax.plot([r['time_ps'] for r in series], [r['crystalline_percent'] for r in series],
                color='0.4' if old else '#0072B2', linestyle='--' if old else '-', alpha=.75,
                linewidth=1.2, label='Earlier trajectory' if old else 'Shooting branch')
        ax.set_title(parent); ax.set_xlabel('Time since branch start (ps)'); ax.set_ylabel('PTM crystalline atoms (%)')
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles)); fig.legend(unique.values(),unique.keys(),loc='lower center',ncol=2)
    fig.suptitle('Ta crystal content: four shooting branches per parent and one earlier reference\nPanel vertical scales differ')
    fig.tight_layout(rect=(0,.06,1,.91)); (dest / 'plots').mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'): fig.savefig(dest / 'plots' / f'crystalline-fraction.{suffix}',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    run(read(parser.parse_args().config))
