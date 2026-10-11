"""Publish timings, actual convergence and changes in structural observables."""
import json
from pathlib import Path
import numpy as np
from src.experiment_runner.artifacts import write_json
from src.experiment_runner.metric_docs import write_metric_rows


def report(c, p):
    root = Path(c['output'])
    rows = [json.loads(f.read_text()) for f in sorted((root/'technical/cases').glob('*/*.json'))]
    lookup = {(r['case'], r['profile']): r for r in rows}
    archive = Path(c['archive'])
    for row in rows:
        base = lookup.get((row['case'], 'sorted_fire'))
        if base is None or 'solver_seconds' not in row or 'solver_seconds' not in base:
            continue
        row['baseline_over_candidate_seconds'] = base['solver_seconds']/row['solver_seconds']
        row['baseline_over_candidate_force_calls'] = base['force_calls']/row['force_calls']
        row['energy_change_meV_atom'] = (row['energy_eV']-base['energy_eV'])*1000/row['atoms']
        a = np.load(archive/row['case']/'sorted_fire/observables.npz')
        b = np.load(archive/row['case']/row['profile']/'observables.npz')
        row['local_coordinate_component_rms_A'] = float(np.sqrt(np.mean((a['patch']-b['patch'])**2)))
        row['embedding_component_rms'] = float(np.sqrt(np.mean((a['embedding']-b['embedding'])**2)))
        row['ptm_label_disagreement_fraction'] = float(np.mean(a['labels'] != b['labels']))
    out = root/'analyses/overview'
    if rows:
        columns = list(dict.fromkeys(k for r in rows for k in r))
        write_metric_rows(rows, out, family='ni_relaxation_speed', name='cases', columns=columns)
    summary = []
    for profile in c['profiles']:
        group = [r for r in rows if r['profile'] == profile['name']]
        valid = [r for r in group if r.get('converged') and r.get('cpu_reference_tolerance_pass')]
        paired = [r for r in valid if 'baseline_over_candidate_seconds' in r
                  and lookup[r['case'], 'sorted_fire'].get('converged')]
        record = dict(profile=profile['name'], attempted=len(group), converged=len(valid),
                      matched_converged=len(paired), tolerance_eV_A=profile['tolerance'])
        for key in ['solver_seconds', 'force_calls', 'callback_seconds', 'publication_seconds']:
            values = [r[key] for r in valid]
            record['median_'+key] = float(np.median(values)) if values else None
        for key in ['baseline_over_candidate_seconds', 'baseline_over_candidate_force_calls',
                    'local_coordinate_component_rms_A', 'embedding_component_rms',
                    'ptm_label_disagreement_fraction', 'energy_change_meV_atom']:
            values = [r[key] for r in paired]
            record['median_'+key] = float(np.median(values)) if values else None
        summary.append(record)
    write_metric_rows(summary, out, family='ni_relaxation_speed', name='summary', columns=list(summary[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    labels = [r['profile'].replace('_', ' ') for r in summary]
    values = [r['median_baseline_over_candidate_seconds'] or 0 for r in summary]
    ax[0].barh(labels, values, color='#227c9d')
    ax[0].axvline(1, color='black', linewidth=1)
    ax[0].set_xlabel('Sorted FIRE time / candidate time (matched converged cases)')
    for i, r in enumerate(summary):
        ax[0].text(values[i]+.02, i, f"{r['converged']}/{r['attempted']} converged", va='center', fontsize=8)
    for r in summary:
        subset = [v for v in rows if v['profile'] == r['profile'] and 'embedding_component_rms' in v and v.get('converged')]
        if subset:
            ax[1].scatter([v['baseline_over_candidate_seconds'] for v in subset],
                [v['embedding_component_rms'] for v in subset], label=r['profile'], alpha=.65)
    ax[1].set_xlabel('Speed ratio relative to sorted FIRE')
    ax[1].set_ylabel('Embedding change RMS relative to sorted FIRE')
    ax[1].legend(fontsize=7)
    fig.suptitle('Ni relaxation: speed, convergence and changes in final structure')
    plots = out/'plots'; plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots/'speed-and-structure.png', dpi=180)
    fig.savefig(plots/'speed-and-structure.pdf')
    plt.close(fig)
    write_json(out/'technical/summary.json', dict(identity=p['identity'], profiles=summary))
    print(json.dumps(summary, indent=2), flush=True)
