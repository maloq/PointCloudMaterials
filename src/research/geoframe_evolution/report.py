"""Collect measured epochs without treating checkpoints as independent repeats."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.experiment_runner.metric_docs import write_metric_table
from .reference import write_json


def lookup(value, path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


PANELS = [
    ('boundary_ap', ('context', 'ap', '2'), 'Boundary AP ↑'),
    ('fault_ap', ('context', 'ap', '3'), 'Al planar-fault AP ↑'),
    ('spatial_auc', ('nonbulk_spatial', 'auc'), 'Nonbulk boundary-aware AUROC ↑'),
    ('liquid_r2', ('liquid_order', 'mean_r2'), 'Liquid bond-order R² ↑'),
    ('continuity', ('continuity', '0.0001', 'normalized_p95'), '1e-4 Å perturbation / pair RMS ↓'),
    ('liquid_rank', ('liquid_collapse', 'effective_rank'), 'Liquid participation rank'),
]


def collect(output):
    root = Path(output); records = {}; raw = {}
    for path in sorted((root/'technical/evaluations').glob('*/metrics.json')):
        name = path.parent.name
        raw[name] = json.loads(path.read_text())
        records[name] = {}
        for rep in ('encoder', 'projector'):
            for material in ('Al', 'Ta', 'Zr'):
                rows = [r for key,r in raw[name].items() if key.startswith('frame_') and key.endswith('_'+material+'_'+rep)]
                if not rows:
                    continue
                series = {}
                for short, field, _ in PANELS:
                    values = [lookup(r, field) for r in rows]
                    values = [v for v in values if v is not None]
                    series[short] = float(np.mean(values)) if values else None
                    series[short+'_defined_frames'] = len(values)
                records[name][material+'_'+rep] = series
            prediction = raw[name]['prediction_'+rep]
            records[name]['prediction_'+rep] = dict(
                residual_mse=prediction['future_residual_9ps']['groups']['all']['mse'],
                hazard_brier=prediction['conditional_hazard']['horizons']['12.0']['brier'],
                hazard_ap=prediction['conditional_hazard']['horizons']['12.0']['average_precision'],
                temporal_response_auc=prediction['temporal_response']['domains']['both_PTM_other']['response_auc'])
    if not records:
        return
    epochs = sorted([s for s in records if s.startswith('epoch-')], key=lambda s:int(s.split('-')[1]))
    for rep in ('encoder', 'projector'):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, (short, _, title) in zip(axes.flat, PANELS):
            for color, material in zip(['#1971c2', '#e67700', '#7048e8'], ['Al','Ta','Zr']):
                pairs = [(int(s.split('-')[1]), records[s][material+'_'+rep][short]) for s in epochs]
                pairs = [(x,y) for x,y in pairs if y is not None]
                if pairs:
                    x,y = zip(*pairs); ax.plot(x,y,'.-',color=color,label=material)
                old = records.get('archived-epoch34', {}).get(material+'_'+rep, {}).get(short)
                if old is not None:
                    ax.scatter([34], [old], color=color, marker='*', s=130)
            ax.axvline(5, color='.7', ls=':', lw=1); ax.axvline(10, color='.7', ls=':', lw=1)
            ax.set(title=title, xlabel='Completed epoch index (0 = first pass)')
            ax.grid(alpha=.15)
        axes.flat[0].legend(); fig.suptitle(rep+' — per-material frame means; stars: archived epoch 34')
        fig.savefig(root/'plots'/f'{rep}-training-curves.png',dpi=170); plt.close(fig)
    correlations = {}
    for rep in ('encoder','projector'):
        correlations[rep] = {}
        for short,_,_ in PANELS:
            for target in ('residual_mse','hazard_brier','hazard_ap','temporal_response_auc'):
                pairs = [(records[s]['Al_'+rep][short],records[s]['prediction_'+rep][target]) for s in epochs]
                pairs = [(x,y) for x,y in pairs if x is not None and y is not None]
                value = None
                if len(pairs)>=5:
                    x,y = np.array(pairs).T
                    if np.std(x)>0 and np.std(y)>0:
                        value = float(spearmanr(x,y).statistic)
                correlations[rep][short+'__'+target] = dict(spearman=value, checkpoints=len(pairs))
    write_json(root/'technical/epoch-summary.json', records)
    write_json(root/'technical/metric-prediction-associations.json', correlations)
    write_metric_table(dict(checkpoints=records, descriptive_correlations=correlations),root,
                       family='geoframe_evolution',name='checkpoint-summary')
    trained = len(list((root/'technical/training').glob('epoch-*.json')))
    lines = ['# GeoFrame epoch-34 recipe: training evolution','',
        f'Training checkpoints retained: **{trained}/35**. Fresh epochs evaluated: **{len(epochs)}/35**.',
        f'Archived reference evaluated: **{"yes" if "archived-epoch34" in records else "pending"}**.', '',
        '[Plots and gallery](index.html) · [Checkpoint summary](tables/checkpoint-summary.csv) · [Metric definitions](tables/METRICS.md)', '',
        'This is one fresh training seed with the original 160-epoch LR clock, stopped after 35 passes. '
        'Classical labels distinguish crystal interior, solid–liquid boundary, Al planar faults and liquid order candidates. '
        'The static probe split has a spatial exclusion gap; encoder training includes these snapshots.', '',
        'Ta/Zr ordered-liquid detections are structural candidates. They do not establish future nuclei. '
        'The Al prediction assay reuses 45 ancestry-disjoint roots, with readouts fitted on 25 roots, selected on 5 and evaluated on 15.', '',
        'Checkpoint correlations with prediction are descriptive; there is no independent-repeat p-value. '
        'A smooth/collapsed representation cannot pass the rank, physical-fidelity and boundary-resolution checks together.', '',
        'Context IDs: 0 unclassified liquid; 1 crystal interior; 2 solid–liquid boundary; 3 Al planar fault; '
        '4 non-template crystal interior; 5 five-fold liquid candidate; 6 ordered liquid candidate.', '']
    (root/'README.md').write_text('\n'.join(lines))
    files = sorted((root/'plots').glob('*.png'))
    cards = ''.join(f'<figure><a href="plots/{p.name}"><img loading="lazy" src="plots/{p.name}" style="width:100%"></a><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in files)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GeoFrame training evolution</title>'
        '<style>body{font-family:system-ui;max-width:1400px;margin:2rem auto}main{display:grid;grid-template-columns:repeat(2,1fr)}figure{margin:1rem}figcaption{font-size:14px}</style>'
        f'<h1>GeoFrame training evolution</h1><p>{trained}/35 epochs trained; {len(epochs)}/35 evaluated. '
        'Static candidates are not verified nuclei. <a href="README.md">Protocol and status</a>.</p><main>'+cards+'</main>')


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    collect(p.parse_args().output)
