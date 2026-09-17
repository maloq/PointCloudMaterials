"""Paired whole-source comparisons of completed physical-path fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.project_runtime.paths import load_json
from src.experiment_runner.metric_docs import write_metric_table
from src.data.predictive_memory.prepare import write_json
from .train import bootstrap_sources


def fit_names(modalities):
    if not modalities or len(set(modalities)) != len(modalities) or set(modalities)-{'x', 'xv'}:
        raise ValueError('Comparison modalities must be a nonempty unique selection of x and xv')
    names = [f'{modality}-H{history}' for modality in modalities for history in (0, 12, 48)]
    names += [f'{modality}-H48-repeat' for modality in modalities]
    return names


def compare(config, modalities=('x', 'xv')):
    root = Path(config['output'])
    names = fit_names(modalities)
    records, scores = {}, {}
    for name in names:
        technical = root/name/'technical'
        state = json.loads((technical/'status.json').read_text())
        if state['state'] != 'complete' or state['step'] != config['training']['steps']:
            raise RuntimeError(f'Comparison requires completed equal-budget fits: {name}: {state}')
        records[name] = torch.load(technical/'evaluation.pt', weights_only=True)
        scores[name] = json.loads((technical/'metrics.json').read_text())
    paired = {}
    for modality in modalities:
        for history in (12, 48):
            reference, candidate = records[f'{modality}-H0']['test']['rows'], records[f'{modality}-H{history}']['test']['rows']
            keys = lambda rows: [(r['source_id'], r['center_id'], r['anchor']) for r in rows]
            if keys(reference) != keys(candidate):
                raise ValueError('Memory comparisons must share exact source/center/anchor identities')
            differences = np.array([a['joint_nll']-b['joint_nll'] for a, b in zip(reference, candidate, strict=True)])
            paired[f'{modality}_H{history}_gain_over_snapshot'] = bootstrap_sources(differences,
                [r['source_id'] for r in candidate], config['bootstrap_draws'], config['seed'])
        reference, candidate = records[f'{modality}-H48-repeat']['test']['rows'], records[f'{modality}-H48']['test']['rows']
        if keys(reference) != keys(candidate):
            raise ValueError('Repeated-anchor control must use identical evaluation tuples')
        paired[f'{modality}_H48_gain_over_repeated_anchor'] = bootstrap_sources(
            np.array([a['joint_nll']-b['joint_nll'] for a, b in zip(reference, candidate, strict=True)]),
            [r['source_id'] for r in candidate], config['bootstrap_draws'], config['seed'])
    destination = root/'comparison'
    (destination/'technical').mkdir(parents=True, exist_ok=True)
    metrics = dict(models=scores, paired_test=paired)
    write_json(destination/'technical'/'metrics.json', metrics)
    write_metric_table(metrics, destination, family='predictive_memory')
    lines = ['# Exploratory predictive-memory pilot', '',
        f"All fits use the same {config['training']['steps']:,}-update budget and one training seed. Lower physical-path NLL is better.", '',
        '| Model | Selected update | Test joint NLL | Test future MSE |', '|---|---:|---:|---:|']
    for name, result in scores.items():
        lines.append(f"| {name} | {result['selected_step']} | {result['test']['joint_nll']['mean']:.5f} | {result['test']['future_mse']['mean']:.5f} |")
    lines += ['', 'Paired NLL gains: positive values favor real history. Intervals resample whole sources.', '',
              '| Comparison | Gain | 95% source interval |', '|---|---:|---|']
    for name, result in paired.items():
        low, high = result['ci95']
        lines.append(f"| {name} | {result['mean']:.5f} | [{low:.5f}, {high:.5f}] |")
    lines += ['', 'These are previously examined sources, one center per source and three adjacent anchors. '
        'The intervals do not cover training-seed variability. Existing full-box float16 observations '
        'cannot distinguish physical memory from quantization-noise averaging. No compression, sufficiency, '
        'kinetic-closure or confirmatory crystallization claim follows from these fits.']
    (destination/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (destination/'plots').mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for modality in modalities:
        offset = {'x': -.6, 'xv': .6}[modality] if len(modalities) > 1 else 0.
        estimates = [paired[f'{modality}_H{h}_gain_over_snapshot'] for h in (12, 48)]
        values = np.array([r['mean'] for r in estimates])
        interval = np.array([r['ci95'] for r in estimates]).T
        ax.errorbar(np.array([12, 48])+offset, values, yerr=np.maximum(0, np.stack((values-interval[0], interval[1]-values))),
                    marker='o', capsize=4, label=modality)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set(xlabel='Observed history (ps)', ylabel='Test NLL gain over matched snapshot', title='Exploratory physical-path prediction')
    ax.legend(); fig.tight_layout(); fig.savefig(destination/'plots'/'history_gain.png', dpi=160); plt.close(fig)
    write_json(destination/'technical'/'status.json', dict(state='complete', fits=len(names)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--modalities', nargs='+', choices=('x', 'xv'), default=['x', 'xv'])
    args = parser.parse_args()
    compare(load_json(args.config), modalities=args.modalities)
