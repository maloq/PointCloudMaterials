"""CPU-only, dated audit of the live continuation's repository-owned logs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/mace_bf16_training_20260908/runs/continued'
PREVIOUS = ROOT / 'output/mace_throughput_20260908/runs/optimized'
OUT = ROOT / 'output/mace_bf16_training_20260908/training_review'


def read_json(path):
    return json.loads(path.read_text())


def train_rows(path):
    # Only complete lines: the producer may currently be appending its next row.
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(line[6:]) for line in lines
            if line.startswith('TRAIN ') and line.endswith('\n')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Replay a previously captured inputs.json')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snapshot = read_json(args.snapshot)
    else:
        snapshot = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                        config=read_json(ROOT / 'experiments/mace_bf16_training_20260908/training.json'),
                        initial=read_json(RUN / 'initial_validation.json'),
                        resume=read_json(RUN / 'resume_validation.json'),
                        epochs=[json.loads(s) for s in (RUN / 'training.jsonl').read_text().splitlines(keepends=True) if s.endswith('\n')],
                        train=train_rows(RUN / 'run.log'), previous=train_rows(PREVIOUS / 'run.log'),
                        sources=[str(RUN.relative_to(ROOT)), str(PREVIOUS.relative_to(ROOT))])
    (OUT / 'inputs.json').write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
    cfg = snapshot['config']
    rows = [dict(step=r['step'], loss=r['loss'], gradient_norm=r['gradient_norm'],
                 backbone_lr=r['learning_rates_used'][0], head_lr=r['learning_rates_used'][1],
                 **r['parts']) for r in snapshot['train']]
    frame = pd.DataFrame(rows).set_index('step')
    frame.to_csv(OUT / 'training.csv')
    weights = dict(spatial_mse=25*cfg['spatial_weight'], temporal_mse=25*cfg['temporal_weight'],
                   tda_mse=cfg['tda_weight'], forecast_mse=cfg['prediction_weight'],
                   future_tda_mse=cfg['joint_objective']['future_tda_weight'],
                   nuisance_mse=cfg['joint_objective']['nuisance_weight'])
    weighted = frame[list(weights)].mul(pd.Series(weights))
    weighted['weighted_spread_reconstructed'] = frame.loss - weighted.sum(axis=1)
    weighted.to_csv(OUT / 'weighted_losses.csv')
    stages = [(0, 'initial', snapshot['initial']), (500, 'resume', snapshot['resume'])]
    stages += [(r['steps'], f"epoch_{r['epoch']}", r['validation']) for r in snapshot['epochs']]
    val_rows = [dict(step=step, stage=stage, material=material, **values)
                for step, stage, val in stages for material, values in val['by_material'].items()]
    validation = pd.DataFrame(val_rows)
    validation.to_csv(OUT / 'validation_by_material.csv', index=False)
    stats = {}
    for key in ['loss', 'gradient_norm', 'spatial_mse', 'temporal_mse', 'tda_mse',
                'forecast_mse', 'future_tda_mse', 'nuisance_mse', 'spread']:
        v = frame[key].to_numpy()
        stats[key] = dict(median=float(np.median(v)), p05=float(np.quantile(v, .05)),
                          p95=float(np.quantile(v, .95)), min=float(v.min()), max=float(v.max()))
    diagnostics = {}
    for key in frame.columns:
        if key.startswith(('task_gradient_norm_', 'gradient_cosine_')) or key in ('gradient_conflict_fraction', 'pcgrad_applied'):
            values = frame[key].dropna()
            diagnostics[key] = dict(count=len(values), median=float(values.median()),
                                    negative_fraction=float((values < 0).mean()))
    overlap = []
    previous = {r['step']: r for r in snapshot['previous']}
    for r in snapshot['train']:
        if r['step'] in previous:
            old = previous[r['step']]
            overlap.append(dict(step=r['step'], fp32_loss=old['loss'], bf16_loss=r['loss'],
                                loss_delta=r['loss']-old['loss'],
                                tda_delta=r['parts']['tda_mse']-old['parts']['tda_mse']))
    last_step, last_stage, last = stages[-1]
    result = dict(captured_utc=snapshot['captured_utc'], first_logged_step=int(frame.index.min()),
                  last_logged_step=int(frame.index.max()), logged_batches=len(frame),
                  last_validation_step=last_step, batch_statistics=stats, gradient_diagnostics=diagnostics,
                  logged_clip_fraction=float((frame.gradient_norm > cfg['gradient_clip']).mean()),
                  median_clip_multiplier=float((cfg['gradient_clip']/frame.gradient_norm).clip(upper=1).median()),
                  fp32_bf16_overlap=overlap,
                  latest_weighted_validation={k: last[k]*v for k, v in weights.items()})
    result['latest_weighted_validation']['weighted_spread_reconstructed'] = last['loss'] - sum(result['latest_weighted_validation'].values())
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax, key in zip(axes.flat, ['loss', 'tda_mse', 'spatial_mse', 'temporal_mse', 'future_tda_mse', 'nuisance_mse']):
        ax.plot(frame.index, frame[key], alpha=.3, linewidth=1, label='Single logged batch')
        ax.plot(frame.index, frame[key].rolling(11, min_periods=1).median(), label='Trailing 11-point median')
        ax.set(title=key, xlabel='Optimizer step'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('BF16 continuation: batch fluctuations and underlying trend')
    fig.savefig(OUT / 'training_curves.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, key in zip(axes.flat, ['tda_mse', 'future_tda_mse', 'spatial_ratio', 'temporal_ratio']):
        for material, group in validation.groupby('material', sort=False):
            y = group[key].to_numpy()/group[key].iloc[0]
            ax.plot(group.step, y, marker='o', label=material)
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.axvline(500, color='gray', linestyle=':', label='BF16 resume')
        ax.set(title=key+' / initial (lower is better)', xlabel='Optimizer step'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Fixed validation: topology improves, pair coherence regresses')
    fig.savefig(OUT / 'validation_tradeoff.png', dpi=160); plt.close(fig)

    change_rows = []
    for material in ('Al', 'Mg', 'Ta'):
        initial = snapshot['initial']['by_material'][material]
        current = last['by_material'][material]
        values = [100*(current[k]/initial[k]-1) for k in ('tda_mse','future_tda_mse','spatial_ratio','temporal_ratio')]
        change_rows.append('| '+material+' | '+' | '.join(f'{v:+.1f}%' for v in values)+' |')
    loss_rows = '\n'.join(f'| {k} | {v:.5g} | {100*v/last["loss"]:.2f}% |' for k,v in result['latest_weighted_validation'].items())
    norm_rows = '\n'.join(f'| {name} | {diagnostics["task_gradient_norm_"+name]["median"]:.3f} |' for name in ('spatial','temporal','topology','prediction'))
    text = f'''# Live MACE training review — 2026-09-08

Captured {snapshot['captured_utc']}; batches {frame.index.min()}–{frame.index.max()}, latest fixed validation at step {last_step}. CPU log analysis only; training was not changed. This is an interim review, not completed static analysis or a statistical significance test.

## Findings

The training loss is not diverging: the central 90% of logged BF16 batch losses is {stats['loss']['p05']:.3f}–{stats['loss']['p95']:.3f}, median {stats['loss']['median']:.3f}. Every 10th update logs its current batch, not a ten-update average. Material quotas are fixed (512 each, with Al split into 256 shooting and 256 ordinary anchors); individual neighborhoods, stages and forecast horizons vary. Fixed validation uses the same 768 anchors (256/material). Only {len(snapshot['epochs'])} epoch-end validation measurements are available.

There is nevertheless a real failure of the requested joint-improvement criterion. Changes below compare the same validation set to the initialization of the original FP32 run; negative error changes are improvements, positive changes are regressions.

| Material | TDA error | Future TDA error | Spatial ratio | Temporal ratio |
|---|---:|---:|---:|---:|
{chr(10).join(change_rows)}

Most spatial/temporal degradation already exists at step 500, before this BF16 continuation. The first overlapping FP32/BF16 updates are recorded in summary.json; at step {overlap[0]['step']} total losses differ by {overlap[0]['loss_delta']:.6g}. This is evidence against the precision switch causing the observed jump, not a proof of identical long-term trajectories.

Training and validation losses have different sample populations and batch sizes (1536 versus 768); the covariance regularizer is batch-dependent. Their absolute gap is not itself evidence of overfitting.

## Why the objectives can disagree

Spatial and temporal MSE reward small absolute pair distances. The reported ratios divide these distances by distances to shuffled same-material views, so they test relative discrimination. Lower raw MSE can coexist with worse ratios. The logs do not contain per-material numerators and denominators separately, so their exact contributions cannot be recovered here.

The weighted shared-encoder gradient medians, measured every 100 steps, are:

| Task group | Gradient norm |
|---|---:|
{norm_rows}

TDA supplies a much larger gradient than temporal VICReg. Negative pairwise gradient cosines indicate locally opposing directions; topology–temporal median cosine is {diagnostics['gradient_cosine_temporal_topology']['median']:.3f}, and spatial–prediction is {diagnostics['gradient_cosine_spatial_prediction']['median']:.3f}. These are sparse, batch-dependent diagnostics, not proof that a particular term causes each regression. The spatial group includes nuisance invariance and variance/covariance terms.

{100*result['logged_clip_fraction']:.0f}% of logged updates exceed the configured gradient clipping threshold 5; the median pre-clipping norm is {stats['gradient_norm']['median']:.2f} and median clipping multiplier {result['median_clip_multiplier']:.3f}. This does not mean clipping failed: gradient_norm is explicitly measured before clipping. Clipping rescales the combined gradient, rather than balancing individual tasks. Peak encoder/head LR is 0.0003/0.003 at step 704, followed by per-step cosine decay. A lower-LR controlled comparison could test sensitivity; these curves alone do not establish that LR is the cause.

The joint no-regression screen still fails. It ranks checkpoints; it does not constrain the optimizer. A weighted sum therefore has no mechanism guaranteeing all physical properties improve together.

## What nuisance_mse means

N = mean((encoder(x) - encoder(x + epsilon))**2). Noncentral atomic coordinates receive independent Gaussian noise with sigma 0.005 angstrom per coordinate; the central atom stays fixed. The same 80 atom identities are retained. This is synthetic coordinate-noise invariance, not an MD continuation, topology error, or a direct test of neighborhood membership swaps.

Its latest validation value is {last['nuisance_mse']:.6g}, weighted by 1000 to contribute {1000*last['nuisance_mse']:.5f} to total loss {last['loss']:.4f}. Its numeric size has no universal good threshold because embedding scale is learned. Suppressing every small displacement can also suppress physically meaningful changes; that tradeoff is a hypothesis requiring an ablation. The logs do not isolate this term's parameter-gradient norm. It also adds a fifth encoded view per quadruplet.

## Metric dictionary

| Logged name | Definition / interpretation | Optimized? |
|---|---|---|
| spatial_mse | Anchor vs one of six nearest atom-centered neighborhood embeddings; latent mean squared error | Yes, weight 25 |
| temporal_mse | Same atom 0.1 ps later; excludes Al shooting pairs (1280 eligible pairs/batch) | Yes, weight 6.25 |
| tda_mse | Decoder prediction of 32 train-PCA-whitened coordinates of 144D H0/H1/H2 alpha-complex persistence images; all four views | Yes, weight 10 |
| forecast_mse | Predicted future embedding vs detached current encoder's future output; forecast horizon is conditioned | Yes, weight 5 |
| persistence_mse | Future-embedding error when predicting no change from the anchor | Baseline only; persistence here is unrelated to persistent homology |
| future_tda_mse | TDA decoded from the predicted future embedding vs actual future TDA target | Yes, weight 5 |
| nuisance_mse | Embedding sensitivity to synthetic 0.005 angstrom coordinate noise | Yes, weight 1000 |
| spread | Average across four views of 25 times variance-floor penalty plus off-diagonal covariance penalty, after subtracting material means | Related to VICReg regularization, but this logged average is not its actual weighted contribution |
| loss | Sum of weighted objectives, including weighted VICReg regularization | Yes |
| spatial_ratio, temporal_ratio | Pair error / error against a fixed shuffle of the same material; lower means better relative coherence | Validation diagnostics, used in checkpoint selection |
| forecast_gain_vs_persistence | 1 - forecast_mse / persistence_mse; 0 means no improvement over no-change forecast | Diagnostic; the latent target space itself changes during training |
| future_tda_gain_vs_observed_persistence | 1 - future TDA prediction error / error from using observed anchor TDA unchanged | Diagnostic in a fixed target space; includes decoder error |
| effective_rank | Exponential entropy of the within-material latent covariance spectrum | Diagnostic; ~3–5 despite 256 coordinates, higher than initialization, not evidence of complete collapse |
| gradient_norm | Combined parameter-gradient norm before clipping | Diagnostic |
| task_gradient_norm_* | Weighted task gradient norm on shared encoder parameters | Diagnostic every 100 steps |
| gradient_cosine_* | Alignment between two task gradients; negative means local opposition | Diagnostic every 100 steps |
| gradient_conflict_fraction | Fraction of the six task pairs with negative cosine | Diagnostic, not an error probability |
| pcgrad_applied | Whether conflict projection was applied | Always zero: disabled in this run |
| joint_selection/* | Per-material relative gains, rank retention, and no-regression screening | Checkpoint selection, not a loss or significance test |
| backbone_lr, head_lr | Actual learning rates used for the update | Optimizer settings |
| anchor_exposures, view_exposures | Repeated training presentations, not unique data counts; ordinary views = 4 times anchors | Accounting; augmented views counted separately |

TDA MSE is not a distance between persistence diagrams. A value below 1 does not establish superiority to a material-conditional mean on this particular held-out set; that baseline and component-level errors are needed. Forecast horizons mix Al shooting 1.2/6/12 ps with ordinary continuations 0.4/2/4 ps; Ta validation also includes 1 ps. Do not interpret the aggregate as one-horizon prediction quality.

## Exact loss accounting

L = 25 S + 6.25 T + 10 D + 5 F + 5 U + 1000 N + R.

R = 0.5(R_anchor + R_spatial) + 0.25 R_future + 0.125(R_anchor_eligible + R_temporal_eligible), where each R_view is the material-centered VICReg variance/covariance penalty. The separately logged spread averages four whole-batch R_view values and cannot simply be added to the weighted MSEs. R below is reconstructed by subtraction; component-wise variance/covariance losses were not logged.

| Latest validation contribution | Weighted value | Percent total |
|---|---:|---:|
{loss_rows}

No topology-aware attraction, topology-distance loss, reliability-weighted TDA regression, ranking, or PCGrad is active in this configuration. The stored reliability weights and ranking tolerances are inactive settings. Their presence in config should not be read as additional active methods.

## Logging issue and next steps

The continuation logs validation_initial at training_step=500, but that object is the original step-zero baseline retained for checkpoint selection. The actual step-500 validation is saved as resume_validation.json and is not logged to W&B. Treating the initial series as a resume measurement creates a misleading timeline. This review plots it at step zero and the actual resume measurement at step 500.

For the next instrumentation change, separate physical validation, weighted loss components, and optimizer diagnostics; log true resume validation and separate variance/covariance terms. Show batch curves with rolling summaries, while keeping the fixed held-out measurements visible. Log per-material ratio numerators and denominators and fixed-target skill relative to material/horizon baselines.

For a scientific follow-up, compare reduced backbone LR and task-gradient balancing against this exact starting checkpoint and fixed validation set, changing one factor at a time. Test whether noise invariance helps physical robustness with a controlled ablation, rather than assuming nuisance_mse must be minimized. None of these changes has been applied to the live run. New Ta branches are not in this run, and its 8192 Ta training quadruplets are repeated 44 times per epoch; additional unique data could improve generalization but does not by itself resolve conflicting objectives.

Artifacts: [batch curves](training_curves.png), [fixed validation comparison](validation_tradeoff.png), [machine-readable summary](summary.json), [weighted components](weighted_losses.csv), [validation table](validation_by_material.csv), and [captured inputs](inputs.json). Reproduction is documented in the experiment README.
'''
    (OUT / 'RESULTS.md').write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
