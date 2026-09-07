# Strict 80-point pretrained MACE, with W&B online — 2026-09-06

Stopped at the user's request and continued from saved step 2,600 in the
[0.1 ps / per-step cosine experiment](../pretrained_mace_80_dt01_cosine_20260906/README.md).
The protocol below records the superseded phase.

User-requested replacement of the 512-point run: use **exactly 80 actual atoms**
(center + 79 neighbors) in training and static analysis, with online W&B logging.

## Run

```bash
conda run -n pointnet python -m src.training_methods.pretrained_mace \
  --config experiments/pretrained_mace_spatiotemporal_80_20260906/config.json
```

This is a **fresh fine-tune from the original MACE-MP-0b2 small MLIP weights**.
It does not resume the 512-point optimizer. The old run was stopped; its files
remain in `output/pretrained_mace_spatiotemporal_20260906/` and its status points
to this replacement. As with the original workflow, use a new output directory
and W&B run ID for another fresh training run. `--stage analysis` reruns only the
selected-encoder export and standard analysis after training.

[Configuration](config.json), [static data](static_data.yaml),
[standard analysis configuration](static_analysis.yaml).

Online dashboard: <https://wandb.ai/teshbek/PointCloudMaterials/runs/qn8kpl1s>.
The workflow explicitly requests online mode, uses the existing authenticated
`PointCloudMaterials` project, and fails loudly if initialization cannot create
an online run. It logs individual loss components, gradients, learning rates,
training exposures, validation metrics and W&B system telemetry. Dataset arrays
and model artifacts are not uploaded by this workflow.

## What changes scientifically

Eighty atoms cannot contain the original complete two-interaction 10 Å context.
The compact model preserves the pretrained **5 Å edge cutoff** and multiplies
edge weights and density-normalization contributions by a smooth function of the
sender's distance from the center:

- Weight 1 for distance ≤ 5 Å.
- Quintic fade `1 - 10u³ + 15u⁴ - 6u⁵`, `u = (r - 5)/1.5`, between 5 and 6.5 Å.
- Weight 0 outside 6.5 Å.

The central first-layer 128D descriptor is unchanged at MLIP initialization.
The second-layer 128D descriptor sees a deliberately smaller context and is
fine-tuned for it. This is not claimed to reproduce the full two-layer MLIP
calculation. The model continues to emit 256 scalar features.

The first excluded neighbor lies beyond 6.5 Å in every cached training/validation
view (global minimum 6.719 Å) and every static Al view (minimum 6.633 Å).
Consequently a change in the 80th-neighbor identity occurs outside the nonzero
support and cannot introduce a finite descriptor jump, up to floating-point
roundoff. All 80 inputs are real atoms; no hidden additional atoms are passed to
the model. `points: 512` in the training configuration describes the existing
**backing cache**; `model_points: 80` slices its memory maps before batching,
scaling and inference. The TDA targets remain valid because they use only the
center and the 64 closest neighbors.

The standard static loader keeps its original reference grid/cache settings to
preserve all **772,953 Al centers**, then `atomic_context.points: 80` supplies the
actual model input. Its complete-context check now checks the explicit compact
6.5 Å support instead of claiming a 10 Å halo.

## Data and optimization

The data split and all four objectives are inherited unchanged from the
[original experiment](../pretrained_mace_spatiotemporal_20260906/README.md):
spatial VICReg, short-lag temporal VICReg, TDA prediction, and conditioned
future-latent prediction. No teacher or EMA model is used.

| Material | Four-view training examples | Distinct neighborhood states |
|---|---:|---:|
| Al | 180,224 | 612,618 |
| Mg | 32,768 | 117,338 |
| Ta | 8,192 | 29,334 |
| Total | 221,184 | 759,290 |

Effective batch: 768 examples / 3,072 views, balanced across materials.
Encoder chunk: 768 neighborhoods. Exact gradient caching preserves the whole
batch covariance objective. There are 704 steps per balanced epoch. Training
stops at 20 epochs, eight hours, or the original validation early-stopping rule.
Al/Mg validation uses held-out sources; Ta has disjoint IDs and later times in
its single trajectory. Correlated neighborhoods are not independent samples.

The first isolated training steps took roughly 1.2 seconds each, versus about
4.8 seconds for the 512-point run. The numerical preflight, run while the old
job still occupied the GPU, took 1.84 seconds per full step and allocated 22.9 GB.
Runtime estimates should use the ongoing `status.json` and epoch measurements.

## Verification and outputs

```bash
PYTHONPATH=. conda run -n pointnet python \
  experiments/pretrained_mace_spatiotemporal_80_20260906/verify.py \
  --config experiments/pretrained_mace_spatiotemporal_80_20260906/config.json
```

The versioned verification checks actual `(B,4,80,3)` input, equivalence with a
wider input under the same finite support, unchanged initial first-layer MLIP
features, rotation invariance, outer-neighbor replacement, and exact gradient
caching against ordinary backpropagation. Measured maximum errors are recorded
in `verification.json`; all checks passed. Static data preparation also verified
that all 772,953 output samples contain exactly 80 points. The exported-checkpoint
and standard analysis-loader integration also passed, with a maximum difference
of 1.85e-6 from the training encoder on its two-sample check.

Repository results: `output/pretrained_mace_spatiotemporal_80_20260906/`.
Live files: `status.json`, `wandb_run.json`, `run.log`, `training.jsonl`.
After training: `best.pt`, `encoder/encoder.ckpt`, `static_analysis/`, `RESULTS.md`.
The full existing static analysis and comparison with old GeoFrame/density
outputs remain automatically queued, using only the selected encoder.
Large cached inputs and optimizer checkpoints stay under `/home/ids/...`;
analysis results and selected weights stay in the repository.

The process is detached inside existing allocation 983527; no new Slurm job was
submitted. `launch.json` records the PID and session. Shared changes in `src/`
are maintained implementation; this directory is an experiment record with
verification code; logs and diagnostics in `output/` are disposable artifacts.
