# Balanced MACE representation training — 2026-09-08

> Completed experiment record — implementation retired on 2026-09-09.
> Commands, plans, and implementation paths below describe the original run,
> not the current supported trainer. Reproduction of that protocol requires
> the run's `tracking/*/source.tar.gz` and recorded configs under `output/`.
> Existing results/checkpoints are retained. Use the
> [current 80-atom recipe](../mace_plain80_20260909/README.md) for new training.

Question: can spatial coherence, temporal coherence and TDA reconstruction improve
together when nuisance and prediction objectives are removed and task gradients
are balanced? The user requested 12 epochs followed by analysis.

Configuration: [training.json](training.json), [queue](plan.json),
[static analysis](static.yaml), [tracked run spec](run_spec.json).
Output: [output/mace_balanced_representation_20260908](../../output/mace_balanced_representation_20260908/).

```bash
conda run -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_balanced_representation_20260908/run_spec.json
```

Fresh official MACE-MP-0b2 small MLIP weights, new TDA decoder, fresh AdamW and
train-fitted scalers. No old fine-tuning weights, forecast head, coordinate-noise
augmentation, future-embedding loss, future-TDA loss, teacher, topology ranking,
or PCGrad. The previous training was stopped at the user's request; its selected
completed-epoch model is preserved as `interrupted_best.pt` in its original run.

The three raw tasks are spatial VICReg, temporal VICReg, and TDA MSE. Spatial
VICReg retains its 25-times pair MSE, half the anchor/spatial variance-covariance
penalties, and the existing quarter-weight regularizer on the fourth view.
Temporal VICReg uses 25-times pair MSE and half its two eligible-view penalties.
TDA supervises all four ordinary views. Future states remain ordinary
representation examples, with no predictive objective.

At step 1 and every 50 updates, measure each raw task's gradient norm on all
trainable encoder parameters. Maintain a norm EMA with decay 0.9. Set relative
weights to inverse EMA norm, clipping ratios against their geometric mean to
[0.05, 20], then normalize the three weights to sum to three. Weights are detached
and held between calibrations; both encoder and head gradients use the same
weights. Calibration uses actual parameter gradients, not just latent gradients.
The EMA, weights and update count are saved as model buffers. This balances
gradient magnitudes, not directions, and does not guarantee joint improvement.
Log raw task-gradient norms, weighted norms, weights and weighted loss components.
Because weights evolve, total loss across distant steps is not a fixed objective;
selection uses the fixed validation properties instead.

12 complete epochs = **8,448 optimizer updates**, batch **1,536** quadruplets,
four views, **80 atoms** each, microbatch 1,536 clouds. One-epoch warmup then
per-step cosine decay; peak encoder LR **0.0001**, decoder LR **0.001**, minimum
LR 0.000001. Patience exceeds the 12-epoch budget. Compensated BF16 radial
matrices, GPU-resident data, compiled radial MLPs and geometry caching retain
the previously qualified execution path; geometry, losses and master weights
remain FP32. Model architecture and target-complete support are unchanged.

The prepared dataset has 245,760 Al, 32,768 Mg and 8,192 Ta training quadruplets:
**286,720 unique stored quadruplets**, containing **993,768 distinct neighborhood
states** according to the producer's counts. Equal material quotas repeat the
smaller pools (Ta 44 times per epoch). Total training presentations are
12,976,128 quadruplets / 51,904,512 ordinary views; these are not unique counts.
Temporal VICReg uses 0.1 ps Al/Mg/Ta pairs and excludes Al shooting pairs.
No new Ta branches or Ti data are introduced by this experiment.

Validation reuses a fixed draw of 256 anchors per material. Select the checkpoint
lexicographically by no-regression feasibility, worst relative gain, then mean
relative gain over TDA MSE, spatial ratio and temporal ratio for every material;
retain at least 90% of initial effective rank. Predictive scores do not enter
selection. All 12 epochs run before selecting/exporting for analysis. `last.pt`
also remains in the run's repository checkpoint directory. Keeping these two
optimizer checkpoints in the shared repository avoids session-specific `/tmp`
mount visibility; their combined size is approximately 200 MB.

The detached queue first compares real-MACE ordinary/calibrated cached gradients
to full backpropagation, then exercises a full batch at peak LR. Preflight updates
are discarded; it uses existing train scalers solely for this numerical check.
After training, the maintained frozen ridge probes evaluate TDA, temporal
coherence and forecast information (evaluation only). The existing encoder-only
static-Al pipeline runs on all six configured frames, with the existing spatial
comparison to archived GeoFrame and MACE results. Static frames include training
ancestors; these visualizations are descriptive, not independent phase labels.

The allocation ends 2026-09-09 05:55 Paris; the queue reserves the final 30 minutes.
Online W&B: <https://wandb.ai/teshbek/PointCloudMaterials/runs/b3r12s08>.
Status and numerical evidence are under the output directory. Training and
analysis completed at 23:29 Paris on September 8: all 12 epochs, selected epoch
12, and 772,953 static-Al neighborhoods. See the
[final review](../../output/mace_balanced_representation_20260908/FINAL_REVIEW.md).
Mean static neighbor/random MSE is 0.459 versus archived MACE 0.495 and old
GeoFrame 0.400; frozen TDA probe mean within-material R² is 0.325. The rank
retention screen failed, so this is not an all-properties success.

Launch verified at update 20/8448 with online W&B, zero forecast-head parameters
and zero nuisance views. Nine CPU tests passed; real-MACE calibrated and ordinary
cached gradients matched direct backpropagation with maximum absolute errors
1.69e-6 and 2.18e-6. Full-batch peak-LR preflight used 59.2 GiB; calibrated
task norms were equal. Early ordinary updates take approximately 2.15 seconds;
allow roughly 5–6 hours for training, followed by probes and static analysis.
See [launch verification](../../output/mace_balanced_representation_20260908/launch_verification.json)
and [numerical preflight](../../output/mace_balanced_representation_20260908/preflight.json).

File roles: this directory is a versioned experiment record. Shared balancing
and preflight implementation is `src/training_methods/mace_balanced.py`, reached
through the existing trainer and queue, with gradient/selection tests in
`tests/test_mace_balanced.py`. Logs, generated reports and status files are run
artifacts. No additional script runner or duplicated training loop was created.

## Interim loss audit — 2026-09-08

Run `conda run -n pointnet python experiments/mace_balanced_representation_20260908/review.py`
from the repository root to capture the live curves and compare the logged
adaptive total with a fixed-coefficient total on identical validation outputs.
The CPU-only diagnostic also fits material-mean TDA baselines using the actual
training pool proportions and fixed validation draw. It changes no live training
state. This is experiment-specific reproducibility code; captured logs, CSV
files, figures and report are generated run artifacts.

[Findings and plots](../../output/mace_balanced_representation_20260908/loss_review/RESULTS.md):
the rising total mostly reflects changing coefficients; the decoder beats
material-mean baselines, while substantial variance concentration causes the
rank-retention screen to fail. The requested 12 epochs and analyses remain queued.
