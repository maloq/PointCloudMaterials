# Matched pretrained MACE objective ablations — 2026-09-07

The control failed at step 2,000 on an `/home/ids` checkpoint quota error.
The user requested recovery and 5× peak LR; the
[replacement queue](../pretrained_mace_ablations_lr5_20260907/README.md) preserves
this failed attempt and restarts all matched trials with node-local checkpoints.

Question: which objectives help the compact pretrained MACE encoder preserve
spatial structure, temporal continuity, TDA information and predictable futures?
Requested as a detached queue **after** the ongoing 0.1 ps MACE training and its
existing static analysis complete.

## Schedule

```bash
conda run -n pointnet python -m src.training_methods.pretrained_mace_queue \
  --plan experiments/pretrained_mace_ablations_20260907/plan.json
```

The controller waits for the predecessor's successful `complete` status **and**
process exit (PID plus Linux process start identity). It uses the current node
inside allocation 983527; it submits no Slurm jobs. The allocation currently
ends September 7 at 18:37 Paris. The controller reads the allocation end before
each stage and reserves its last 15 minutes. Child failures stop the queue with
a traceback and retained logs. Insufficient time is recorded as pending work,
never completion. The controller is single-instance and is not a resume command;
do not rerun it over completed trial directories.

| Order | Configuration | Removed encoder-training loss |
|---|---|---|
| 1 | [all_objectives.json](all_objectives.json) | None; matched control |
| 2 | [no_temporal.json](no_temporal.json) | Temporal VICReg |
| 3 | [no_tda.json](no_tda.json) | TDA prediction |
| 4 | [no_forecast.json](no_forecast.json) | Future-latent prediction |
| 5 | [no_spatial.json](no_spatial.json) | Spatial VICReg |

Each model trains for **four complete epochs / 5,632 optimizer updates**,
then receives frozen-encoder probes. Only after all five trainings/probes finish
does the queue run the five static analyses and update the comparison table.
At the observed approximately 0.95 seconds per update, training takes about
1.5 hours per model, plus preparation/probes and analysis. The plan reserves
95 minutes for admission of each training and estimates 15 minutes per static
analysis; these are estimates, not guarantees of finishing within the allocation.

## Matched scientific protocol

Every trial starts afresh from the **original small MACE-MP-0b2 MLIP weights**
with the same seed, head initialization, train-only scaling/PCA, sampling order,
batch size and learning-rate schedule. No trial initializes from the ongoing
fine-tune, because it has already learned all the objectives being ablated.
That longer warm-started run remains a useful reference, but is not the matched
control. There is no GeoFrame teacher or EMA model.

All trials retain actual 80-atom neighborhoods, a smooth 6.5 Å context, Al/Mg/Ta
and verified 0.1 ps temporal pairs. The existing prepared cache is shared read
only. The data contains **286,720 distinct quadruplet records / 993,768 distinct
source-frame-center states**, representing 1,146,880 view slots. A batch contains
128 Al shooting +128 Al continuation +256 Mg +256 Ta anchors. Four epochs give
4,325,376 anchor exposures /17,301,504 view exposures. Correlated and repeated
neighborhoods are not independent samples.

The first epoch warms up from 5% to peak LR (encoder 3e-5; heads 3e-4); cosine
decay updates after each optimizer step over the remaining three epochs, ending
at 1e-6. Validation early stopping is disabled for this fixed-budget comparison.
**Final-epoch checkpoints** are analyzed consistently, rather than selecting
different epochs using differently weighted validation losses. The legacy
filename `best.pt` holds the selected analysis checkpoint; the summary records
`checkpoint_selection: last` and the actual selected epoch/validation separately
from the best weighted validation epoch. The unchanged main run still selects
its best validation checkpoint.

Removing spatial or temporal VICReg removes the entire corresponding pair loss,
including its variance/covariance terms. Removing forecasting removes the
future prediction loss. All variants retain the explicit shared 0.25 future-view
variance/covariance regularizer; this is not a test of every future-view signal.
TDA gradients during encoder training are zero in `no_tda`; evaluation probes
are fitted afterward on frozen embeddings. Al shooting's 0.3 ps pairs remain
excluded from temporal VICReg in every variant.

## Evaluation

Identical ridge probes (alpha=10, train-only feature standardization) fit on
2,048 training anchors per material and evaluate on the training workflow's
fixed 256 validation anchors per material. They measure TDA reconstruction,
conditioned future TDA prediction and future-latent prediction relative to
persistence. The disabled in-training heads are never used to rank encoders.
TDA R² in the table averages within-material R², preventing element identity
alone from producing a good score. Per-material effective rank and temporal
MSE relative to shuffled same-material pairs accompany these metrics.

Static analysis uses the same standard pipeline and all **772,953 centers**
across six Al frames. Each `*_static.yaml` changes only checkpoint/output paths
and disables Blender ray tracing relative to the current main analysis config.
Clustering, projections and interactive plots otherwise retain its settings.
The main run's full analysis is unchanged. Spatial neighbor/random feature
ratios and adjusted neighbor cluster agreement are collected into the table.

This is a **single-seed four-epoch ablation screen**, not a convergence or
statistical-significance claim. Static frames include ancestors of training
continuations and are not an independent test. Smooth clusters alone do not
prove physical phase identification. Follow-up training should depend on the
resulting curves, rank, temporal, probe and spatial metrics together.

## Verification, outputs and file roles

```bash
PYTHONPATH=. conda run -n pointnet python \
  experiments/pretrained_mace_ablations_20260907/verify.py
```

Checks cover matched configurations and update counts, exact objective removal,
dependency exit gating, failed child propagation and a known linear probe.
The queue runs online W&B for every training; IDs are fixed in each configuration.

- `output/pretrained_mace_ablations_20260907/status.json`: detached controller state.
- `output/pretrained_mace_ablations_20260907/RESULTS.md` and `comparison.csv`:
  incremental table; pending rows remain explicit.
- `output/pretrained_mace_ablations_20260907/runs/NAME/`: selected weights,
  probes, training curves, static plots and analysis reports in the repository.
- Bulk optimizer checkpoints stay under
  `/home/ids/vmorozov/experiments/pretrained_mace_ablations_20260907/`.
  Only these new runs' disposable static inference caches are removed after
  reporting. Selected encoders, plots and metrics remain; existing caches and
  checkpoints are not deleted by this queue.

The configs, plan, verification code and this README are experiment records.
Maintained orchestration is `src/training_methods/pretrained_mace_queue.py`;
shared probes/collection are `src/analysis/pretrained_mace_ablation.py`. Training
and static analysis reuse their existing family commands. Logs, launch records
and generated diagnostics are disposable run outputs. Findings are pending.
