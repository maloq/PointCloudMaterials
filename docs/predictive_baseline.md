# Local shooting predictive baseline

Implementation: `src/research/predictive_baseline/`. Recipe:
[`al480_20261001.json`](../configs/predictive_baseline/al480_20261001.json).
Scientific question and protocol: [research record](../experiments/predictive_baseline_20261001/README.md).
Metric formulas: [frozen metric specification](metrics/predictive_baseline.md).

Use conda `pointnet-torch214`. The installed e3nn constants require the same
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` setting as existing native MACE workflows.

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m src.research.predictive_baseline.queue prepare --config configs/predictive_baseline/al480_20261001.json
python -m src.research.predictive_baseline.queue preflight --config configs/predictive_baseline/al480_20261001.json
python -m src.research.predictive_baseline.queue submit --config configs/predictive_baseline/al480_20261001.json
```

Preparation verifies the existing shooting release, preserves all roles, creates
and seals the training-defined feature bank, and records exact input contracts.
It refuses to overwrite a sealed bank. The full 256-row preflight needs a GPU
allocation. It measures a real backward pass and invariance without W&B.

Submission freezes source, metric definitions and required recipes under the
STORE run's `technical/code`. Four local CPU controls and a three-element GPU
array run independently; a CPU collector depends on both successful completion.
The collector performs frozen readouts, source bootstrap, plots and final W&B
summary updates. Slurm jobs continue independently of the chat/session.

Read `output/predictive_baseline/al480-274-20261001/technical/launch.json` for IDs.
Each seed has `analyses/joint-seed-SEED/technical/progress.json`, `last.pt`,
`best.pt`, `wandb.json` and completion receipts. Log files are under run-level
`technical/`. Training checkpoints after every epoch. On a stopped allocation,
reuse the frozen `fit.sbatch` with `sbatch --array=INDEX .../fit.sbatch`; it resumes
the last completed epoch with the same W&B ID, optimizer and random states.
If dependencies failed, resubmit the recorded collector with updated `afterok`
dependencies after all required fits complete. Never edit a frozen bundle.

Targets and training arrays use IDS; protected checkpoints, predictions, metric
exports and frozen producers use STORE. The repository output entry is a link to
the STORE run. No new MD is generated. The six-entry existing encoder-cache
policy remains in place; preparation reads frozen exports with shared leases.

## Matched head, target and MM-TDA follow-up

Use `configs/predictive_baseline/followup_20261001.json` with module
`src.research.predictive_followup.queue`. Stages are `prepare`, `encode`,
`preflight`, `submit`, `joint`, `probe`, `controls`, and `collect`.
Run `prepare`, then GPU `encode` and `preflight`, then `submit`, all with
`--config configs/predictive_baseline/followup_20261001.json` and the same conda/
e3nn environment above. No new trajectories or target-map derivation is needed.

The exact MM-TDA-BLOCK-DIRECT-FULL checkpoint is extracted through its recorded
producer into the established six-entry leased feature cache. Prepared baseline
arrays remain sealed. Preparation records the actual MM-TDA packed source paths
and retains limitations on unknown shared archived preparation ancestry.

Submission freezes a new producer and queues12 scientific GPU fits,36 local CPU
frozen-head fits, local ridge references, and dependent collection/plots. Three
seeds cross full/moment supervision and free/nonnegative variance. Every new fit
uses the same moment-based validation selector. MM-TDA remains frozen at z256;
new encoders retain the default128 dimensions and batch/microbatch256.

Slurm uses L40S or RTX6000PRO and excludes node52, where the baseline's CUDA
initialization failed. GPU jobs request30 minutes based on the measured baseline
runtime. Resume a stopped arm through its unchanged frozen sbatch script and
array index; update collector dependencies if necessary. Complete progress and
checkpoint receipts live in each arm's `technical/`. Frozen heads remain local;
only joint encoder fits open online W&B runs. Scientific protocol and interpretation:
[follow-up record](../experiments/predictive_baseline_20261001/FOLLOWUP.md).

The initial array submission exceeded the scheduler's per-user submitted-job
limit after the controls job was accepted. The maintained grouped launcher is
`python -m src.research.predictive_followup.lanes submit --config configs/predictive_baseline/followup_20261001.json`.
It defaults to four CPU workers (nine frozen fits each) and one GPU worker (12
joint fits), requesting90 and120 minutes respectively. `--gpu-lanes` can change
the GPU worker count before those workers are submitted. Every fit still invokes
the original frozen worker and retains its own checkpoints, selector and tracking.
The operational wrapper has its own checksum under `technical/execution-lanes-v1`;
the frozen scientific bundle and existing metric exports are preserved.
The launcher can continue a partially accepted submission without duplicating
its accepted workers. The actual accepted worker counts and any recovered
scheduler errors are retained in `technical/launch.json`.

## PaCMAP views of the completed follow-up

`python -m src.research.predictive_followup.pacmap_views run --config configs/analysis/predictive_pacmap_20261001.json`
projects the saved VICReg128, Epi128, MM-TDA256 and all twelve new joint embeddings.
Run in `pointnet-torch214` with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2`. No encoder or readout is retrained.
The `project` and `render` stages also run separately; rendering reuses saved
coordinates. Projections fit on training observations and transform the unchanged
historical test rows, with training-only coordinate standardization and no color
labels in the fit. Every encoder uses identical PaCMAP settings.

The [interactive comparison](../output/predictive_baseline/heads-targets-mmtda-20261001/analyses/pacmap-v1/index.html)
offers present structure, measured future means/spread and source colors, with
population filtering and linked point selection. The joint encoder selector
includes all four treatments and three seeds; frozen reference panels stay fixed.
Static PNG/PDF views cover present structure, future6ps crystallinity, clear-liquid
future6ps crystallinity and source audit. Full definitions and interpretation limits
are in [the PaCMAP protocol](metrics/predictive_pacmap.md). The separate output
bundle preserves the original prediction metrics and their frozen definitions.
