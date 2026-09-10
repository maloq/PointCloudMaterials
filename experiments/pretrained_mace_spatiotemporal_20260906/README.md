# Pretrained small MACE with spatial and temporal structure objectives — 2026-09-06

> Completed experiment record — implementation retired on 2026-09-09.
> Commands, plans, and implementation paths below describe the original run,
> not the current supported trainer. Reproduction of that protocol requires
> the run's `tracking/*/source.tar.gz` and recorded configs under `output/`.
> Existing results/checkpoints are retained. Use the
> [current 80-atom recipe](../mace_plain80_20260909/README.md) for new training.

**Superseded at the user’s request by the [strict 80-point online-W&B run](../pretrained_mace_spatiotemporal_80_20260906/README.md).** The 512-point training process was stopped; its outputs are retained for audit.

Train a smooth atomic encoder that preserves spatially meaningful variations and
supports forecasting, without a GeoFrame teacher. Start from the actual
[MACE-MP-0b2 small MLIP weights](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model),
not the earlier scratch MACE. The checkpoint SHA-256 is
`d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9`.
The original 89-element embedding, radial transform, 5 Å cutoff, density
normalization, two interactions, correlation products, and scalar channels are
preserved. Analysis uses both layers' 256 central scalar features with a fixed,
train-only initialization scaler. There is no output projector or teacher.

## Reproduction

```bash
conda run -n pointnet python -m src.training_methods.pretrained_mace \
  --config experiments/pretrained_mace_spatiotemporal_20260906/config.json
```

Stages `--stage prepare`, `--stage train`, and `--stage analysis` are available.
Training is a fresh fine-tune, not a resume command. Do not relaunch it into an
existing run directory. Preparation reuses the exact recorded complete shards;
use a new cache for different data settings. The full stage chains preparation,
training, selected-encoder export, the existing analysis pipeline and its report.

Configuration: [config.json](config.json). Shared code:
`src/data_utils/pretrained_mace.py`, `src/models/encoders/pretrained_mace.py`,
`src/training_methods/pretrained_mace.py`,
`src/analysis/pretrained_mace_adapter.py`.

## Training data and objectives

| Material | Quadruplets | Distinct neighborhood states |
|---|---:|---:|
| Al | 180,224 | 612,618 |
| Mg | 32,768 | 117,338 |
| Ta | 8,192 | 29,334 |
| Total | 221,184 | 759,290 |

A quadruplet contains a current center, an actual spatial neighbor center, the
same atom at a short future lag, and the same atom at a longer future lag.
There are 884,736 view slots; repeated source/frame/atom identities are deduplicated
in the distinct-state count, including shared initial shooting states across
replicas within each temperature parent; different temperature parents have
different initial coordinates. Neighborhoods overlap and are correlated; these are
not counts of statistically independent samples. Every view has 512 actual atoms
(center + 511 neighbors) and an excluded-neighbor radius above 10.02 Å.

Center IDs are sampled without replacement across each complete cached center
pool using a recorded seed; they are not the leading entries of a sorted ID list.
`sampling_coverage_audit.json` records coverage of atom-ID and spatial quartiles;
initial-state hashes verify deduplication across shooting replicas. The corrected
training centers cover all four quartiles in each Cartesian direction.

Al uses 22 training parents, two velocity shots per parent, 1,024 center IDs and
four anchors per shot. Mg uses four training sources and 2,048 center IDs per
source. Ta uses 2,048 center IDs in one branch, with four anchors. Al/Mg validation
sources remain excluded; Ta uses disjoint atom IDs and a later time block of the
same source. Ta therefore has no independent-source validation. Mg/Ta source
positions are float16, limiting fine-time continuity evaluation. Static Al
analysis snapshots are not added as training replay.

The nearest-neighbor view is drawn from the six nearest atoms. Temporal VICReg
uses 0.3 ps in Al and 0.1 ps in Mg/Ta. Forecast horizons are 1.2/6/12 ps in Al and
0.4/2/4 ps in Mg/Ta training; later Ta validation uses 0.4/1/2 ps.
Forecast conditions contain log(1 + lag/ps), material and shooting temperature
(temperature is inactive for the ordinary trajectories).

- Spatial VICReg acts **directly on the analyzed latent**, weight 1.
- Short-time temporal VICReg also acts on that latent, weight 0.25.
- Both use 25 × invariance + 25 × variance-floor + covariance penalties.
  Material means are subtracted for variance/covariance regularization; chemical
  identity alone cannot prevent collapse. Future features also get a spread term.
- All four latents predict 144D alpha-complex H0/H1/H2 persistence images of the
  center and 64 nearest neighbors. A train-only PCA reduces these to 32 whitened
  targets; retained variance is recorded in `scaling.json`. TDA loss weight is 10.
- A conditioned residual predictor forecasts the long-time latent, weight 5.
  Its target is the current encoder's detached future output. Future features
  also receive TDA and variance/covariance gradients. There is **no EMA network**.

Fine-tune all used backbone weights with AdamW at 3e-5; heads use 3e-4. Effective
batch size is 768 quadruplets (256/material), or 3,072 neighborhoods. Exact
latent-gradient caching with 192-neighborhood chunks evaluates VICReg on the
whole effective batch; it is not accumulation of smaller covariance objectives.
There is no stochastic augmentation between gradient-cache passes.

A balanced epoch is 704 steps / 540,672 quadruplet exposures: every Al example
once, with shuffled Mg/Ta cycling. Limit: 20 epochs or 8 hours of training,
whichever comes first; validation-based early stopping has patience four epochs
and a four-epoch minimum. The best validation composite objective selects the
checkpoint; plateau learning-rate reduction is enabled. Validation uses 256
fixed anchors per material, including all 256 Ta validation anchors. Reports
explicitly distinguish reaching the time limit from convergence. Latest
checkpoints are saved every 100 steps and at epoch end; best checkpoints are
selected at epoch end. Checkpoints do not provide automatic resume semantics.

## Correctness and analysis

Reproduce numerical checks with `PYTHONPATH=. conda run -n pointnet python
experiments/pretrained_mace_spatiotemporal_20260906/verify.py --config
experiments/pretrained_mace_spatiotemporal_20260906/config.json` (requires prepared
data and CUDA; run when the GPU is available). This is experiment verification
code, not another training runner.

The central readout prunes only operations outside the center's computational
dependency graph. Native full vs pruned output maximum error was 8.35e-7;
cuEquivariance vs native was 7.16e-7; rotation error 1.91e-6. Gradient norms agree.
Nine existing static-cache/analysis tests passed. The standard checkpoint and
dataloader bridge was exercised on all 772,953 centers; a two-patch inference
check agreed with the training encoder within 1.20e-6.
Exact gradient caching vs ordinary full-batch backpropagation matched loss and
all populated parameter gradients (maximum absolute difference 3.17e-7).
Preflight batch 768 used 34.63 GB of CUDA allocations; its first complete step
took 6.73 s (an estimate, not a steady-state benchmark).

[static_analysis.yaml](static_analysis.yaml) retains the scientific and display
settings of `configs/analysis/static.yaml`, changing only checkpoint, data override
and output paths. [static_data.yaml](static_data.yaml) keeps all **772,953 original
Al grid-selected atom centers**. The optional static `atomic_context` input
expansion uses `src/data_utils/atomic_context.py` to query 512 real nearest atoms
around those same centers. The full 10 Å halo and distance to the nonperiodic
file boundary are checked. No reflected/padded atoms or inferred periodic boxes
are used. Clustering, continuous connected-regime outputs, projections, MD plots,
representatives and Blender rendering use the existing pipeline.

After analysis, the workflow writes a table of continuous spatial coherence and
cluster-neighbor agreement against the user's old GeoFrame and the previous
predictive density run, restricted to exactly shared centers. These models are
evaluation references only. No PTM label is treated as ground truth. Greater
spatial coherence alone does not establish more physically useful embeddings.

## Outputs and status

Repository output: [`../../output/pretrained_mace_spatiotemporal_20260906/`](../../output/pretrained_mace_spatiotemporal_20260906/).
Live status: `status.json`; logs: `run.log`; epoch metrics: `training.jsonl`;
completed report: `RESULTS.md`; full static outputs: `static_analysis/`.
The selected weights, MLIP initialization, provenance, data counts, metrics and
figures stay in the repository. Bulk neighborhood caches, expanded static inputs
and optimizer checkpoints are under the configured `/home/ids/...` cache.

The first brief run was discarded during epoch 1 when the sampling audit found
a bias from taking leading sorted atom IDs. Its artifacts are preserved in
`output/pretrained_mace_spatiotemporal_20260906/preliminary_sorted_subset/`. The
active run starts afresh from the MLIP weights with uniform subsets.

The detached process runs inside existing allocation 983527, not a new Slurm
submission. `launch.json` records its PID, session, command and start time.

New `src/` components are maintained scientific implementations; this directory
is a versioned experiment record. `output/` scripts/logs are disposable
diagnostics and generated results. Do not interpret a launched run as a completed
or converged experiment; consult `status.json` and `RESULTS.md`.
