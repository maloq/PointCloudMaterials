# Encoder training and context comparison

Recipe: `configs/encoder_context/al64_20260925/campaign.json`.
Implementation: `src/research/encoder_context/` using the existing supervised and
context trainers. [Scientific protocol](../experiments/encoder_context_epochs_20260925/README.md).

The [larger mixed-material structural release](datasets/structural_multimaterial_256.md)
has a separate prepared recipe at
`configs/encoder_context/multimaterial256_20260925/campaign.json`. It uses bounded
disk-backed pretraining, fixed material length normalization, and the original
geometry-only encoder with one constant atom channel. Run the same campaign commands
with that config only after the new structural manifest is complete; existing
launched fits retain their frozen recipes and code.

Use conda `pointnet-torch214` from the repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCHINDUCTOR_COMPILE_THREADS=4
python -m src.research.encoder_context.queue prepare --config configs/encoder_context/al64_20260925/campaign.json
python -m src.research.encoder_context.queue check --config configs/encoder_context/al64_20260925/campaign.json
python -m src.research.encoder_context.queue launch --config configs/encoder_context/al64_20260925/campaign.json
```

Preparation freezes recipes/source identities and audits current-frame context
ancestry. Checks exercise real-data gradients without online runs. Launch
freezes code and starts two detached one-GPU Slurm tasks in the declared current
allocation, with checkpointed sequential stages. No scientific run is left
untracked: W&B online group `encoder-context-al64-epochs-20260925` in
`teshbek/PointCloudMaterials`. Training curves include epoch, objective, gradients,
learning rates, validation proper scores and AP diagnostics where labels apply.

Results: `${storage:analysis}/encoder_context/al64-epochs-20260925`.
Caches: `${storage:scratch}/training-cache/encoder-context/al64-epochs-20260925`
(outside the repository). Existing sealed inputs remain on IDS.
Structural positions fit directly in VRAM. Fixed-80 spatial graphs are constructed
on the GPU per batch; CPU/GPU parity checks cover outputs and gradients against
the compact graph producer. Context features are exported once per encoder and
shared by both predictors. One source shard is normalized at a time to bound host
RAM, with training-only feature statistics. Dense observed evaluation uses all
test centers/frames and is prepared once for the four observed encoders.

The lane states and per-stage logs/checkpoints live under `technical/`. A failed
stage stops its lane and records the traceback. Resume its `worker` command with
the frozen launch config and the same stage/method/domain, inside a valid GPU
allocation. Do not resume using edited workspace code. Completed stages are
verified and skipped; a timeout never marks an incomplete epoch budget complete.
The last lane collects tables/plots after both lanes complete. To collect manually:

```bash
python -m src.research.encoder_context.queue report --config configs/encoder_context/al64_20260925/campaign.json
```

Full evaluation tables are per encoder (`base-hot` / `base-cold`), while each
context predictor keeps its full predictions and separate legacy16 scores.
`RESULTS.md`, `tables/comparison.csv`, metric definitions and `plots/` summarize
the complete study. No context predictor is trained on an old encoder merely
because its feature cache already exists.

Storage policy (user update): no quota checks, archive scans or earlier-run
recovery dependencies in this queue. New cache directories are created directly
on SCRATCH. The independent historical-cache archive was stopped with its
originals retained; that operation is not a prerequisite for scientific work.
The current pretraining jobs keep their frozen code and online run IDs. A
machine-local path mapping directs their campaign's later cache writes to the
same external SCRATCH directory without restarting training.

The active run's `technical/continuation-receipt.json` records its detached
continuation. It waits only for the running structural fits and dense evaluation
inputs, then starts the eight encoder/predictor pipelines. It uses the frozen
bootstrap code and the external-cache mapping recorded in
`technical/cache-location.json`; no earlier-run export or storage archive blocks
it. Historical interrupted Al64 exports remain separately marked incomplete.

## Shared caches and training scheduling

New launches use the bounded cache implementation in
`src/research/equivariant_context/cache.py`. The already launched Al64 campaign
continues from its frozen bootstrap code. Do not resume that campaign from the
modified workspace. These execution changes preserve the likelihood objectives,
source splits, checkpoint selection, feature normalization and predictor inputs.

- **Six encoder caches total**, across methods, observed/relaxed encoders and
  GPU lanes, in `${storage:scratch}/training-cache/context-features/entries/`.
  Observed and relaxed checkpoints count separately. Access updates recency;
  admission evicts the least recently used inactive entry before writing a new
  one. Active process leases prevent deletion during export or predictor fits.
  If all six entries are leased, admission waits with a deadline; it does not
  temporarily create a seventh. Kernel locks release on process exit/crash.
- **One cohort geometry cache**, separate from encoder features, in
  `${storage:scratch}/training-cache/context-geometry/entries/`. All treatments
  reuse its observed/relaxed patch membership, coordinates, directed edges,
  offsets, representative identities and spatial relationships. Its key includes
  population/ancestry inventory, cutoff and geometry producer fingerprints.
  Per-frame locks permit concurrent producers/readers without partial reads.
  An old cohort is evicted when an inactive slot is needed for a different one.
- Checkpoints, predictions, metrics, training data and fixed releases are outside
  eviction roots. The small `technical/feature-cache-{domain}.json` and
  `geometry-cache-{domain}.json` receipts locate disposable caches. Evictions
  are logged in each managed root's `evictions.jsonl`. Cache existence is never
  treated as training completion. Missing features are re-exported from the
  selected encoder checkpoint when a pending predictor needs them.
- A pipeline holds one feature-cache lease and loads the union of required
  fields once. Both predictors use views of the same normalized GPU tensors;
  each receives exactly its declared fields and its own recorded scalers.
  Normalization remains source-weighted and fitted on train sources only.
- CPU geometry/dense-input preparation overlaps fitting. After each predictor's
  selected checkpoint exports all row-aligned probabilities,
  `technical/gpu-complete.json` records training completion. A separate CPU
  `metrics` worker performs calibration, source bootstrap, metric exports and
  updates the **existing** online W&B training run. `complete.json` is written
  only after those steps succeed. There is at most one metrics child per lane.
  Full readout, noise and dense-trajectory evaluations follow each lane's core
  fits; they no longer sit between consecutive encoder training pipelines.

The campaign worker stages are `geometry`, `dense`, `pretrain`, `pipeline`,
`metrics` and `evaluate`. Standalone context submission uses one `pipeline` GPU
job per domain followed by a dependent CPU `metrics` job. Resume using the frozen
launch config and the corresponding stage. A failed CPU child is reported as a
lane failure, and the final collector waits for both fits and evaluations.

Local implementation checks used real observed/relaxed patches and saved feature
shards: cached graph arrays matched exactly; shared versus separate corpus loads
gave bitwise-equal scalers, normalized inputs and predictor outputs. Cache
admission checks covered recency, shared SCRATCH leases and refusal of a seventh
entry while all six slots are active.
CPU scoring reproduced a completed run's calibrated probabilities, predictive
scores and all 1,000 source-bootstrap draws exactly.
These checks created no online W&B runs or new automated test suite. Whole-queue
speedup has not yet been measured.

## Expanded campaign launch, 25 September 2026

The original recipe is `configs/encoder_context/multimaterial256_20260925/campaign.json`.
Results and the frozen launch receipt are under
`${storage:analysis}/encoder_context/multimaterial256-epochs-20260925-r2`.
It runs detached in allocation 1008517 on node59, using two RTX PRO 6000 GPUs,
one per lane: physical then scratch; VICReg then Epi. Physical initialization
uses all 14,058,484 mixed-material training patches. VICReg/Epi use the available
345,600 Al observed/relaxed pairs. Each structural treatment requests 12 epochs;
the eight encoder continuations and sixteen context predictors request 24 each,
at batch/microbatch 256, with online W&B and the unchanged Al64 evaluation.

An initial startup attempt in the unsuffixed output was interrupted after its
CPU geometry worker hit a filesystem shared-lock error. Its frozen code/logs and
scientific run receipt remain recorded there; it is not a completed treatment.
The correction opens shared lease files with read/write access, verified on
SCRATCH before the r2 launch. Use r2's frozen source for continuation. Training
checkpoints before the allocation deadline if its full epoch budget is unfinished.

## Batch-1024 restart, 25 September 2026

The batch-256 r2 step was killed at 17:19 Paris time; its checkpoints and logs
are retained. The replacement recipe is
[`multimaterial256_batch1024_20260925/campaign.json`](../configs/encoder_context/multimaterial256_batch1024_20260925/campaign.json).
It uses allocation 1009176 on node61, two GPUs with one per lane, and writes to
`${storage:analysis}/encoder_context/multimaterial256-b1024-20260925`.
Use the same `prepare`, `check`, and `launch` commands above with this recipe.

Effective batch and microbatch are both **1024** for structural and supervised
encoder training and the context predictors; frozen probes also use batch 1024.
This is an explicit run-level deviation from the batch-256 default. The seed,
data releases, learning rates, 128-update warmup, label-free pretraining objectives,
predictive likelihood objectives and 12/24/24 epoch budgets are unchanged. The
43,523 fitting windows require 43 updates per epoch and 1,032 updates per
24-epoch supervised/predictor fit, with the final partial batch retained.
Validation and checkpoint saving remain once per epoch, with supervised
selection allowed from epoch 12 onward.

The replacement starts from the matched random seed, **not** the batch-256
optimizer state. Changing batch size changes the update schedule and paired
covariance objectives; it is a new training run, not an exact checkpoint resume.
Online W&B uses group `encoder-context-multimaterial256-b1024-20260925` and fresh
stable run IDs. The structural release, prediction population, shared geometry
and dense-input caches are reused. Six generated encoder feature caches remain
the global limit; checkpoints, predictions and metrics are preserved.

## Epi reference-runtime recovery

After all four batch-1024 VICReg context evaluations completed, the Epi stage
failed before training while copying the cuEquivariance reference runtime.
Copying its cached graphs can recreate CPU constants and invalid fused input
bindings. The reference now uses a freshly constructed encoder with the exact
initial state loaded, inside a restored RNG scope. It remains frozen. A local
batch-1024 observed/relaxed check verified identical state tensors, preserved
CPU/CUDA RNG, finite reference scores and output agreement within absolute 1e-6.

The active physical fit and completed VICReg artifacts retain their frozen code.
Only Epi is restarted from a corrected code snapshot under
`configs/encoder_context/multimaterial256_batch1024_epi_recovery_20260925/` and
`${storage:analysis}/encoder_context/multimaterial256-b1024-epi-recovery-20260925`.
Its detached recovery runner uses the existing worker/lane commands, then runs
the deferred VICReg full evaluations using the original frozen code. Batch,
microbatch, data, seed, epoch budgets, objectives and context policy are unchanged.
The parent run's `technical/epi-recovery.json` points to its separate frozen
config, launch receipt and output. The failed parent Epi receipt is preserved;
parent `lane-1.json` must not be interpreted as the recovery's live status.

## Physical treatment discontinued

At the user's request, Slurm step 1009176.1 was cancelled on 25 September 2026
at 22:27 Paris time. Physical pretraining stopped with its update-43,264
checkpoint preserved (3.1513 epochs); its two encoder continuations and four
context predictor fits are cancelled. W&B records `training/status=stopped_by_user`.
The original frozen configuration remains historical evidence, superseded for
dispatch by `technical/physical-cancellation.json` in the parent output.

The remaining scratch jobs run in detached step 1009176.5 on GPU 0, using their
original frozen configuration. `technical/scratch-continuation.json` records
the launcher, and `scratch-continuation-state.json` records live progress.
It trains both scratch encoders and both context heads per domain, overlaps CPU
scoring with the next fit, then runs full encoder evaluations. Epi continues
independently in step 1009176.4 on GPU 1. Completed VICReg results are preserved;
its deferred diagnostics follow the Epi recovery. The retained scientific
comparison contains 12 context fits across scratch, VICReg and Epi-variance.
