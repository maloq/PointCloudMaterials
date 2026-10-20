# Ni spatial-scale kinetic comparison

This workflow adds observed regional context to the existing Ni kinetic model.
It reuses the central measurement bank, source split, birth assay and evaluation
definitions. The [scientific protocol](../experiments/ni_spatial_scales_20261015/README.md)
states the frozen comparison and its interpretation limits.

- Recipe: [`configs/ni_latent_sim/spatial_scales_20261015.json`](../configs/ni_latent_sim/spatial_scales_20261015.json).
- Implementation: `src/research/ni_spatial_scales/`, using shared kinetic training and evaluations.
- Result root: `${storage:training_storage}/ni_spatial_scales/radius-20261015-v1`.
- Derived cache: `${storage:cache}/ni-spatial-scales/radius-20261015-v1`.
- Registry entry: `ni-spatial-context-20261015`.
- Metric definitions: [`docs/metrics/ni_spatial_scales.md`](metrics/ni_spatial_scales.md).

## Pinned inputs and new collection

The base is `${storage:training_storage}/ni_kinetic_latent/native-20261015-v1`,
with frozen plan SHA256
`ec26712af4b0113bea6c4b5dd95df0bc6baf200b33e251951bea29e8d03094be`.
Its 14,400 native sequences and 244-channel targets are unchanged. The separate
birth history bank retains 19,652 unique current states / 19,800 assay rows.
All original roles and ancestral identities remain recorded; new fits use
train and test only. See the [parent workflow](ni_kinetic_latent.md) for the
central producer and exact physical annotations.

Regional context reads positions, velocities and periodic boxes from the
existing Ni Mendelev2012 EAM source histories. The relevant union contains
87 sources, 52,578 source frames and 999,971 unique source/frame/center
observations. All radii are collected together. These are deduplicated feature
requests, not new simulations or an enlarged independent source cohort.
Actual saved timelines are checked at exact 0.1-ps offsets. No new neighborhood
filter is applied to the birth assay.

For each center, the producer takes every observed atom inside 8/12/20 Angstrom,
excluding the center, without a nearest-atom cap. It uses center-relative
positions and velocities; periodic minimum images and center subtraction are
calculated in float64, and GPU measurements in float32. The 87 regional values
contain density74, shell-motion9 and affine-motion4. There is no regional TDA
calculation. Explicit valid masks move padding outside compact support, so
padding cannot create spurious central density or motion.

Per-source jobs save `cache/context-sources/<source>/partial.pt` while running,
then `data.pt` and checksum-bearing `complete.json`. Progress is also visible
at `technical/collection/<source>/progress.json`, with completed source frames,
measured centers and read/index/feature timing. Checkpoints are written every
64 source frames and when pausing near a deadline. Overlapping native and birth
requests share the same measured source/frame/center values.

Assembly creates `cache/context.pt` and `context.json`, with
`technical/context-complete.json` as the result-side readiness receipt. Native
arrays have shape [14400,64,3,87], and unique birth histories [19652,4,3,87].
Each radius has a separate train-fitted affine/asinh/affine transformation.
Only the 3,840 native training windows enter those statistics; the transforms
are then applied unchanged to test and birth observations. Checkpoints retain
the exact input normalization alongside the central target normalization.

All model inputs have 331 entries. The local arm appends 87 exact zeros to
central244 and does not wait for context collection. Other arms append the
selected radius's normalized87. Decoder targets remain central244. Models
never receive box vectors, temperature, age, absolute time or phase labels.
Physical labels are reused from the base annotation cache and serve evaluation
only. Missing annotations do not prevent scientific fitting or birth readouts.

## Detached execution

Use conda `pointnet-torch214`. The maintained entry point is:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.ni_spatial_scales.queue launch --config configs/ni_latent_sim/spatial_scales_20261015.json
```

Launch freezes the source/configuration bundle and records
`technical/submission.json`. Four GPU workers share one absolute 12-hour
deadline: one starts detached in the current allocation, and three request
compatible RTX6000PRO/A100/A40/L40S partitions as alternatives. Submission
does not depend on a future chat review. Do not relaunch a live queue to inspect
it; use its existing submission and progress receipts.

The three zero-context local fits are ready immediately. Other work includes
independently resumable source collection, context assembly, nine radius fits,
native trajectory evaluations, actual-birth readouts and shared central
controls. Worker locks prevent duplicate stages. Each scientific fit contains
12 representation, 12 fresh-decoder and 12 dynamics epochs at batch/microbatch
1,024. The fixed final checkpoint is the reported model; a partial checkpoint
is not promoted to a completed treatment.

Bounded recovery allows at most two attempts per stage, retains failure receipts
and continues other independent work. Training checkpoints preserve stage,
optimizer and RNG state. An allocation continuation may resume within the
original absolute deadline; it does not extend that deadline. Only jobs named
in this launch's receipts belong to this queue. The earlier kinetic experiment
and its frozen source bundle/results remain separate and unchanged.

Scientific pipelines use online W&B in `teshbek/PointCloudMaterials`, with
stable `niscales-` run IDs and local `wandb.json` receipts. Debug/integration
checks, collection, frozen probes and comparisons remain local. Final metrics
update the associated existing scientific run; tracking failures are recorded
instead of silently switching offline.

## Timing evidence and initial status

Launched October 15, 2026 at 14:49 CEST with an absolute deadline of October 16
at 02:49 CEST. The current node60 allocation 1041402 runs the local RTX PRO6000
worker; remote workers 1041564 and 1041566 started on A40 and L40S GPUs. Initial
lane 2 job 1041565 failed on node34 before Python started (zero runtime, Slurm
signal 53, no worker receipt). Replacement 1041578 uses the same frozen bundle
and original deadline, excluding node34 for this one retry. The original failure
and replacement command are preserved under `technical/`.

The first live verification found all three local-control fits complete at
7,308 updates each, with final checkpoints and online W&B receipts. Regional
collection had completed six source packets and was continuing on other
sources. Regional fits and scientific evaluations were still pending; these
initial training completions are not evidence that a radius is better.
Source/shape/invariance and interruption-recovery integration checks passed
before launch. Current stage receipts supersede this launch-time status.

The [context benchmark](/store/PERSO/vmorozov/experiments/ni_spatial_scales/radius-20261015-v1/technical/context-benchmark.json)
samples three training-source frames on an RTX PRO6000. Warm-cache frame times
are about 0.042 seconds for 56 centers and all three radii. Extrapolating the
measured frame rate gives approximately **0.62 single-GPU hours for collection**,
excluding source startup, checkpoint writes and assembly. Shared-storage reads,
per-frame center counts and contention can materially change that estimate;
it is not an ETA for training and evaluation or a completion guarantee.

The operational receipts are under the result root:

- `technical/context-benchmark.json`: sampled frame timings, atom counts, symmetry checks and the qualified collection extrapolation.
- `technical/context-local-preflight.json`: exact local central inputs, zero context and native/birth dimensions.
- `technical/preflight-trainer/benchmark-local.json` and `receipt.json`: local training integration, stage resume and finite sampled rollouts; debug evidence only.
- `technical/submission.json`: actual jobs, origin allocation, start and absolute deadline.

The feature benchmark observed roughly 179/613/2,848 atoms per ball at the
three radii in its sampled liquid frames. Those are examples, not a uniform
dataset support size. The protocol uses every eligible observed atom.

## Reading outputs

Read per-arm `analyses/<scale>-s<seed>/technical/progress.json`, `last.pt`,
`final.pt` and `complete.json` for training. Independent markers are
`evaluation-complete.json` for native trajectories and
`birth-evaluation-complete.json` for actual birth readouts. Top-level
`technical/queue-finished.json` distinguishes complete from partial queues and
lists pending stages. A training-complete marker alone does not establish
successful generation or a complete comparison.

Each arm publishes named scientific tables and plots under its analysis bundle,
with `tables/METRICS.md` and implementation hashes. The comparison groups by
spatial scale and pairs each radius with local using identical seeds, rows and
source units. It reports liquid retention, phase, path and actual-nucleus
metrics separately. Explicit context331/history1324 readouts distinguish
available regional information from encoded information; central244/history976
controls remain unchanged.

Autonomous generation uses only the initial four context observations. The
retrospective encoded-MD control alone uses actual future observations and is
clearly labeled. Raw inverse-transformation overflow produces undefined raw
order metrics and a failure receipt while preserving finite normalized/phase
scores. Nonfinite autonomous generation records a generation failure without
discarding valid current-state retention or independently completed birth
results. Missing seed/source results remain visible in comparisons.

These outputs concern independent central measurement paths and a separate
full-cell nucleus-prediction assay. They do not demonstrate a jointly generated
atomistic nucleus. Wider context may reveal an existing outer crystal; a
liquid-precursor interpretation requires the additional audits described in
the scientific protocol.
