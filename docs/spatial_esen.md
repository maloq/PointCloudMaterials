# Spatial architecture comparisons

## TDA component difficulty, October 8

The completed MACE L3/C3, MACE L2/C2 and small-eSEN final checkpoints have a
local frozen-head breakdown in
`output/spatial_architecture/tda-difficulty-20261008-v1/analyses/descriptor-difficulty/`.
All three ran 12 epochs. The original validation panel contains 4,096 **Al**
patches from 15 sources/ancestors, following mixed-material pretraining.
No probe is fitted; inference reuses the trained nonlinear heads and each run's
frozen source, original BF16 settings and jitter views. Logged aggregate losses
are reconstructed within 9e-6 absolute error.

Only 113 of 258 TDA outputs have nonzero training weight. Difficulty is ranked
by validation R2 over those active outputs, with normalized MSE and the
training-mean baseline score retained alongside it. The 18 block rankings agree
strongly across runs (Spearman 0.988–0.992). The 80-atom H0 death spectrum is
easiest as a group; the 32-atom H1 Betti curve is hardest. Individual long-lifetime
image bins and maximum H0 lifetime can be much harder than their block averages.
These are descriptive single-seed results, without a statistical superiority claim.

Implementation: `src/research/spatial_architecture/descriptor_audit.py`.
Run this file directly with stage `infer`, `--run`, `--architecture`, `--label`,
`--cache`, and `--output`; this ensures model imports come from the frozen run.
Then use stage `analyze` with the same cache/output, `--labels`, and the original
rich-descriptor `--descriptor-manifest` to verify the validation population.
The saved source revisions and per-model receipts in `technical/` reproduce this
export; [metric definitions](metrics/spatial_tda_difficulty.md) describe its scores.
Publication-only rendering improves the layout without recomputing any metrics:

```bash
python -m src.research.spatial_architecture.descriptor_plot output/spatial_architecture/tda-difficulty-20261008-v1/analyses/descriptor-difficulty
```

### TensorNet progress checkpoint

The October8 progress query captured update6144 atomically while training
continued. Its 1.488-epoch TDA audit is in
`output/spatial_architecture/tensornet-20261008-v1/analyses/tda-progress-u6144/`.
It uses the same original4,096 Al panel and frozen TensorNet producer; there is
no refitting. The combined descriptor loss reconstructs the original validation
value within5.4e-8. TDA-only block-balanced MSE is0.213061, pooled R2 is0.627212,
and error reduction versus the fitting-mean baseline is73.24%.
This intermediate checkpoint is not a matched12-epoch architecture comparison.

`src/research/spatial_architecture/descriptor_progress.py` provides separate
`infer` and `summarize` stages, taking `--output` and `--cache`; inference also
takes the original `--run`. The output's `technical/checkpoint.pt` and
`snapshot.json` bind the requested update to its validation record before any
inference. The frozen script, checkpoint and predictions are retained there.
[Progress metric definitions](metrics/spatial_tda_progress.md) preserve the
earlier final-checkpoint export and label actual epochs separately from planned
epochs. This frozen diagnostic creates no W&B run and changes no training job.

## TensorNet replacement, October 8

At the user's request, pending eSEN-3.2M job **1028497** was cancelled before it
started, using its current-chat launch receipt. Its recipe and frozen source
remain preserved; `technical/replacement.json` in the eSEN output records the
action. MACE L2/C2 job **1028442** was not changed.

Replacement recipe: `configs/spatial_architecture/tensornet_20261008.json`.
TensorNet uses 128 channels, two O(3) Cartesian-tensor interaction layers, 32
fixed exponential-normal radial functions, a 5 Å edge cutoff, the established
8 Å nearest80 support and a 128-dimensional exported embedding. It has
**828,160 encoder / 1,003,690 total training parameters**. Scalar export uses
the common center-plus-tapered-mean pooling and projection. All atoms share one
learned constant embedding; the center flag is the only additional varying
feature. There are no species, material IDs, charge, temperature or velocity
inputs. Fixed per-material length normalization remains preprocessing.

The vendored TorchMD-Net v2.4.11 definitions are pinned to revision
`7e22a231f34a855f894daaeb218c8b393068f846`, with MIT license, source hashes and
verbatim extracted block/radial definitions. `spatial_tensornet.py` adapts input
and pooling, keeps normalization reductions in FP32 under BF16 neural matmul,
and constructs differentiable distances with ordinary PyTorch operations.
Upstream TensorNet self edges and cosine cutoffs remain explicit. No specialized
neighbor kernels or pretrained force-field weights are loaded; the shared
environment is unchanged.

The data, seed, heads, objective, optimizer, batch=microbatch256 and 12-epoch
schedule match the spatial comparison: 1,056,768 rows, 49,536 updates, one-epoch
warmup, peak LR0.001 and cosine decay. Execution compiles the tensor blocks and
uses direct backward. The queued worker first runs local numerical/precision,
encoder-derivative and full-batch timing probes, then starts online scientific
training. Derivative probes use the eager FP32 encoder and are not a full motion
training run or an end-to-end response-training speed measurement.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.spatial_architecture.workflow queue --config configs/spatial_architecture/tensornet_20261008.json --after-job 1028442
```

The single RTX6000PRO job requests up to 24 hours and can resume its own saved
optimizer/checkpoint if needed. Results belong to
`${storage:training_storage}/spatial_architecture/tensornet-20261008-v1`.
CPU real-data audits are in `technical/architecture-audit.json`; GPU checks and
timings are recorded in `technical/probe.json` once its allocation starts.

On October 8 at 14:25 Europe/Paris, the user requested using the current
allocation instead of waiting for a new GPU. A detached supervisor (PID3713429)
now runs the existing frozen probe/training workflow in node60 allocation
1028490, sharing GPU0 with its existing Ni simulation. No simulation or allocation
settings were changed. This run retains its exact 12-epoch recipe and online
tracking; timings during sharing are not exclusive-GPU speed measurements.
`technical/local-allocation/launch.json`, `state.json`, `probe.log` (one directory
above) and `training.log` record the handoff and live phase.

The receipt-owned pending TensorNet job1028528 now has dependency
`afterany:1028490`, so it can resume the same checkpoint if the local allocation
ends. The supervisor cancels only this unused successor if training completes
locally. The previous launch receipt is preserved in
`technical/local-allocation/original-launch.json`; orchestration is
`src/research/spatial_architecture/allocation.py`, copied into that directory.

GPU checks passed and online training reached update50 at 14:28. The eager FP32
encoder JVP finite-difference relative error was 0.000761; all nested derivative
parameter gradients were finite. The co-scheduled probe median was 0.7247 s/update
with 17.64 GiB peak allocated memory; early training updates were about0.69 s.
The online run is [spatial-5490509e5fc6c8ff](https://wandb.ai/teshbek/PointCloudMaterials/runs/spatial-5490509e5fc6c8ff).
These throughput measurements include sharing with Ni and should not establish
an architecture speed ranking against the earlier exclusive-GPU measurements.

Submitted **TensorNet job 1028528** with `afterok:1028442`. At submission it
was pending on the running MACE job. CPU audits passed invariance, forward-AD
versus reverse JVP agreement, central differences, FP32/FP64 nested parameter
derivatives and BF16 execution. FP64 JVP finite-difference relative error was
6.18e-8; CPU BF16 embedding relative RMS difference was 1.42%. GPU checks remain
pending. The replacement receipt links cancelled eSEN job1028497 to this job.

## eSEN-3.2M materials preset, October 8

**Superseded before training:** the requested replacement is TensorNet above.
The following records the preserved original submission.

Recipe: `configs/spatial_architecture/esen_3m2_20261008.json`. This uses the
MPTrj-3.2M architecture from [the paper's Table 6](https://arxiv.org/html/2502.12147v2#A4):
two blocks, widths128/128/128, L2/M2, 6 Å cutoff and ten Gaussian radial functions.
The existing geometry-only local80 wrapper exports 128 features; measured counts
are **3,211,392 encoder / 3,386,922 total training parameters**. This is a spatial
adaptation trained from scratch, retaining our 1,056,768 preserved spatial rows,
VICReg/bond-order/TDA objective, batch256, BF16, 12 epochs, one-epoch warmup and
cosine schedule with peak LR0.001. The cutoff and radial basis differ from the
preceding small eSEN, so this is a complete preset comparison.

Queue one GPU after the MACE L2/C2 job:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.spatial_architecture.workflow queue --config configs/spatial_architecture/esen_3m2_20261008.json --after-job 1028442
```

The worker receives one RTX6000PRO for up to 24 hours and resumes its own
checkpoint if continuation is needed. It runs a local symmetry/precision and
speed probe first, then 49,536 scientific updates with online W&B. Full-epoch
timings have the same scope as the MACE and smaller eSEN runs. The output root is
`${storage:training_storage}/spatial_architecture/esen-3m2-20261008-v1`;
`technical/launch.json` records the dependency and owned job, while
`technical/code` preserves the exact executed source. Existing jobs retain
their original frozen settings.

Submitted job **1028497**, with `afterok:1028442`. At submission MACE L2/C2
was training and eSEN was pending on that dependency. The frozen metric contract,
CPU constructor parameter count, schedule/data parity, 6 Å input record and
Slurm script were verified. The GPU numerical/timing probe remains pending until
the allocation starts; the 24-hour reservation is a limit, not a measured ETA.

## MACE angular2/correlation2 ablation, October 8

Recipe: `configs/spatial_architecture/mace_l2_c2_20261008.json`. Train one fresh
geometry-only MACE with angular order 2 and correlation 2. Width128, exported
dimension128, two interactions, batch=microbatch256, BF16/cuEquivariance/direct
backward, seed, 1,056,768 spatial patches, VICReg/bond-order/TDA objective and all
optimizer/schedule settings match the ongoing L3/C3 MACE reference. Both receive
49,536 updates (12 epochs), with one-epoch warmup and cosine decay from peak
LR0.001 to 0.00001. Changing both orders together does not isolate their effects.

The comparison reference is the MACE identity
`c00ba4468374af9d4550bae306ce1c3fc8ce453c6955e7ba30cb32423f4227dd`
in `esen-small-20261007-v3`. Original runs and their frozen code remain unchanged.
The new result root is
`${storage:training_storage}/spatial_architecture/mace-l2-c2-20261008-v1`.

Submit after the current scientific job finishes, without sharing its busy GPU:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.spatial_architecture.workflow queue --config configs/spatial_architecture/mace_l2_c2_20261008.json --after-job 1027882
```

This freezes source and provenance, then submits one six-hour RTX6000PRO job.
The assigned worker runs the local numerical/timing probe first, then online
scientific training. It can requeue only its own recorded job and retains the
optimizer/checkpoint on continuation. The `afterok` dependency leaves a failed
reference explicit rather than starting a comparison without its reference.
Submitted job **1028442**, dependent on successful completion of **1027882**.
The CPU constructor audit verified angular2/correlation2, width128, export128
and **635,008 encoder / 810,538 total training parameters**. Numerical producers
and all non-order model settings match the pinned reference. The GPU numerical
probe and timing run when the queued allocation starts.

The queue ID is recorded in `technical/launch.json`; `queued-stage.json` separates
the probe from fitting. Final tables compare validation components, descriptor
prediction error, embedding rank, parameters and full 12-epoch training time in
`analyses/order-comparison/`, with frozen metric definitions and source hashes.

## eSEN recipes

The active smaller-model recipe is
`configs/spatial_architecture/esen_small_20261007.json`: eSEN C64/H32/E32,
lmax=3, mmax=2, two blocks, output128. It has 809,600 encoder parameters, 18.86%
fewer than MACE128's 997,760. Including the common heads, totals are 985,130 and
1,173,290. It uses the same batch256, optimized execution, data, objective and
12-epoch schedule. Results belong to
`${storage:training_storage}/spatial_architecture/esen-small-20261007-v3`.
Use `esen_small_20261007.json` with the probe/prepare/launch commands below for
this revision. The v2 worker was stopped from its owned launch receipt after
saving update192; its wider-model checkpoint remains intact. Its replacement
receipt is `technical/small-model-replacement.json` in the v2 output.

The preceding v2 recipe uses GPU-efficient, architecture-appropriate widths rather than
matching parameter counts, as requested on October 7. Configuration:
`configs/spatial_architecture/esen_efficient_20261007.json`. Both fresh models
receive the same 1,056,768 preserved unrelaxed spatial patches, VICReg +
bond-order + TDA objective, batch=microbatch=256, and 12 complete epochs. Existing
completed MACE checkpoints remain unchanged. This is spatial pretraining.

MACE uses width 128, correlation 3, angular degree 3, two interactions and the
fused cuEquivariance backend: 997,760 encoder parameters. eSEN uses spherical,
internal and edge widths 128, lmax=3, mmax=2, two gated spectral blocks and 32
Gaussian radial functions: 6,188,672 encoder parameters. Both export 128 scalar
features from the same geometry-only local80 support and pooling. eSEN has no
MACE correlation-order parameter. Widths 64 and 128 were benchmarked locally;
width 128 has about twice the parameters for 27% more unoptimized update time.
This is an efficiency-informed choice, not an exhaustive accuracy search.

The eSEN blocks follow the [paper](https://arxiv.org/abs/2502.12147), pinned to
FAIR revision `977a80328f2be44649b414a9907a1d6ef2f81e95`.
Upstream sources, hashes and MIT license are retained in
`src/models/encoders/esen_vendor/`. The wrapper is
`src/models/encoders/spatial_esen.py`; execution changes are in
`src/models/encoders/esen_runtime.py`. The accelerated eSEN compiles tensor blocks,
replaces dense one-hot permutations with indexing, and keeps both view graphs
for direct backward. Numerical audits compare against identical eager parameters,
data and augmentation; the loss and sample count are unchanged.
See [metric and timing definitions](metrics/spatial_architecture.md).

Use conda `pointnet-torch214`, with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the
installed e3nn/MACE versions:

```bash
python -m src.research.spatial_architecture.workflow probe --config configs/spatial_architecture/esen_efficient_20261007.json
python -m src.research.spatial_architecture.workflow prepare --config configs/spatial_architecture/esen_efficient_20261007.json
python -m src.research.spatial_architecture.workflow launch --config configs/spatial_architecture/esen_efficient_20261007.json
```

`probe` performs real-data symmetry/precision checks and synchronized timing;
it creates no W&B runs. `prepare` binds packed data, ancestry, diagnostic selection,
completed reference and implementation hashes. `launch` freezes execution source,
starts one detached local GPU worker and reserves one successor RTX6000PRO after
the current allocation. It resumes the same optimizer and checkpoint before the
second architecture. Only its own recorded successor can be requeued.

The main output is
`${storage:training_storage}/spatial_architecture/esen-l3-efficient-20261007-v2`.
Each `analyses/<architecture>/technical/` contains progress, last/final checkpoints,
online W&B receipt, validation and completion. Epoch and learning tables retain
metric definitions and implementation hashes. `technical/launch.json` owns the
local PID and successor; `technical/code` is the executed snapshot. Timing excludes
validation, checkpoint I/O and queue delay; lazy compilation during an update is
included in actual epoch totals. Probe extrapolations use warmed timings and are
estimates, not completed 12-epoch measurements.

## Preserved capacity-matched pilot

`configs/spatial_architecture/esen_l3_20261007.json` and
`${storage:training_storage}/spatial_architecture/esen-l3-20261007-v1` retain the
original eSEN widths 128/16/32, lmax=mmax=3, and 970,592 encoder parameters.
Its measured eager update times were 2.59795 s for eSEN and .323945 s for MACE.
The scientific eSEN pilot reached a saved update-64 checkpoint; it was paused
for the user's optimization request, then superseded by architecture-appropriate
widths. Its launch-owned local PID 3602664 was stopped and its pending continuation
job 1027829 cancelled. Receipts are `analyses/runtime-optimization/technical/pause.json`
and `technical/pilot-superseded.json`. This pilot is not an active 12-epoch run.
Its online record remains
[spatial-1b357bc999b7e6aa](https://wandb.ai/teshbek/PointCloudMaterials/runs/spatial-1b357bc999b7e6aa).

## Local execution benchmarks

All rows below use RTX PRO 6000 Blackwell, full batch 256 and BF16 neural matmul.
Eight-update optimization checks discard three startup updates:

| Encoder | Execution | Median update | Peak allocated GiB |
| --- | --- | ---: | ---: |
| eSEN 128/128/128 | eager replay | 3.321 s | 40.74 |
| eSEN 128/128/128 | compiled, indexed, direct backward | 1.542 s | 47.71 |
| MACE 128 | fused cuEq, replay | 0.324 s | 5.57 |
| MACE 128 | fused cuEq, direct backward | 0.251 s | 9.22 |

Identical-checkpoint optimized-versus-eager relative L2 differences were 0.202%
for eSEN embeddings and 0.309% for its gradients; MACE differences were 0.000015%
and 0.0380%. Gradient comparison is before clipping. These are BF16 numerical
agreement checks, not a claim of bitwise equality or completed training quality.
The complete measured variants are in
`analyses/runtime-comparison/tables/benchmarks.csv`, with metric definitions and
implementation hashes. The final 14-update probe uses the training row order,
discards four startup timings and records a separate median in `technical/probe.json`.

## Wider v2 launch (superseded)

Detached local worker PID **3609921** runs on node60/GPU0 in allocation
**1026285**. Its recorded successor is **1027867**, requesting
one RTX6000PRO for 24 hours after the current allocation. The worker runs eSEN
then MACE, continuing saved optimizer state if needed.

The final esen probe measured 1.475893 s/update,
101.54 min/epoch and 20.31 h for
12 epochs, excluding validation/I/O/queues.

The final mace probe measured 0.251569 s/update,
17.31 min/epoch and 3.46 h for
12 epochs, excluding validation/I/O/queues.

These are launch and benchmark records. Model quality and actual full-epoch
time must be read from the completed scientific results.

Online eSEN run: [spatial-ed4ea994645e2c1c](https://wandb.ai/teshbek/PointCloudMaterials/runs/spatial-ed4ea994645e2c1c).
Scientific update 1 completed on the detached worker; MACE follows after eSEN.

## Smaller-than-MACE speed result

The new eSEN widths 64/32/32 give **809,600 encoder parameters**, versus **997,760**
for MACE. The optimized eight-update benchmark measured **0.63646 s/update**,
**20.70 GiB** peak allocated memory, and **8.76 h** extrapolated for 12 epochs.
The same benchmark's earlier wider eSEN and MACE measurements were 1.54233 and
0.25077 s/update: the smaller eSEN is 2.42 times faster than the wider one and
2.54 times slower than MACE on this GPU. All use batch256/BF16 and the same rows,
view seeds and objective; the earlier reference JSONs are retained and linked
with hashes in the new runtime-comparison analysis. Its optimized/eager gradient
relative L2 difference was 0.320%; embedding difference was 0.283%.

Only the launch-owned v2 worker and its pending successor 1027867 were stopped.
The v2 update192 checkpoint and its training recipe remain preserved; v3 replaces
that wider model in the requested 12-epoch comparison. No other jobs were changed.

## Smaller v3 launch

Detached worker PID **3612031** runs on node60/GPU0 in allocation
1026285. Continuation job **1027882** resumes the checkpoint
when this allocation ends; MACE follows the smaller eSEN.

The final 14-update probe measured 0.635451 s/update for smaller eSEN and
0.250298 s/update for MACE, corresponding to 8.74 and 3.44 hours of warmed
training compute for 12 epochs. It passed rotation, translation, permutation,
repeat and mixed-precision checks. Model quality remains to be measured.

Online scientific run: [spatial-352a339ba7874cef](https://wandb.ai/teshbek/PointCloudMaterials/runs/spatial-352a339ba7874cef).

## Ni-enriched Small eSEN, October 8

Recipe: `configs/spatial_architecture/esen_small_ni_20261008.json`.
Dataset: the completed Ni-enriched `spatial-training-ni20` release documented
in [spatial_ni_enriched.md](datasets/spatial_ni_enriched.md), identity
`0325195ca0b4e060f6a41390823265b58f65267a3ae130148ffff28c4b688162`.
This is a fresh fit of the established 809,600-parameter Small eSEN. It uses
1,373,798 training patches (317,030 Ni), 198 BO/CNA/TDA outputs (197 active),
12 full epochs, batch=microbatch256, BF16 matmul, one-epoch warmup, peak LR
0.001 and cosine decay to 0.00001. All model inputs remain geometry-only.
The new dataset's training moments are applied to new pooled and Ni monitoring
panels; the final epoch is fixed without held-out selection.

Submit an independent GPU with:
```bash
python -m src.research.spatial_architecture.workflow queue --config configs/spatial_architecture/esen_small_ni_20261008.json
```
There is no dependency on TensorNet. One worker requests compatible GPU types
as alternatives, runs the numerical/timing probe, then trains with online W&B.
Checkpoint continuation preserves the same run and optimizer. Full provenance,
launch receipts and results are under
`output/spatial_architecture/esen-small-ni-20261008-v1/`.
This changes both training data and targets versus the original spatial runs;
it is not an isolated architecture or Ni-data ablation.

Launch receipt: job **1028962**, independently allocated **one L40S on node50**
at 16:25 CEST on October 8. No dependency on TensorNet and no local GPU worker
were created. Its initial 24-hour allocation supports checkpoint/requeue if the
12-epoch fit needs more time. Scientific results remain pending during startup.
TensorNet continues with its original frozen dataset/targets on node60.

The L40S startup probe passed rotation, translation, permutation, repeated
execution, BF16 and compiled-execution checks. Its warmed median was about
0.589 seconds/update, projecting 10.54 hours of training compute plus validation,
checkpointing and startup. The online run initialized at 16:28 CEST:
[spatial-3100cd6134c60489](https://wandb.ai/teshbek/PointCloudMaterials/runs/spatial-3100cd6134c60489).

Completion, October 9: all 12 epochs and 64,404 updates finished. The final
pooled descriptor half-MSE is 0.08531; the Ni panel is 0.11120. Embedding
participation ranks are 99.30 and 90.06 of 128 respectively. These are structural
monitoring diagnostics, not clustering or nucleus-prediction results. Training
compute took 10h40m, with about 17 additional minutes for validation/checkpoint
I/O. Final results are in `analyses/esen/technical/complete.json` under the
recorded output above.

## Relaxed/static Small eSEN retry, October 9

All 2,670,500 relaxed/static target rows are prepared, including all 6,571 Ni
birth-window relaxations. The 1,339,157-row training subset retains its original
roles. See [paired relaxed data](datasets/spatial_paired_relaxed.md). The first
training launch (1029531) failed before fitting because the local symmetry probe
rotated coordinates inside BF16 autocast. The corrected probe rotates in FP32;
neural arithmetic remains BF16. The same-row compiled rotation error is 0.380%
instead of 3.03%, passing the unchanged 2% threshold. This fixes the audit, not
the model or training objective.

Retry recipe: `configs/spatial_architecture/esen_small_relaxed_static_20261009.json`.
Output: `output/spatial_architecture/esen-small-relaxed-static-20261009-v3/`.
The packed data and monitoring panels are reused exactly; 12 epochs, 62,784
updates, batch256, one-epoch warmup and cosine LR .001 to .00001 are unchanged.
The frozen failed launch is retained for provenance.

```bash
python -m src.research.spatial_architecture.workflow queue --config configs/spatial_architecture/esen_small_relaxed_static_20261009.json
```

The retry started October 9 at 11:23 CEST on node58 GPU0, inside existing
allocation 1029087. Receipt-owned successor 1030439 waits for that allocation
to end and is cancelled automatically if the local fit finishes first. The
allocation supervisor changes only this newly submitted successor. The corrected
full-batch probe passed (0.362% compiled rotation discrepancy) and measured
0.2833 s/update, projecting 4.94 hours of update compute before validation,
checkpointing and startup. Online run:
[spatial-5a3167d76429c0b2](https://wandb.ai/teshbek/PointCloudMaterials/runs/spatial-5a3167d76429c0b2).

At the user's subsequent request, this single-GPU execution was stopped after
475 observed updates (0.091 epoch); its checkpoint remains intact. Only its
own worker process group and pending successor 1030439 were stopped. The
two-GPU replacement uses
`configs/spatial_architecture/esen_small_relaxed_static_2gpu_20261009.json` and
`output/spatial_architecture/esen-small-relaxed-static-2gpu-20261009-v4/`.
It starts from the original seeded initialization and retains the exact data,
global batch256, 12-epoch schedule and objective. Two parallel 128-row geometry
forwards feed one global-batch VICReg/descriptor loss and one optimizer. Startup
checks compare full and partial-batch gradients to the single-GPU calculation.
The replacement's local and continuation workers both request two GPUs.

The two-GPU run started at 11:35 CEST on node58 GPUs 0/1, with owned two-GPU
successor 1030471. The benchmark passed the corrected symmetry checks, with
full/partial-batch relative gradient differences of 0.267%/0.778% against the
single-GPU BF16 calculation. Loss differences were 0.0018%/0.0090%. Its median
complete update takes 0.15124 seconds (about 1.87 times the one-GPU throughput),
projecting 2.64 hours of update compute plus monitoring/checkpoints. Peak
allocated memory was 11.09/11.00 GiB per GPU. Online run:
[spatial-ad876c75ab8d9423](https://wandb.ai/teshbek/PointCloudMaterials/runs/spatial-ad876c75ab8d9423).

For this two-GPU configuration, `workflow launch` freezes and queues a two-GPU
successor, then starts the detached worker inside the current Slurm allocation.
The local supervisor cancels only that recorded successor if fitting finishes
before the allocation ends; otherwise it preserves the checkpoint for resuming.
