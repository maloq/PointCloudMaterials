# Spatial architecture and equal-epoch training speed

## Two local GPUs, October 9

The optional `runtime.gpus=2` profile splits each global batch of 256 rows into
128 rows per device, with one geometry replica on each GPU. The two seeded
jitter views are generated for the entire original batch before splitting.
Embeddings are gathered with autograd onto GPU0; the unchanged VICReg statistics
and descriptor mean are computed across all 256 rows. The replica geometry
gradients are summed, not averaged, before the original global gradient clipping
and one primary AdamW update. Updated geometry weights are copied to GPU1 at the
next update. There is one canonical model, optimizer, EMA, checkpoint and online
W&B run; GPU1's replica is disposable. The model's parameter/checkpoint names
and all scientific hyperparameters remain unchanged. The configured microbatch
is the global microbatch; parameter receipts separately record 128 rows/device.

The final 21-row batch is split 11/10 without padding, dropping or duplicating
observations. Startup audits compare the global loss and all used parameter
gradients against single-GPU direct backward, on both a complete real batch and
the actual partial-batch size, at identical weights and view seeds. Finite
gradients, relative loss error <=2%, and gradient relative L2 error <=5% are
required under BF16; raw discrepancies and loss components are retained. These
are execution checks, not scientific optimizer updates or online runs.

Timing synchronizes both devices around the complete update, including view
generation, weight/embedding/gradient transfers, one optimizer update and EMA.
The warmed median still excludes the first four of 14 updates. Per-GPU peak
allocated memory and GPU names are recorded. Monitoring retains the original
GPU0 evaluation implementation and global 256-row statistics. Validation and
checkpoint costs remain separately accumulated. One process dispatches the
geometry replicas through two CUDA-device threads, with sequential initial
compilation. The compiler variant-cache limit is 32 for this profile, to retain
the device, gradient-context and partial-batch specializations of the same
vendor code objects; full-graph compilation still fails on unsupported work.
This profile does not create two independent scientific fits.

## Rotation-probe correction, October 9

The local numerical probe now constructs the rotated coordinates once, outside
autocast in FP32, and supplies that identical geometry to both FP32 and accelerated
forwards. The previous accelerated check evaluated `q @ matrix` inside BF16
autocast. Its reported rotational error therefore included coordinate rounding,
not only encoder symmetry error. On the first eight relaxed/static probe rows,
this introduced 0.00846 A RMS coordinate error (0.03419 A maximum), producing
3.03% compiled embedding discrepancy; rotating in FP32 gives 0.380% instead.
The 2% accelerated symmetry threshold and all training settings are unchanged.
Historical probe records retain their original definition. No training update
was executed by failed relaxed/static job 1029531. The October 9 retry reuses
the sealed 1,339,157-row training cache and monitoring panels without target
recomputation, in a new output/implementation revision.

## Paired relaxed Small eSEN, October 8

The paired relaxed recipe reuses the Ni-enriched Small eSEN architecture and
198-output BO/CNA/TDA objective below, with VICReg on two jittered relaxed views.
Dynamic relaxations require available unrelaxed originals; those originals are
audit-only and are not training inputs or teachers. User-confirmed already-relaxed
static Mg/Al/Zr/Ta are explicitly included without this pairing requirement. Precisions,
duplicate policy, retained ancestry and the pending Ni birth-window population
are defined in `spatial_paired_relaxed.md`. Material length scales are inherited
unchanged; target moments and active masks are fitted anew on relaxed training
rows. All 12 epochs cover every fitting row. One-epoch warmup and cosine decay
retain the same learning rates, with update counts recalculated from the new
population. The 4,096-row pooled and Ni panels are diagnostics only. This is a
new data-domain experiment, not a matched same-population comparison against
the prior unrelaxed run. Earlier recipes and published definitions remain frozen.

## Ni-enriched Small eSEN, October 8

`configs/spatial_architecture/esen_small_ni_20261008.json` starts a fresh
809,600-parameter Small eSEN on the completed Ni-enriched release: 1,373,798
training patches, including 317,030 Ni patches. It retains C64/H32/E32, L3/M2,
two blocks, 128 exported coordinates, full batch256 and BF16 neural matmul.
The head outputs 198 columns: 40 bond-order, 45 CNA and 113 TDA. The release's
train-only moments leave 197 active; `tda/n32_h2_image20` has zero loss weight.
Each family receives one third of descriptor loss weight; TDA is further
equally weighted across its 18 blocks and their active columns. CNA is a
local neighbor fingerprint target, not a supplied phase or nucleus label.

All 12 epochs visit every training row once: 5,367 updates per epoch,
64,404 total, with a final 102-row batch each epoch. Warmup lasts one epoch,
then cosine decay from 0.001 to 0.00001. The final epoch is fixed in advance.
Monitoring uses a seeded uniform 4,096-row sample of the merged 248,081-row
`evaluation` pool and a separate 4,096-row Ni sample. Ni metrics have a `Ni_`
prefix inside validation; both panels use the same train-only moments,
family weights, jitter seeds and objective definitions as the pooled panel.
The panels may overlap and are not independent samples. Their exact row IDs,
material counts and overlap are saved. Neither panel selects checkpoints or
hyperparameters. Al calibration remains outside the merged evaluation role.
The old fixed-Al monitoring packet and target moments are not reused.

The Ni release adds crystal-interior/interface patches using PTM as sampling
metadata; PTM, material, potential, temperature and time never enter the model.
Fixed material length normalization precedes the geometry-only encoder.
Training and evaluation ancestry are checked for disjointness. This changes
the dataset and target definition together and cannot be interpreted as a
matched architecture comparison with the ongoing original TensorNet run.

One independent GPU job requests alternative compatible partitions. If a
continuation changes GPU model, the old numerical/timing probe is archived
and a new local probe must pass before resuming the same optimizer/checkpoint.
Worker receipts record each GPU. The synchronized training-time definition is
unchanged; heterogeneous timing must not be represented as one-GPU speed.
Probe epoch/budget extrapolation uses the actual pinned row count and updates.

This experiment compares a local geometry-only eSEN with the October 7 spatial
MACE, using the same packed 1,056,768 unrelaxed nearest80 Al/Mg/Ti/Ta observations,
ancestry roles, fixed material length normalization and 4096 diagnostic selection
rows. It opens preserved derived patches, not retired raw Al histories. There
are no temperatures, times, species, velocities, explicit conditions, relaxed
teachers, crystal labels or nucleus labels in either encoder/head input.

The original October 7 comparisons expose 128 scalar coordinates, with recipe-specific spherical widths, two message
blocks, maximum angular degree 3, a 5 Å edge cutoff, and 8 Å support without a
halo. Pooling concatenates center scalars and the tapered weighted mean; the
taper is flat through 6 Å and cosine to 8 Å. The LayerNorm/projection/residual
export matches the historical spatial MACE. The MACE remains correlation 3,
with its original six Bessel functions, radial MLP 32, cuEquivariance ir_mul and
fused convolution.

The eSEN imports unchanged low-level FAIR code from revision
977a80328f2be44649b414a9907a1d6ef2f81e95, with its MIT license and file hashes.
Its scalar/edge interfaces use one constant channel and the center indicator.
The v2 recipe uses lmax=3, mmax=2, 32 Gaussian radial functions, polynomial edge envelopes,
spectral SiLU gates (no sampled-grid nonlinearities), equivariant layer_norm_sh,
128 internal hidden channels and 128 edge channels. Widths are chosen for each
architecture, without matching parameter counts: 6,188,672 eSEN versus 997,760
MACE. The superseded v1 pilot used lmax=mmax=3 and internal widths 16/32 for
970,592 parameters; its saved checkpoints and frozen recipe retain that identity.
The v3 smaller-than-MACE recipe retains lmax=3, mmax=2 and uses 64 spherical,
32 hidden and 32 edge channels: 809,600 encoder parameters versus MACE's 997,760.
The common descriptor and unused masked heads add 175,530 parameters to each,
for totals of 985,130 and 1,173,290. Support, angular order, block count, exported
embedding, objective and batch size are unchanged; width is an explicit capacity
ablation. The wider v2 checkpoint is preserved as a superseded partial run.
eSEN has no MACE
correlation-order setting. Its initial degree embedding adds an edge aggregation
before the two blocks; equal block count is not an assertion of equal body order
or identical receptive-field computation. A deterministic least-aligned-axis
edge frame replaces the upstream random gauge for reproducible gradient replay.
The original Wigner conversion/pole handling is retained. This local encoder is
not a pretrained energy model, and no energy-conservation result is claimed.

The common objective is the existing `vicreg_descriptors` implementation:
25 times mean-square two-view invariance + 25 times variance penalty + covariance
penalty + descriptor half-MSE. Views independently jitter each coordinate by
0.01 Å and recenter. VICReg acts directly on the exported 128-vector. For unbiased
batch covariance C, variance is mean(relu(1-sqrt(diag(C)+1e-4))) and covariance is
sum(offdiag(C)^2)/128, both averaged over the two views. The descriptor head is
128→256→298 with SiLU. It predicts the original 40 bond-order and 258 TDA values
using existing training-only standardization. Half the weight belongs to bond
order, half equally across the 18 TDA blocks, then equally across active columns
within each block. `descriptor_half_mse` is half this weighted MSE averaged over
the two views. The unused masked head/EMA are retained identically for parity
with the historical runner; `masked_half_mse` is zero.

New compact-target recipes may declare `descriptors.selection`: a checksum-bound
training-only active-column contract. The spatial cache and head then expose only
those named columns (153 for the current fit), with the same half bond-order/half
18-block TDA weights and original target definitions. The current mask removes
only zero-weight outputs. Training/checkpoint metadata records the resulting
column names and weights. The 12-epoch results described above retain their
original 298-output heads; no existing checkpoint is converted or relabelled.
The new CNA-inclusive selection has 198 outputs: 40 bond order, 45 CNA and 113
TDA. For this explicitly different objective each family receives one third of
the descriptor weight, with TDA still divided equally among its 18 blocks and
then active columns. CNA is the existing local bond fingerprint, not a crystal
label. A new derived panel/head is required; frozen comparisons are unchanged.

Each model uses 12 full epochs: 4128 updates per epoch, 49,536 updates total,
batch=microbatch=256. Both receive identical seeded epoch permutations and jitter
seeds. AdamW uses one-epoch warmup, peak LR .001, cosine decay to .00001, weight
decay 1e-5 and gradient clipping at norm 5. The exported checkpoint is fixed at
epoch 12, never selected by AP or validation. BF16 autocast is used for neural
operations; geometry/Wigner construction, scalar export, losses, optimizer and
normalization remain float32. TF32 is disabled. The v1 pilot computes full-batch
latent cotangents before encoder graph replay. V2 records the execution variant
for each model explicitly: direct backward retains both view graphs and avoids
the two additional no-gradient forwards. Both use full-batch VICReg statistics.
The compiled eSEN variant uses dynamic PyTorch compilation of degree embedding,
message blocks and normalization; graph construction remains eager. Its indexed
variant replaces two dense one-hot matrix permutations with index selections,
retaining the original BF16 casts. Parameters and learned operations are the same.
Independent identical-checkpoint audits compare embeddings and gradients against
eager replay and record their BF16 differences. These are numerical tolerances,
not a claim of bitwise equality. Runtime variants do not change epoch count,
sampling, augmentation, loss or optimizer settings.

`learning.csv` records objective components, unclipped gradient norm, epoch,
learning rate, and `update_seconds`. Timing begins before CPU batch gathering
and GPU transfer and ends after backward, clipping, AdamW and EMA, bracketed by
CUDA synchronization. Validation, checkpoint writes, model initialization, W&B
logging, queue delay and allocation idle time are excluded from this field.
Lazy compilation within an update is included in that update and epoch total;
the warmed hardware benchmark excludes its first four updates. Compilation of
validation graphs, when needed, is included in validation time.
`epochs.csv` sums the observed update timings in each complete epoch; rows count
reused observations, not independent samples. Validation uses the same objective
plus mean/minimum embedding standard deviation and participation rank
(sum eigenvalues)^2 / sum squared eigenvalues of the sample covariance.

Technical timing receipts separately accumulate validation and checkpoint I/O.
Checkpoint state contains time measured before its own serialization; a resume
therefore omits the last checkpoint's serialization time from that diagnostic.
Training update and epoch times remain exact sums of retained executed updates.
Both full runs use RTX PRO 6000 Blackwell, with all compute-segment receipts kept.
The local hardware probe times the same first 14 full updates for both models,
discarding four warmup timings and taking the median of ten. Its epoch/12-epoch
extrapolations are estimates, excluding validation and I/O, and must not be
reported as completed training totals. Probe runs remain local; scientific fits
use online W&B with stable resumable identities. One seed is exploratory.

The preliminary execution-optimization benchmark in `src/hardware_benchmark/esen.py`
uses eight full updates and the median of the final five. Execution variants for
one architecture restore identical model/optimizer state, rows and view seeds.
Its first-update gradient comparison is relative L2 error before clipping;
embedding comparison uses eight observations before the first optimizer step.
Peak memory is CUDA's maximum allocated bytes divided by 2^30 (GiB, despite the
historical JSON field name `peak_memory_GB`). `update_seconds[0]` also includes
local gradient/embedding capture; this update is excluded from the warmed median.
Width comparisons are initialization benchmarks, not completed model quality
comparisons. Final equal-epoch timings are produced by the scientific fits.

## MACE angular/correlation ablation, October 8

`configs/spatial_architecture/mace_l2_c2_20261008.json` changes the MACE maximum
angular degree from 3 to 2 and correlation from 3 to 2 together. Channels and
exported embedding remain 128, with two interactions, the same radial basis,
support/pooling, constant atom channel, BF16 cuEquivariance/direct execution,
training rows, validation observations, seed, losses, optimizer and 12-epoch
schedule. The pinned reference is the MACE model in `esen-small-20261007-v3`.
This is a joint order ablation, not a separation of the two orders' effects.

`analyses/order-comparison/tables/final_models.csv` compares both fixed epoch-12
checkpoints using their existing validation metrics, parameter counts and actual
summed update times. `mean_update_seconds` includes lazy compilation within
updates and excludes validation, checkpointing and queue time. Source completion
and checkpoint hashes accompany the table; no metrics or checkpoints are chosen
using AP. The summary's timing and descriptor-error ratios are L2/C2 divided by
L3/C3; values below one respectively mean less training time and lower descriptor
error. Lower pretraining loss or greater embedding rank is not by itself evidence
of improved nucleus prediction. The GPU probe runs locally in the queued
allocation before online scientific training; it creates no W&B run.

## eSEN-3.2M materials preset, October 8

`configs/spatial_architecture/esen_3m2_20261008.json` adapts the MPTrj-3.2M
architecture in [Table 6 of the original paper](https://arxiv.org/html/2502.12147v2#A4):
two blocks, 128 spherical channels, maximum degree/order 2/2, 6 Å cutoff and ten
Gaussian radial functions. Hidden and edge widths are 128. Gated spectral SiLU,
layer_norm_sh and polynomial envelopes use the existing pinned eSEN blocks.
The constructor has 3,211,392 encoder parameters and 3,386,922 including the
common heads. These are measured counts of our spatial adaptation.

The geometry-only constant atom/edge interface, center indicator, deterministic
edge gauge, nearest80 crop, 8 Å support, taper and 128-dimensional export remain
the established local spatial implementation. No published pretrained weights
are loaded. Data, descriptor teachers, loss, optimizer, batch256, BF16 execution,
seed and 12-epoch schedule remain the spatial protocol above, not the paper's
energy/force/stress training recipe. This changes width, angular order, cutoff
and radial basis together relative to the smaller October 7 eSEN; it is a preset
comparison rather than an isolated angular-order ablation.

The worker runs the existing real-data symmetry/precision and full-batch timing
probe before online training. Actual `epochs.csv` and `learning.csv` timings use
the same synchronized scope defined above, enabling equal-epoch comparisons
against both completed MACE runs and the smaller eSEN. The input contract records
the cutoff separately for each architecture, including this preset's 6 Å graph.

## TensorNet spatial adaptation, October 8

`configs/spatial_architecture/tensornet_20261008.json` replaces the unstarted
eSEN-3.2M recipe in the queue. It uses two TensorNet O(3) interaction blocks,
128 channels and the original scalar/antisymmetric/symmetric-traceless Cartesian
tensor decomposition. Upstream block and radial definitions are extracted
verbatim from TorchMD-Net revision `7e22a231f34a855f894daaeb218c8b393068f846`;
the local wrapper replaces species with one shared learned constant plus the
center flag. Charge is an internal fixed zero, never a model input. Self edges
are retained as in upstream TensorNet. Distances and unit vectors are computed
with ordinary PyTorch operations, with analytic zero self distances to avoid
undefined higher derivatives of sqrt(0).

The graph has a 5 Å cutoff and 32 fixed exponential-normal radial functions with
the upstream cosine envelope. Support is the same nearest80/8 Å crop. The norm
of each tensor component supplies invariant scalar features, then the standard
center-plus-tapered-mean pool exports 128 features. The model has 828,160 encoder
parameters and 1,003,690 parameters including the common heads. Norm reductions,
geometry, scalar normalization, pooling and losses are FP32 under BF16 neural
matmul. Edge-message accumulation uses the promoted message dtype (FP32 when
BF16 projections multiply FP32 radial factors), rather than the input tensor
dtype. This fixes the upstream block's dtype mismatch under autocast without
changing its FP32 algebra. Tensor blocks are compiled; graph construction stays eager. All data,
training loss, schedule, full-batch statistics and timing scopes above apply.

The local CPU audit uses two real packed observations in FP64 to measure
rotation, inversion, translation and permutation errors, forward-AD versus
reverse-over-reverse JVP disagreement, and a central-difference JVP check. It
also verifies finite parameter gradients through both JVP and Hessian-vector
product losses. Random centered directions are numerical probes, not velocities.
These checks establish local derivative behavior; nearest80 selection and graph
boundaries are not differentiable, and a cosine cutoff is not globally C2.

Before block compilation, the queued GPU probe measures eager FP32 derivatives
on two real observations. `jvp_relative_fd_error` is the L2 discrepancy from a
central difference with ±0.001 times a unit-variance centered random direction,
divided by the JVP L2 norm (floored at 1e-12). `jvp_seconds` measures synchronized
value-and-JVP computation. `nested_parameter_backward_seconds` measures the
backward pass of mean(JVP²)+mean(HVP²), after their graphs have been built; HVP
is for the scalar mean(E(Q)²). It includes mixed third derivatives but excludes
HVP graph construction, MD, tracking and optimizer updates. These are local
encoder diagnostics, not full response-training timing or scientific fitting.
The subsequent fourteen-update spatial benchmark and final epoch timings retain
the existing definitions. No diagnostic creates an online W&B run.
