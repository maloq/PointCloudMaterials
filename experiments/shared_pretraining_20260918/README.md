# Twelve-epoch shared structural and causal pretraining

September 18 follow-up: a fresh five-epoch **mixed-material MACE + equivariant
bond-order** fit uses the same dynamic-only release, B=2048, grouped VICReg,
present/TDA anchors and temporal-only curvature as the latest GATr protocol.
It adds q4m/q6m prediction from MACE's learned atom tensors before pooling.
The auxiliary is not decoded from the invariant state; its performance alone
cannot establish information retention in z. The existing physical/TDA decoder
continues to constrain z. It is an augmented-objective MACE comparison, not a
controlled backbone-only ablation against the twelve-epoch GATr fit. See the
[recipe](../../docs/shared_pretraining_mace_bond_order_20260918.md) and
[metric definitions](../../docs/metrics/shared_pretraining.md).

**Outcome, September 18:** the batch-512 campaign failed its intended learning
objectives. The [failure diagnosis](../../output/shared_pretraining/diagnosis-20260918/RESULTS.md)
documents unstable updates, saturated heads, GATr collapse, the unsafe causal-head
initialization and information retained by some early checkpoints. Further runs
of this recipe, including batch 1,024, are on hold pending short controlled checks.

September 18, 2026. This supersedes the short-budget launch as the main training
campaign; the [pilots](../structural_pretraining_20260917/README.md) remain separate.
The user approved fresh initialization for three structural variants followed
by causal continuation from each selected parent, with peak learning rate 0.02.
One seed, 20260919, is used throughout.

## Questions and comparisons

| Parent | Structural input/objective | Causal input |
| --- | --- | --- |
| MACE–VICReg | Snapshot; spatial/temporal neighbors | Snapshot |
| GATr–VICReg | Snapshot; spatial/temporal neighbors | Snapshot |
| GATr–JEPA/SIGReg | Three causal frames; next-snapshot prediction | Three causal frames |

Physical geometry85 and instantaneous TDA144 remain anchors decoded from z128.
The earlier physical/TDA definitions, species vocabulary, per-material cutoff
normalization and within-material regularization are retained. No velocities,
relaxed targets or new simulations enter this campaign. The JEPA input contains
three observed position frames; its successor is a separately encoded target.

There are no physical-only controls in the approved three-variant matrix.
Consequently, differences do not isolate VICReg's contribution. The JEPA arm
changes both the objective and history input. Historical pilots differ in batch
size and schedule and are not matched causal controls for those changes.

## Structural training

Use the frozen 250,000-anchor five-metal release, with 128-dimensional exported
states and 64-dimensional projectors. Each statistical batch has **512 anchor
pairs**, four times the pilot batch, from one material/potential family and
static/dynamic category. Spatial and temporal pairs are selected with equal
probability for dynamic VICReg updates; static inputs have only spatial pairs.
JEPA dynamic updates always target the successor. Statistics see all 512 pairs,
using exact gradient caching across physical microbatches.

Twelve epoch equivalents mean three million anchor draws; rounding to a full
batch gives **5,860 updates and 3,000,320 draws (12.00128 equivalents)**. An epoch
equivalent does not guarantee visiting every record before repetition. Partner
views and earlier input frames are not extra anchor draws.

AdamW uses weight decay 1e-4, gradient clipping at 5 and FP32 with TF32 disabled.
Learning rate warms up linearly for 586 updates to the user-specified **0.02**,
then decays by cosine to **0.0002** at update 5,860. The learning-rate formula
uses the absolute saved update count across allocation changes.

Microbatches are chosen separately for memory: MACE 176, GATr snapshot 416,
GATr three-frame JEPA 48. Uneven final chunks are included. Preflight profiles
on actual training observations measured approximately 35–40 GiB reserved
memory, with a 40 GiB allocator budget and expandable CUDA segments. These
operational settings do not change the statistical batch or scientific losses.

Structural selection uses the original fifteen Al selection sources and 480
windows, with source-balanced geometry + 0.25 instantaneous-TDA error.
Checkpoint at update 1 and every 32 updates; validate every 128 and at completion.

## Causal predictive continuation

Initialize encoder, projector and present physical/TDA heads from each completed
structural parent's best selection checkpoint. Add a new MLP from z128 to fixed
geometry/TDA targets at **0.75, 3 and 9 ps**. These future physical values are
not learned latent targets and cannot enter the encoder. The three-frame input
uses offsets -1.5, -0.75 and 0 ps; the two longer forecast horizons exceed the
observation history. Snapshot inputs retain the same three future tasks.

Use the existing native 150-source Al grid: sixteen tracked centers and sixteen
origins per source, 38,400 windows. Preserve **90/15/15/30** independent
train/selection/calibration/test source roles. This yields 23,040 training,
3,840 selection, 3,840 calibration and 7,680 test windows. Current PTM labels
are retained only for evaluation populations; they do not filter training.

Retain present anchors and the parent's representation objective on current /
next-observation pairs. Add future physical MSE + 0.25 future TDA MSE, averaged
equally over the three horizons. Future targets use training-only channel
normalization across current and future native targets, floor 1e-4. Present
heads retain the broad structural target normalization.

Every fourth update replays the broad structural data without future labels.
This keeps multi-material anchors active during Al adaptation. Twelve native
epoch equivalents give **540 native batches plus 180 replay batches**, or 720
updates: 276,480 native and 92,160 broad replay anchor draws. Causal warmup is
72 updates to 0.02, followed by cosine decay to 0.0002. The optimizer starts
fresh for the causal phase; allocation continuation restores it exactly.

Causal checkpoint selection uses a fixed seeded 64 windows per selection source
(960 total). Its score adds present and future geometry + 0.25 topology losses.
The source roles and frame identities are immutable. No held-out gradients or
normalization are permitted, including through replay.

## Frozen evaluations

For every variant, independently evaluate its selected structural and causal
encoder on the same native windows. Fit matched ridge and nonlinear residual
readouts of z128 for current geometry/TDA and all three future horizons. Ridge
strength and nonlinear checkpoints are selected using selection sources only.
The nonlinear predictor is 128→256→256→916, trained for 1,000 updates with a
10% warmup/cosine schedule peaking at 0.02 and ending at 0.0002. Its zero-output
initial residual preserves ridge as a candidate.

Report trained physical heads separately from these fresh frozen probes.
Baselines are training means, temperature-only ridge and persistence of the
current physical target. Report standardized radial, pair, angular, moment,
H0/H1/H2 errors, physical/TDA aggregates and forecast skill over persistence.
Evaluate all test rows, current noncrystalline rows and separate temperatures.
Paired whole-source bootstrap intervals use 4,000 resamples; one seed cannot
quantify training-seed uncertainty. These are independent-source Al metrics;
non-Al archives do not become independent test sets through pretraining.

The exact calculations are in the [metric contract](../../docs/metrics/shared_pretraining.md).
The full study yields one selected structural parent and one selected causal
encoder per variant, rather than overwriting the shared parent. Task-specific
fine-tuning is a later stage outside this submitted matrix.

## Validation and reproduction

### Corrected VICReg restart

The [second loss/rotation audit](../../output/shared_pretraining/restart-diagnosis-20260918/RESULTS.md)
finds retained structural information, weak trained readouts and precision-dependent
rotation errors. See the [mixed-precision architecture proposal](MIXED_PRECISION_PROPOSAL.md)
for the next controlled comparison and candidate efficiency improvements.

Fresh MACE and GATr VICReg runs retain the same broad data, physical/TDA targets,
one seed and 12-epoch exposure, with batch 1,024 and corrected peak LR 0.002.
BF16 mixed precision and normalized scalar readout/head interfaces address the
failure audit; objective weights and spatial/temporal view definitions remain.
Both use snapshot inputs. The selection protocol additionally monitors actual
state/projector/decoder variation and a train-only group-mean baseline. This is
a repaired recipe with several changes, not an isolated LR or batch-size ablation.
The separate matched FP32/BF16 timing comparison measures compute speed only.
See [execution and checks](../../docs/shared_pretraining_restart_20260918.md)
and [metric definitions](../../docs/metrics/shared_pretraining.md).

### H200 batch-size comparison

The additional H200 arm doubles encoder-training batches to **1,024** for all
three variants and both training phases. It retains the same data, one seed,
12-epoch exposure, objective coefficients, model widths and peak LR 0.02.
Structural budgets become **2,930 updates / 3,000,320 draws**; causal budgets
become **270 native + 90 replay updates** with unchanged native/replay draws.
Warmup lengths are 293 and 36 updates. Validation/checkpoint intervals become
64/16 updates to match local draw intervals. Frozen probes retain their original
batch of 1,024, update budget, split and normalization.

This tests larger statistical batches at matched exposure, with fewer optimizer
updates. Sample-count-dependent SIGReg penalties and random batch draws can
change; it is not an isolated hardware speed comparison. Compare the same
source-balanced physical/TDA metrics and preserve each run's measured throughput
and hardware identity. The three variants still lack physical-only controls.
See the [H200 task and recipes](../../docs/h200_shared_pretraining_task_20260918.md).

### Checks

Tests verify held-out exclusion, causal gradients, source-balanced calculations,
uneven-chunk gradient equivalence and exact optimizer/schedule/SIGReg resume.
Real-data integration covers both stages, online W&B, selection, checkpoint
export and the frozen-analysis pipeline. See [execution and continuation](../../docs/shared_pretraining_20260918.md)
for the concrete recipes, allocation receipts and launch state.
# Geometry-protected capacity update

The [2× snapshot architecture implementation](../../docs/shared_pretraining_geometry_fp32_2x.md)
and [validation results](../../output/shared_pretraining/geometry-fp32-2x-20260918/RESULTS.md)
cover enlarged MACE/GATr, selective precision, real-observation rotation checks
and runtime measurements. They establish implementation correctness, not improved
trained prediction. The prepared v4 fits have not been submitted.

The September 18 VICReg-plateau investigation, controlled repair pilots and kernel validation are recorded in [the repair report](../../output/shared_pretraining/vicreg-repair-20260918/RESULTS.md). New training retains the physical/TDA anchors and tests an additional physical-correlation anchor; changes in training loss do not establish downstream improvement.

The [Al-only stability diagnosis](../../output/shared_pretraining/stability-20260918/RESULTS.md)
separates compiler failure, poorly conditioned readouts, and stale normalization
statistics. Its validation remains source-held-out native Al; results are not
claims of other-material or future-prediction improvement.

## Broad full-TDA continuation

The follow-up asks whether the Al-trained GATr can adapt to Al/Mg/Ti/Ta/Zr
while retaining native Al information. It doubles shooting anchors to 75,000,
retains all other parent samples, supplies instantaneous TDA to every supervised
view, and fits three epochs from the validated Al checkpoint. These changes
are bundled; this experiment cannot isolate data quantity from material coverage
or label density. Step-zero and later selection use identical new normalization
and the same 15 held-out Al sources. Other-metal held-out performance remains
unmeasured. The [recipe](../../configs/shared_pretraining/broad_full_tda/campaign.json)
and [metric definitions](../../docs/metrics/shared_pretraining.md#broad-full-tda-structural-continuation)
specify the comparison.

## Dynamic-only mixed-material GATr

The next fit tests whether a shared snapshot representation can retain physical
and instantaneous-TDA information across Al/Mg/Ti/Ta while penalizing temporal
backtracking. It fixes the broad continuation's pooled evaluation-moment mismatch
using domain-specific head moments consistently in training and evaluation.
VICReg remains within-domain even though computational batches mix materials.
Static inputs are removed. Previous/current/next snapshots share encoder weights
and receive a small time-corrected second-difference penalty, leaving deployment
snapshot-based. The run starts from scratch for twelve epoch equivalents at batch
2,048. See [recipe and limitations](../../docs/shared_pretraining_mixed_triplets_20260918.md).

These bundled changes are not an isolated ablation of temporal curvature.
Selection remains fifteen native-Al sources; per-material training losses alone
cannot establish cross-material generalization or improved future prediction.

The mixed-material fit transitions at update 250 to temporal-only curvature,
with a fixed coefficient calibrated against scalar loss and encoder-gradient
norms on training batches. Spatial pairs return to two encoder views. This
preserves the total twelve-epoch exposure budget and optimization state, but
changes the training objective partway through the fit; it is not a matched
from-scratch curvature ablation. The original checkpoint and curves remain
separate. See [coefficient selection and transition](../../docs/shared_pretraining_temporal_backtracking_20260918.md).

## Corrected local structural scale (v10)

The structural observation now has radius 8 normalized units with a 6–8 cosine
taper, replacing radius 17. All present/future/spatial snapshots are cropped
before either encoder. Training-data audit: about 122–125 input atoms and
78–80 weighted atoms across Al/Mg/Ti/Ta; all existing instantaneous-TDA and
physical target support is retained. This matches the approximate scale of the
80-point GeoFrameV2 baseline without imposing hard nearest-k membership.
MACE's pooling supports are 0–3, 3–5 and 6–8, with no outer halo. GATr sees the
same region. Fresh fits keep the data, physical/TDA anchors, mixed-domain VICReg
and temporal-only curvature protocol, recalibrating its small coefficient at the
new initialization. These are corrected fits, not an isolated radius ablation:
MACE pooling scales and GATr count normalization also change. Previous oversized
fits are retired. [Recipe and audit](../../docs/shared_pretraining_local_structure_20260918.md).

The revised comparison uses five epochs for each backbone and bond-order
supervision for GATr as well. GATr pools even-rank harmonics of learned atom-level
multivector streams before the auxiliary q4m/q6m readout; it does not derive
high-order tensors from a pooled vector that could vanish in symmetric crystals.
The invariant state remains z128. Both heads use the same nearest-12 targets,
fixed normalization and 0.1 loss coefficient. Differences in the two auxiliary
readout architectures should be reported alongside backbone comparisons.

## Expanded-data MACE follow-up (19 September)

Question: does the local snapshot MACE benefit from four times as many dynamic
observations with complete instantaneous TDA? Train from scratch on 1,018,080
Al/Mg/Ti/Ta anchors for five epoch equivalents; retain the completed 254,520-anchor
fit as a reference. Static data stay excluded and selection sources stay fixed.
This increases updates from 622 to 2,486 and therefore tests data plus training
budget, not an isolated compute-matched data effect. One seed; no claim about
seed uncertainty. Recipe: `configs/shared_pretraining/mace_expanded_dual/`.
