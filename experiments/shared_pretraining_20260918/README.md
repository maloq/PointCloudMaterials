# Twelve-epoch shared structural and causal pretraining

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
