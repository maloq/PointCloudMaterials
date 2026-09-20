# Crystallization prediction from the expanded local MACE checkpoint

Question: does general structural pretraining improve prospective local onset
prediction over a frozen representation, jointly adapted representation or the
same encoder trained only on onset? Does retaining orientation in spatial
context improve the frozen and fine-tuned cases?

The parent is the validation-selected checkpoint from the completed five-epoch,
1,018,080-anchor mixed-material fit. Its file hash, actual selected update and
encoder producers are pinned in the new run's `technical/plan.json`. No test
outcomes choose the parent or downstream variants.

## Population and labels

All 150 audited independent 70,304-atom Al sources, 400–520 K, are retained with
the original 90 training / 15 selection / 15 alarm-calibration / 30 test source
split. The existing outcome-independent 16 tracked centers per source and verified
PTM timelines are reused. Origins are every 3 ps from 48 to 498 ps: **362,400
candidate windows**, versus 38,400 in the old native assay. This increases temporal
coverage 9.44-fold, not the number of independent trajectories. It does not claim
that correlated windows are independent samples. No synthetic/new simulations
are required. Shooting descendants are not added without matching validated
onset timelines and ancestry checks.

A sample is eligible only before the first three-frame sustained PTM crystal
run, with three preceding/current noncrystalline observations. The event is the
start of that run, not its later confirmation. Six hazard bins end at 0.75, 3, 9,
24, 48 and 96 ps. All origins have enough future follow-up and confirmation;
examples with no event survive all six bins. Ground-truth future labels never
enter observations. The model estimates local transformation, not nucleation
anywhere in the cell. Sources used for pretraining fitting/selection are checked
against downstream calibration/test ancestry and any overlap fails.

## Observations and comparison

The native encoder retains its original material normalization, local radius 8
in model units, and no halo. Larger context consists of additional *local*
embeddings. Seven context centers are available: the tracked atom and three
farthest-point representatives in each of the (0,12] and (12,25] Å annuli.
The first representative is the nearest atom, the rest maximize coverage.
Selection uses current geometry only; neighborhoods are rebuilt independently
at each observed time. Spatial aggregation uses relative periodic coordinates,
smooth support weights, and either a weighted mean or pair-distance attention.
These sparse contextual observations are not a dense encoding of every atom in
the 25 Å sphere.

Histories are sparse, causal snapshots at offsets:

| Duration | Observed offsets, ps |
| --- | --- |
| 0 | 0 |
| 3 | -3, 0 |
| 12 | -12, -3, 0 |
| 48 | -48, -12, -3, 0 |

All frames are independently encoded. A separately trained repeated-current
48 ps attention control shares capacity and time-offset inputs.

The queue includes frozen scalar features; jointly fine-tuned MACE; scratch
MACE of identical size; and tensor-aware context for frozen/fine-tuned MACE.
The tensor variants retain native 32×(l=0,1,2) center features until contractions
with actual neighbor directions. Tensor norms and directional contractions enter
context attention, and normalized l=1/l=2 alignment between contextual tensors
also contributes to pairwise attention scores. The final hazard remains rotation invariant. This is a
specific tensor-context architecture, not a fully equivariant transformer.

Baselines include condition-only hazards (temperature and time since quench),
linear current-z hazards, nonlinear current-z hazards, and a stronger current
physical-packet/bond-order predictor. That descriptor baseline includes velocity
information already present in the packet; MACE observations use positions only.
A constant no-transition/persistence baseline is reported on the identical
at-risk population. All context choices are declared before test evaluation.

Each fit has one seed, 2,048 updates and 64 source-balanced samples/update, with
uniform eligible sampling within source and no transition oversampling. Fine-tune
and scratch use the same budget and encoder LR 3e-5; head LR is 5e-4, with warmup,
cosine decay, AdamW and gradient clipping. BF16 is confined to the established
MACE mixed-precision path; tensor operations and predictive heads use FP32.
Checkpoint selection uses source-weighted hazard NLL on up to 64 fixed windows
per selection source. Full natural at-risk calibration/test populations are
scored once using the selected checkpoint. Width/budget convergence, additional
training seeds and denser history are beyond this first screen.

See [metric definitions](../../docs/metrics/crystallization_transfer.md) and the
[execution recipe](../../docs/crystallization_transfer_20260919.md). These previously
examined sources make this an exploratory comparison. Report source uncertainty
conditional on one seed; do not claim training-seed confidence or select further
variants using test outcomes.

## Radius, duration and data scaling extension

Predeclared while the initial queue runs; no test scores determine these choices.
The common reference is **12 ps history, 25 Å context, three full training epochs**,
using all **90 training sources / 109,838 eligible training windows**. Compare
frozen scalar, fine-tuned scalar, scratch scalar, frozen tensor and fine-tuned
tensor models. The initial fixed-update screen remains a separate protocol.

| Factor | Values | Held fixed |
| --- | --- | --- |
| Context radius | 0, 6, 12, 18, 25 Å | All training data, three epochs, 12 ps history |
| Training duration | 1, 3, 6 full epochs | All training data, 25 Å context, 12 ps history |
| Independent training data | 30, 60, 90 sources | 5,151 updates, 25 Å context, 12 ps history |
| Window coverage control | 25%, 100% within each of all 90 sources | 5,151 updates; scalar variants |

Training counts are temperature-balanced (6, 12 or 18 independent sources per
400/450/500/510/520 K), selected with fixed outcome-independent random ranks and
nested across data amounts. Window subsets are nested random samples within each
source. The identical selection/calibration/test sources and eligible windows
remain intact. Sources and windows are different kinds of data; correlated
windows are not treated as extra independent trajectories.

Epochs now mean shuffled full passes through eligible training windows, using
per-row source weights to preserve the source-balanced objective. With batch 64,
one epoch has **1,717 updates** (1,716 full batches plus a final batch of 14),
three epochs 5,151 updates, and six epochs 10,302 updates. All fits restart from
the same declared initialization and use budget-specific warmup/cosine schedules;
this tests complete training recipes, not extension of a previously decayed
schedule. Validation chooses a checkpoint within each budget. Report its actual
step alongside the stopping budget. One seed remains in use.

The radius comparison uses the same geometry-selected candidate pool, suppressing
all contributions outside each support. Smaller radii therefore admit fewer
candidates; it is not a constant-density or constant-count radius ablation. Local
MACE graphs retain the pretrained support. A contextual center near the support
boundary sees its own local neighborhood beyond that boundary. These radii
specify where contextual **centers** may contribute, not a strict cutoff on every
input atom. Mean-pooling controls cover all four positive radii for the frozen
scalar encoder. This totals **52 fits**, with shared reference configurations
counted once. The six-epoch variants follow the shorter comparisons.

Reproduce with `configs/crystallization_transfer/mace_scaling_20260919.json` and
the existing queue command. Compare the same classification, timing-with-misses
and sparse spatial metrics, with paired source uncertainty. No new labels,
trajectories or simulations are generated.

## Completed results, 19 September

Both queues completed: 102 trained models plus the unfitted no-transition
baseline. The [full results report](../../output/crystallization_transfer/summary-20260919/README.md)
contains all metrics, paired source intervals and scaling plots. Frozen spatial
attention benefits from broader context and 12 ps history. Six epochs improves
over three on event NLL, and 90 versus 30 training trajectories helps at matched
updates. Tensor context and 48 versus 12 ps history show no clear added benefit.

The fine-tuned/scratch arms have poor geometry sensitivity. A training-only GPU
audit identified large feature-mean drift under fixed initial normalization and
heavy early gradient clipping. Their scores are completed experimental outcomes,
but do not establish an intrinsic disadvantage of trainable encoders. Corrective
training was not launched as part of this read-only results analysis.

The follow-up [adaptive interface and attention study](ADAPTIVE.md) reruns
fine-tuning/scratch with corrected normalization and extends validation-selected
recipes to 12 and 24 epochs.

The [structural trajectory comparison](PATHS.md) adds 12/24-epoch direct, autoregressive, mixture and diffusion forecasts on the same source split.

The [targeted trajectory follow-up](PATH_REFINEMENT.md) diagnoses overfitting, autoregressive feedback mismatch and diffusion instability, then tests 30 remedies with validation-only promotion to longer fits.

## Consolidated recent report

[Local encoder recipes and all recent crystallization studies](../../output/crystallization_transfer/recent-report-20260919/README.md) combines completed results, the 35-fit path refinement and interim corrected encoder screens. The [completion-update note](REPORT_UPDATE.md) tracks the remaining ten longer encoder fits.
