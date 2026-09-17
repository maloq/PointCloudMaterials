# Shared structural pretraining with physical and topology anchors

Implemented September 17, 2026. The user narrowed the initial eight-fit proposal
to **three runs**, adding three-frame inputs to GATr JEPA. One seed: `20260919`.
See [execution and reproduction](../../docs/structural_pretraining_20260917.md).

## Decision and scope

Train shared, species-aware encoders across Al, Mg, Ti, Ta and Zr. Each exports
one 128-dimensional state. Geometry and instantaneous-TDA heads decode this
state; a separate 64-dimensional projector receives representation losses.

| Run | Encoder input | Representation objective |
| --- | --- | --- |
| MACE–VICReg | Current positions and species | Spatial/temporal neighbor VICReg |
| GATr–VICReg | Current positions and species | Spatial/temporal neighbor VICReg |
| GATr–JEPA | Three causal position frames and species | Next-snapshot latent prediction + LeJEPA SIGReg |

JEPA observes f−2, f−1, f and separately encodes target f+1 using the same
trainable encoder. Static batches have one frame and a spatial partner; those
updates use physical/TDA anchors and SIGReg without a temporal target. No
velocities are invented; this positions-only wave uses no relaxed-TDA, bending
or direct slowness loss. Temporal VICReg itself encourages persistence.

**These runs do not isolate VICReg's contribution:** physical-only controls are
outside the requested scope. JEPA also changes both history and objective.
Historical Al-only fits are references, not matched controls for this protocol.
The intended continuation is structural parent → causal predictive pretraining
with retained physical anchors → frozen evaluations and separately named
fine-tuned copies. This wave trains the structural parents only.

## Training population and splits

The frozen release has **250,000 training anchor records**, **480 Al selection
records** and 1,010 shards. Selection uses 16 tracked centers at frames 104 and
504 in each of the original fifteen selection sources.

| Training stratum | Anchor records |
| --- | ---: |
| Native Al MEAM training histories | 62,500 |
| Ancestry-eligible Al shooting descendants | 37,500 |
| Al EAM, million-atom MEAM source/melt and static Al | 25,000 |
| Mg EAM dynamics and static Mg | 37,500 |
| Ti MEAM source and branches | 37,500 |
| Ta EAM dynamics and static Ta | 37,500 |
| Static Zr | 12,500 |

The eligible pool contains 90 native Al training sources, 456 shooting records,
51 other dynamic/static records and fifteen selection sources. The first
extraction schedule uses **261 source records**; it does not sample every
eligible branch. Three precision Al sources remain available for later expansion.
Dynamic anchors store three causal views, a successor and a spatial partner;
static anchors store two views. Record counts do not imply independent samples.
The audit found 249,966 unique training identities and 34 repeated records.

Extraction cycles over ancestry groups, then samples trajectory, recorded frame
and centers without replacement within a shard. In mixed static/dynamic strata,
one in five tasks uses static configurations. Frames are uniform over recorded
indices, not physical time when sampling cadence varies. Cell size does not
increase a stratum's quota. Training draws homogeneous material/potential and
static/dynamic groups in proportion to their realized anchor counts.

Preserve Al 90/15/15/30 train/selection/calibration/test roles. Held-out descendants
and known ancestry conflicts are excluded from gradients and normalization.
Smoke trajectories and duplicate fixed-duration branch exports are excluded.
Non-Al archives have limited independent ancestry; they provide training
diversity, not independent test confidence intervals. Source manifests,
potentials, target producers and extraction tasks are frozen in the plan.
Unknown generating potentials for static configurations remain explicit.

## Normalization, observations and encoders

R_m is fixed once per material from training sources: `cKDTree.query(k=160)`
distances at 4,000 centers, quantile 0.995, multiplied by 1.02. This query includes
the center itself. Up to eight sampled sources contribute equal numbers of
centers; dynamic calibration uses frame 2 and static calibration uses interior
centers. Calibration does not cover every phase/time and is not refitted.

| Material | R_m, Å |
| --- | ---: |
| Al | 9.121389 |
| Mg | 10.148998 |
| Ti | 9.309865 |
| Ta | 9.387337 |
| Zr | 10.353009 |

Supply `(x_i−x_c) * R_ref/R_m`, where R_ref=9.192189. Observation radius is 17
model units, smoothly weighted from 15 to 17; MACE edge cutoff is 5 model units.
The encoder receives `log(R_m/R_ref)` so z128 can retain absolute scale.
Physical targets retain their physical Å units and support. Fixed material
normalization preserves local compression and density differences.

VICReg variance/covariance and SIGReg see a **full batch of 128 anchors within
one material/potential family**, further separated into static/dynamic groups.
Between-material means cannot satisfy this within-material variance floor.
Sources and cadences may differ inside a dynamic batch; actual time offsets
are supplied per observation.

MACE uses two spatial blocks, width 16, scalar/vector/rank-two channels,
cuEquivariance and smooth multiscale pooling to z128. GATr uses two spatial and
two causal temporal blocks, eight multivector/128 scalar channels, species and
scale inputs, and a current-center scalar readout. Temporal blocks track atom
identities and are active for three-frame JEPA inputs. Element vocabulary is
Mg/Al/Ti/Zr/Ta. Source, potential, phase, temperature and absolute time are not
encoder inputs.

## Spatial and temporal views

VICReg chooses spatial or temporal pairs with equal probability per dynamic
batch; static batches use spatial pairs. A spatial partner is another atom
within 0.25 of the physical observation radius, selected using geometry only.
Temporal pairs use the same atom in consecutive saved frames of one source.
Actual physical times are retained, including adaptive sampling and 33/34-step
interval alternation. Separate branches are never concatenated.

JEPA uses temporal pairs on every dynamic update. Its causal input never
contains the next-frame target. Both encoder branches receive gradients;
there is no EMA teacher or stop-gradient target. Static JEPA updates use only
SIGReg and reconstruction anchors, with no invented temporal successor.

## Physical decoder and topology anchors

The physical decoder maps z128 to **85 geometry channels**: 32 radial, 32
pair-distance, 16 angular and five count/radial moments. These reproduce the
geometry-only components of the native packet directly from positions, with
its smooth 5–7 Å target support.

Instantaneous TDA has **144 channels**: H0 curve16 and H1/H2 images64 each.
It retains the existing physical nearest-80 selection and 3.5 Å death cutoff.
Descriptor boundaries are not redesigned here. Static inherent configurations
supply topology of their supplied coordinates, not thermal-to-relaxed targets.

Each training anchor is labelled independently with probability 0.25
(62,350 labelled anchors in the realized release); its
current/spatial/successor views share availability. Selection targets are fully
labelled. The sampler does not oversample labelled rows. Each endpoint
reconstructs its **own** target, and missing TDA is masked. Older context frames
have no reconstruction loss.

Train-only channel means and population standard deviations (floor 1e−4) use
valid release targets: current/spatial/successor for geometry, labelled views
for TDA. Loss averages standardized errors equally over four physical and three
TDA blocks. The common task anchor is `A = physical + 0.25 TDA`.

## Representation objectives

The projector is `128 → 256 → 64`, SiLU, without batch normalization.
VICReg is `(25 I + 25 V + C)/51`: paired-projector MSE, standard-deviation floor
one (epsilon 1e−4), and off-diagonal sample covariance squared sum divided by 64.
Variance/covariance use B−1 and average over the two views.

Temporal JEPA is `0.95 next-projector-MSE + 0.05 SIGReg`; the predictor receives
the anchor projector and actual future delta in ps. SIGReg averages the authors'
sliced Epps–Pulley test across endpoint projectors: 256 directions, 17
trapezoidal points on [0,3], with symmetry weighting. Static batches omit the
prediction component. Total loss is `A + 0.1 representation`.

This is **temporal JEPA with LeJEPA's SIGReg**, an adaptation of the requested
method. See the [paper](https://arxiv.org/html/2511.08544v3) and
[pinned implementation](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052).
These coefficients are initial engineering choices, not matched optimal weights.

## Training, validation and interpretation

AdamW uses lr 3e−4, weight decay 1e−4, clipping at 5. Target budget is 4,096
updates, batch 128 pairs, FP32 with TF32 disabled. Microbatches of 32 use exact
gradient caching: a no-gradient encoder pass, one full-batch head/regularizer
backward, and encoder recomputation with cached state gradients and replayed
RNG. Independent microbatch regularizers are not averaged.

Checkpoints are saved at step 1 and every 64 updates. Selection runs every 256
updates and at update 4,096, exporting the best encoder separately. Deadline
stops save exact model/optimizer/RNG state. If a deadline truncates a run,
compare common completed update counts. No hardware benchmark runs in training.

Selection score is source-balanced physical error + 0.25 TDA on fifteen Al
selection sources. It chooses checkpoints; it is not a held-out test result.
See the [metric contract](../../docs/metrics/structural_pretraining.md).
Frozen linear/nonlinear probes, prediction, within-material variation, boundary
response and fine-tuning comparisons remain follow-up work after the user
requests analysis. Raw latent errors across encoders cannot rank quality.
One seed provides no training-seed uncertainty estimate.

## Implementation checks

Tests cover geometry-producer agreement, periodic extraction, finite targets,
rotation invariance, packed versus separate examples, species/scale inputs,
GATr causal isolation and earlier-frame gradients, collapse detection,
export/reload, and full-batch versus gradient-cache updates for all three arms.
Real-data integration exercises all trainers, selection/export and resume.
The release audit records ancestry, exact anchor identities, label availability
and actual timeline intervals before the detached fits start.
