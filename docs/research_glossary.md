# Research glossary

Definitions of terms central to **our local atomic-state research**, concentrating
on meanings specific to our protocols and comparisons. Use these entries when
explaining the method or its results. General textbook definitions of algorithms,
statistics, and training concepts are outside this glossary's scope.

Examples: [local structure](#local-structure), [relaxed topology](#relaxed-topology),
[smooth inner pooling](#smooth-inner-pooling), [readout](#readout), and
[normalized RMS jump](#normalized-rms-jump). Keep entries anchored to their role
in our research; use an established method's name where needed without adding a
general introduction to that method. Add or update entries when our terminology
or protocol-specific meanings change.

Exact formulas, units, scales and sampling belong in `docs/metrics/` and each
run's exported `tables/METRICS.md`. This glossary explains those definitions
without overriding them. Details labelled `mace_local_phase_space_v1` describe
the September 15, 2026 coordinate/velocity experiment, not every MACE checkpoint.

| Subject | Terms used in our discussions |
| --- | --- |
| [Local object](#the-local-object-we-describe) | Local structure/state, tracked center, temporal pairs and observed history |
| [Structural targets](#our-structural-targets) | Group observables, PTM margin, instantaneous/relaxed topology and their errors |
| [Embedding construction](#our-embedding-construction) | Halo, smooth inner pooling, center embedding, readout and motion blocks |
| [Training comparisons](#our-training-comparisons) | Teacher/student, retention, excess temporal change and checkpoint selection |
| [Stability questions](#our-stability-questions) | Normalized jumps, coherence, atom membership and low-dimensional local motion |
| [Maps and assignments](#our-learned-maps-and-state-assignments) | Temporal coordinates, learned physical distance, catalog states and uncertainty |
| [Evaluation comparisons](#our-evaluation-comparisons) | Within-liquid comparisons, held-out sources, siblings and storage effects |

## The local object we describe

### Local structure

The spatial arrangement of atoms in a specified neighborhood: their distances,
bond directions, packing, and geometric organization. Two liquid neighborhoods
can have different local structures even when both receive the same liquid-like
label. Local structure alone does not include velocities or describe the progress
of the whole simulation.

### Local state

A description of a local atom group at a particular time or short period of time. Depending on the
protocol, it can include structure, relative motion, and information from a short
observed history. It need not be a discrete class. Our encoder objective is an
informative, stable local state; predicting its future is a separate task.

### Local environment

The atoms and geometric context surrounding a selected atom or location. The
neighborhood rule is part of the definition: nearest 80 atoms, a radius cutoff,
and a smoothly weighted region are different environments. Always identify which
rule an embedding or target uses.


### Tracked center

The same atom, identified by its persistent atom ID, followed across frames.
Some earlier local encoders place the center first. The native history producer
sorts the union of observed atom IDs, so the center is not necessarily at index
zero; GATr verifies the unique observed zero-displacement atom. Tracking the
center does not freeze its neighbors' identities: they may enter or leave the region.

### Temporal pair and temporal neighbor

Two observations of the same tracked center at different times. Temporal neighbor
means nearby in time, whereas spatial neighbor means nearby in space. In the
velocity protocol, the two observations are encoded independently and compared
in a training loss; they are not jointly fed into a history encoder.

### Observed history and causal input

A sequence ending at the time being described. Causal input uses current and past
observations only. A short history can help distinguish vibration from a persistent
rearrangement. Averaging or attending to future frames instead defines an offline
smoother and must be declared. The current velocity encoder has no history input.

## Our structural targets


### Persistence image and TDA vector

The particular topology descriptor used for our local targets: a 16-bin H0
component-merging curve and two flattened 8×8 H1/H2 images summarizing loops and
cavities. These give 144 components in total. Birth/death scales and persistence
lifetimes are geometric lengths here, not physical times. Historical targets
with other widths or normalization are different quantities. See the
[descriptor producer](../src/analysis/liquid_structure.py).

### Instantaneous topology

Also called observed, current, or hot topology. TDA calculated on the positions
in the observed snapshot, including thermal displacement. In the velocity protocol
it uses the nearest 80 atoms, including the center. Changes can reflect physical
motion, changes of selected atom identities, or sensitivity of the descriptor.
Hot names the unrelaxed input here, not a universal temperature range.

### Relaxed topology

TDA calculated after the observed configuration undergoes the recorded
energy-minimization procedure. For our context/diagnostic cohort, the full cell
is minimized with FIRE before constructing the relaxed targets. This is not a
later frame of the MD trajectory or a label of a future phase. Minimization can
reduce thermal distortion but also change local organization; the potential,
constraints and stopping criteria remain part of the target definition.

### Instantaneous-TDA error

The difference between a readout's prediction and calculated instantaneous TDA,
using the run's declared error and scaling. Lower means the embedding/readout
combination better reconstructs that target. It does not establish that the
embedding is smooth, clusters are meaningful, or the same model predicts the future.

### Relaxed-TDA error

The difference between a readout prediction and the calculated relaxed-topology
target. An encoder can predict this well while discarding instantaneous details.
Always distinguish an error against independently calculated relaxed targets from
changes in a relaxed-TDA readout when no new relaxed target was computed.

Exact definitions: [topology metrics](metrics/topology.md), [diagnostic metrics](metrics/mace_encoder_diagnostics.md), and [velocity metrics](metrics/mace_velocity.md).

## Our embedding construction

### Halo

Surrounding atoms supplied to make message-passing context complete. Halo atoms
can influence the inner atoms' features even when they receive zero pooling
weight. Supplying only the final pooled atoms would change the computation and
can create artificial sensitivity to patch boundaries.

### Smooth inner pooling

Our weighted group average with full weight through 5 Å, a continuous taper
between 5 and 7 Å, and zero weight at and beyond 7 Å. The taper is C2: its first
and second derivatives are continuous at the joins. This addresses a particular
spatial boundary; it does not guarantee small changes over physical time.

### Tracked-center embedding

The feature vector of the tracked atom after it has received neighborhood
messages. It replaces a group average with one center-node output while retaining
surrounding context. It is not an isolated-atom representation. Earlier experiments
compared it with inner pooling; the current velocity model uses inner pooling.

### Readout

Two related meanings occur in this repository. A **graph readout** selects or
pools atom features into a group embedding. A **physical readout** maps an embedding
to a quantity such as q6 or TDA, using a linear model or small neural network.
In explanations of prediction errors, readout usually means the second. Name
which meaning applies when it could be ambiguous.

### Projector

In our earlier VICReg models, the transformation after the raw MACE encoder used
for the training objective. Raw encoder features and projector features have
different geometry and can retain different information, so each result must
identify which was measured.

### VICReg in the earlier encoder protocols

The earlier VICReg trainer matches representations of two declared views,
penalizes insufficient per-channel variation across a batch, and penalizes
off-diagonal covariance. View construction and whether the loss acts on encoder
or projector features are part of that protocol. It is not a synonym for any
training that uses coordinates without manually annotated labels.

The September 17 backbone-v2 GATr/MACE physical screen does **not** use VICReg.
It trains on computed present and future physical targets. Its separate onset
models use event/survival likelihood only; its TDA probes freeze the physical
encoder. See the [current training/data audit](data_usage/gatr_20260917.md).

### Shared structural pretraining

The first stage learns one exported structural state from multiple
materials using same-time geometry and species, with physical and topology
prediction heads attached to that state. Frozen evaluations use its unchanged
weights; task-specific fine-tuned copies have separate checkpoint identities.
Causal predictive pretraining is a subsequent continuation that adds observed
motion and multi-horizon physical supervision while retaining structural
anchors. The structural stage tests snapshot MACE/GATr with spatial/temporal
VICReg and GATr with three causal position frames predicting the next separately
encoded snapshot. Instantaneous TDA is the only topology target in this wave. See
the [September 17 protocol](../experiments/structural_pretraining_20260917/README.md).

The [September 18 continuation](../experiments/shared_pretraining_20260918/README.md)
uses twelve epoch equivalents (anchor draws divided by the training population)
with batch 512 and warmup/cosine decay. Its causal phase retains present and
representation anchors, predicts fixed physical/TDA values at 0.75/3/9 ps,
and replays broad structural batches every fourth update. Causal epoch counts
exclude replay draws. Frozen probes compare unchanged selected structural and
causal checkpoints on the same whole-source test split; allocation continuation
is resuming a fit, whereas causal continuation is a new scientific phase.

### Physical decoder

A learned head that reconstructs explicitly calculated physical descriptors
from the exported embedding. The native physical protocol uses a small MLP for
the present 128-channel geometry/motion packet and a separate future head.
This is not an MD integrator or an energy/force model. The structural
stage instead uses 85 geometry-only channels: 32 radial, 32 pair-distance,
16 angular and five count/radial moments. That is a new target contract, not
the old 128-channel metric with missing velocities filled by zero.

### LeJEPA and SIGReg in the structural-pretraining proposal

LeJEPA combines related-view agreement with Sketched Isotropic Gaussian
Regularization (SIGReg), which compares random one-dimensional projections of
embeddings with a Gaussian target. It is not inherently next-time prediction.
The revised atomic adaptation predicts the same center's next temporal
projector embedding using a three-frame causal embedding and actual time delta, with
SIGReg on encoder/projector outputs. This is temporal JEPA with SIGReg rather
than the paper's original view-agreement loss. Physical and instantaneous-TDA
heads decode each view's own exported state. View construction and regularizer
weight must be tested on this domain; the paper's results do not establish an
advantage for atomic encoders. [Original paper](https://arxiv.org/html/2511.08544v3).

### Fixed material-cutoff normalization

In the structural-pretraining protocol, each material has one cutoff
R_m fitted from training-only neighborhoods. Centered coordinates are divided
by R_m and expressed in a common model reference unit; edge cutoffs, observation
support and pooling boundaries convert consistently. This differs from dividing
every material by the same constant or rescaling each patch to its instantaneous
maximum radius. The fixed scale also enters the encoder so its exported state
can retain absolute size. Physical and instantaneous-TDA targets stay in physical
length units. See the [normalization protocol](../experiments/structural_pretraining_20260917/README.md#normalization-observations-and-encoders).

### Within-material representation statistics

VICReg's variance floor can be satisfied by different material means even when
features vary little inside any one material. Cutoff normalization alone does
not exclude that solution. The revised proposal uses full 128-anchor batches
from one material/potential family for variance, covariance and SIGReg, with
material balance across updates. This preserves the statistical batch size and
excludes between-material means from the regularizer. It does not establish
independence of nearby observations or eliminate within-material phase shortcuts.

### Native MACE in the current experiments

Our project label for a custom MACE-based encoder trained directly on atomic
positions and velocities, with optional observed history. The current
[NativeEncoder](../src/research/local_predictability/native_model.py) uses the
MACE library's interaction and many-body product blocks, plus our motion inputs,
atom-level temporal attention and smooth multiscale pooling. The parent model
starts from random weights; matched continuations inherit its trained weights.
There is no pretrained foundation-model checkpoint in this protocol.

The tested configuration has two spatial blocks, width 16 in each scalar, vector
and rank-two tensor channel family, a 17 Å observation radius, 5 Å spatial edge
cutoff and one rotation-invariant 128-dimensional output. The snapshot uses the
current frame; history12 uses 17 frames over 12 ps at 0.75 ps cadence, with
temporal updates between spatial blocks; repeat12 fills those time slots with
the current observation. Pooling happens after atom-level processing.

Physical-target and onset-target studies train separate copies of this
architecture. A **native prediction head** is the decoder fitted jointly with
its encoder; a **frozen-state readout** is fitted afterward while holding that
encoder fixed. cuEquivariance is an execution backend for the tensor operations,
not another scientific objective or the meaning of "native."

### Physical packet in the native predictability protocol

The fixed 128-channel target calculated from neighbors around a tracked center:
radial structure, pair distances, angular structure, speed, radial velocity and
mixed geometric/motion moments, on a smooth 5–7 Angstrom support. It is distinct
from the 144-channel persistence-image TDA vector and from PTM phase labels.
Backbone-v2 predicts the current packet and six future packets using train-only
channel normalization. A packet is a target, not the raw atom input or exported
embedding. See [metric definitions](metrics/local_predictability_backbone_v2.md).


## Our stability questions


### Embedding increment and jump

The vector difference `Delta z = z(t + lag) - z(t)` and its measured size. Jump
usually means a large observed increment; it does not by itself prove a
mathematical discontinuity. Finite-lag movement can include physical evolution,
membership changes and numerical effects. Always state the lag, block and scale.

### Normalized RMS jump

For stability audit, each increment length is divided by
`sqrt(2 * sum(var(training_reference_embeddings)))`, using population variance.
The reported RMS is the square root of the mean squared normalized lengths.
Thus 0.10 means one tenth of the training-reference RMS independent-pair distance
scale.

The native causal-state protocol uses a distinct denominator: covariance is
computed within each evaluated source, and source-level J values are averaged
equally. Its declared lag is 0.75 ps. Do not substitute the training-reference
normalization above when interpreting those tables; see its
[frozen metric definition](metrics/mace_causal.md).

### Temporal smoothness

Regularity of the path traced by an embedding through time, including changes of
direction and speed. Small isolated increments do not establish this property.
Consecutive observations are needed to measure bending or erratic motion; two
endpoints cannot determine what happened between them.

### Temporal coherence

Consistency of a local-state description across nearby times, while allowing real
transitions. Depending on the question, measure continuous embedding changes,
physical readout consistency, or assignment agreement. Agreement caused by a
single constant label is not sufficient evidence of a useful temporal description.

### Spatial coherence

Organization of nearby local descriptions in physical space, consistent with real
structural variation and interfaces. Adjacent groups overlap, which itself creates
correlation. A useful spatial-coherence assessment must consider overlap, physical
observables, label frequencies and assignment coverage.


The separate [causal native MACE protocol](mace_causal.md) sends same-atom history
messages between spatial MACE layers, before its final pooling. Future observations
are supervision only. A separately trained repeated-anchor control receives the
current frame at every historical offset, retaining the same architecture.

### Predictive memory in partial observations

In `predictive_memory_partial_observation_v1`, memory gain is the reduction in
held-out negative log probability of a fixed continuous physical future path
when a model receives observed atom history instead of a matched snapshot.
Both models use the same physical targets, source split, radius and optimizer
budget. A separately trained repeated-anchor control tests whether real history
helps beyond extra temporal computation. The total observation radius includes
all message-passing context; there is no additional hidden halo. Crystallization
and topology labels do not train or select these encoders. This predictive score
is neither exact mutual information nor proof of a Markov state. See the
[protocol-specific metric definition](metrics/predictive_memory.md).

### State sufficiency diagnostic

In the causal MACE protocol, compare matched physical predictors given frozen z
and either the original observed atomic history or one constant training history.
Only z varies in the constant-history control. Better held-out future prediction
with real history indicates information available in the inputs but discarded by
z. No improvement does not establish that z is a complete or Markovian state.
These diagnostic predictors do not replace the exported encoder.

### Finite-horizon local transition risk

The causal MACE hazard head predicts the first confirmed sustained crystalline
episode of the tracked center within the retained observation segment, conditional
on remaining event-free through earlier time bins. It uses the same local PTM
assay and explicitly handles right censoring and confirmation follow-up. This is
neither whole-system nucleation probability nor a committor between specified
competing basins. See [exact event rules](metrics/mace_causal.md).
