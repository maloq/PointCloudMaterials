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

### Local group or patch

The collection of atoms used for a local calculation. A group observable may
average measurements over several atoms; a center observable measures only the
central atom. A patch can also include a surrounding [halo](#halo) used for
computation without including all halo atoms in the final average.

### Tracked center

The same atom, identified by its persistent atom ID, followed across frames. The
center is the first atom in our local encoder inputs. Tracking the center does
not freeze the identities of its neighbors: those may enter or leave the region.

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

### Local-group physical targets

The 16 structural targets of the velocity run: smoothly weighted means and
standard deviations of q4, q6, normalized w4/w6, neighbor-averaged q6, q6 bond
coherence, nearest-shell density, and smooth coordination. They describe the
inner group, whereas some older experiments use center-only observables. The
weights match smooth inner pooling; the exact per-atom definitions are in
[group physics](../src/research/mace_local_state/physics.py) and
[velocity metrics](metrics/mace_velocity.md).

### PTM RMSD and PTM margin

PTM RMSD is the template-fit mismatch reported by PTM. In our encoder diagnostic,
the cutoff margin is `0.1 - PTM_RMSD`: a positive margin passes that cutoff and a
negative one fails. A missing template has an undefined margin. This is distinct
from a top-two cluster membership margin or a learned classifier score.

### Crystal fraction in elemental simulations

For the Al/Ti crystallization producer, the crystal fraction is the fraction of
all atoms matching FCC, HCP or BCC local templates in full-system periodic PTM.
The million-atom Al run uses RMSD cutoff 0.10; its 94% stopping condition requires
two consecutive qualifying assessments. Reaching its separate 400 ps duration cap
does not mean this condition was met. PTM Other includes unclassified environments,
including liquid-like atoms, defects and interfaces; it is not a pure liquid fraction.
Spatial figures classify the whole box before selecting a display slice. This
simulation criterion is distinct from learned local-state classes and PTM margin.

### Liquid-like selection

A declared rule for selecting disordered environments in an analysis. Our recent
stability audit uses group mean qbar6 < 0.30 at both times; the frozen static
state-discovery study uses PTM Other. These are different proxies and can include
defects or interfaces. Neither selection establishes a thermodynamic phase or
proves that all selected neighborhoods have the same local structure.

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
identify which was measured. The latest 304-channel velocity embedding does not
include the old VICReg projector.

### Structure feature block

The 256 coordinate-derived features of `mace_local_phase_space_v1`, standardized
using training statistics. Velocities do not directly enter this block at inference.
During joint training, motion losses can still change the shared coordinate
backbone. Structure-block stability and full-embedding stability are different scores.

### Activity

The 32 learned motion features that are unchanged when velocities reverse. They
support prediction of quantities such as relative speed squared and squared
deformation rates. Individual channels are learned mixtures, not individually
named physical observables. Small numerical sensitivity does not imply that
activity remains constant over a physical lag.

### Flow

The 16 learned motion features that reverse sign with the velocities. They support
signed quantities such as expansion/contraction and radial motion. They are
rotation-invariant scalars, not a global Cartesian flow vector. The activity and
flow blocks vanish for zero relative motion by construction.

Implementations: [context encoder](../src/models/encoders/mace_context.py) and [velocity encoder](../src/models/encoders/mace_velocity.py).

## Our stability questions

### Direct temporal regularization

The `mace_local_smooth_v1` experiment directly penalizes squared changes of a
learned local state, relative to its training within-context variation. It averages
physical-lag bins and separately includes low-order environments. Covariance
control and current physical targets discourage collapse and information loss.
Unlike the older excess-change loss, this term does not stop at the teacher's
amount of movement. Its training normalization differs from the reported
fixed-reference normalized RMS jump; see [the metric definitions](metrics/mace_local_smooth.md).

### Embedding increment and jump

The vector difference `Delta z = z(t + lag) - z(t)` and its measured size. Jump
usually means a large observed increment; it does not by itself prove a
mathematical discontinuity. Finite-lag movement can include physical evolution,
membership changes and numerical effects. Always state the lag, block and scale.

### Normalized RMS jump

For the velocity stability audit, each increment length is divided by
`sqrt(2 * sum(var(training_reference_embeddings)))`, using population variance.
The reported RMS is the square root of the mean squared normalized lengths.
Thus 0.10 means one tenth of the training-reference RMS independent-pair distance
scale, not 10% of atoms moving or 10% prediction error. Each block has its own scale.

### Squared change versus RMS change

RMS means root mean square. A normalized RMS of 0.10 corresponds to normalized
mean squared change 0.01 under the same definition. Percentage reductions in these
two quantities differ. The older frozen-map temporal metric also uses a different
variance denominator, so taking a square root alone does not make it comparable.

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

### Neighbor membership and neighbor-ID retention

Membership says which atom identities enter the local calculation. In the original
80-atom diagnostic, retention is the common IDs among the 79 noncentral neighbors
divided by 79. It differs from similarity of their positions and from agreement
between cluster memberships.

### Boundary crossing and controlled substitution

A boundary crossing occurs when a small geometric motion changes inclusion in a
hard nearest-neighbor or radius selection. A controlled substitution exchanges
selected atom identities while holding other coordinates fixed. These tests
isolate particular support effects; substitutions are diagnostic inputs, not
necessarily physically realized trajectories.

### Atom-matched geometric displacement

A displacement calculated for the same atom IDs at both times, with periodic
geometry and the chosen reference frame handled consistently. Matching row
positions in two independently sorted neighbor lists can compare different atoms
and produce a false estimate of motion.

### Low-dimensional local motion

Our proposed requirement that short-time embedding changes near similar local
states follow a few shared directions. Those directions can rotate along a
curved manifold. This differs from forcing the entire state description into
very few coordinates or fitting a separate curve to each atom track. We must
check the directions on independent groups and retain structural information.
The original velocity model has no motion-direction constraint. The separate
`mace_local_motion_v1` frozen-state experiment implements a shared current-state
basis and assesses directions on held-out source groups; implementation is not
evidence that a physical manifold has been learned.

### Local motion basis

In `mace_local_motion_v1`, a shared small network takes the current local state
and returns four or eight orthonormal directions. The same function applies to
all tracked groups and receives no future frames or source identity. We measure
how much of a subsequent observed increment lies in those directions. This is
a representation constraint and diagnostic, not a forecast of that increment.
A separate evaluation fits directions only from nearby training preparations.

### Temporal bending penalty

Our sequence experiment penalizes changes between consecutive embedding
velocities, using the actual interval durations. Its time-adjusted bend equals
`z_next - 2*z_current + z_previous` at equal cadence and vanishes for constant
velocity even at uneven cadence. It is a finite-lag smoothness constraint; it
must not be interpreted as requiring physically stochastic motion to have zero
acceleration. See [exact calculation](metrics/mace_local_motion.md).

### Information retention and smoothness tradeoff

Information retention asks what physical quantities remain recoverable from an
embedding on held-out data. Teacher-feature matching is only one preservation
strategy. A tradeoff curve shows how errors change as smoothness is strengthened.
If useful topology survives only in a fast auxiliary branch, that does not prove
the main smooth state retains it.

Definitions: [velocity stability](metrics/mace_velocity.md) and [context smoothness](metrics/mace_context_smoothness.md). Proposed motion constraints: [smooth-manifold review](../experiments/mace_velocity_20260915/LITERATURE_REVIEW_SMOOTH_MANIFOLD.md).

## Our learned maps and state assignments

### Short-time temporal coordinates

Coordinates learned to emphasize relationships between local observations separated
by a short physical lag. Our completed frozen comparison uses regularized temporal
canonical correlation analysis (TCCA), which matches correlated directions in
current/previous feature sets. Its deployed output needs one snapshot. The method
does not itself fit a future-event label or prove smooth temporal curvature.

### Learned physical distance

A distance between embeddings whose transformation is fitted to chosen physical
descriptors. Our frozen experiment fits a linear map to ten local-group statistics,
then uses Euclidean distances after the map. Similarity therefore reflects those
targets. It is not distance in Å, a unique physical law, or automatically sensitive
to physical quantities excluded from fitting.

### Physical-neighbor recall and rank imbalance

Physical-neighbor recall measures agreement between nearest groups in representation
space and nearest groups under the chosen descriptor distance. The frozen protocol
compares top-eight lists. Rank imbalance instead measures how poorly the closest
representation neighbor ranks in physical space. These concern similarity between
groups, not retention of atom identities inside a patch.

### State discovery and state catalog

State discovery searches for recurring regions or patterns in the representation.
A catalog is the set discovered on a fitting population. Applying a fixed catalog
to later frames can assign known states or reject observations, but does not
discover a new state absent from the fitted catalog.

### Assignment uncertainty

In our frozen-state protocol, insufficient support for any catalog state is
reported as unassigned mass; ambiguity among supported states is described by
conditional membership entropy and a top-two membership margin. These are not
calibrated phase probabilities. Report assignment coverage alongside agreement:
rejecting nearly everything or assigning one label cannot establish useful
spatial or temporal coherence. See [state metrics](metrics/mace_local_state.md).

Exact map and assignment definitions: [frozen local-state metrics](metrics/mace_local_state.md).

## Our evaluation comparisons

### Native-encoder training

Our direct experiment changes the MACE message-passing weights that produce the
pooled local embedding. Physical readouts and a direction predictor are auxiliary
training heads; their outputs do not replace that embedding. This differs from
fitting a new map on frozen encoder features. In the current coordinate/velocity
architecture, the 256 structural channels depend on coordinates and the additional
48 motion channels also use relative velocities. See [exact protocol](metrics/mace_data_amount.md).

### Independent-source learning curve

Our data-amount study varies the number of independently prepared training
trajectories, keeping tracked groups/frames per trajectory and held-out sources
fixed. Shared preparation descendants do not count as independent sources. Nested
temperature-balanced subsets share a small normalization core. Every fit starts
from the same original pretrained MACE, so this measures additional local-state
training needs conditional on that pretraining. See [data-amount metrics](metrics/mace_data_amount.md).

### Matched-update comparison

Our primary data-amount comparison gives every encoder the same number of
optimizer updates and clouds per update. Smaller datasets are revisited more
often. Scores use the final common update; separately saved validation-selected
checkpoints can come from earlier updates. This tests quality under a fixed short
compute budget, not quality after every data size has fully converged.

### Within-context and within-liquid evaluation

Within-context evaluation measures variation among local groups in the same
source/frame condition; some scores subtract context means. Within-liquid
evaluation restricts the population to a declared liquid-like selection and may
recompute its reference spread. These tests expose local differences that strong
global temperature or liquid/crystal separation can otherwise obscure.

### Source-held-out evaluation

Our comparisons with complete preparation lineages reserved for evaluation.
Related shooting descendants stay together, so holding out frames or files is
insufficient if their preparation is shared with training. A source-held-out
cohort that has already guided research decisions provides development evidence;
a confirmatory claim needs an untouched independent population.

### Sibling divergence

Differences between shooting branches after a shared starting configuration.
Our diagnostic distinguishes branches with the same momenta but different
thermostat streams from branches with different momenta. This measures unresolved
physical evolution under the branch protocol. It differs from repeated encoder
inference on identical inputs, and siblings from one parent are not independent
preparations.

### Storage round trip and high-precision comparison

A round trip casts or saves a value to the storage format and reads it back for
comparison. It measures loss relative to the starting values. If those values were
already quantized to the same format, a tiny additional difference cannot establish
that the original storage loss was tiny. A high-precision reference must contain
independent retained precision.

Exact sampling and intervention definitions: [encoder diagnostics](metrics/mace_encoder_diagnostics.md) and [velocity metrics](metrics/mace_velocity.md).
