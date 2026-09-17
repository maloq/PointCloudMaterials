# Predictive memory in partial atomic observations

Question: does a native, jointly trained atomic-history encoder improve fixed
physical future predictions over a matched snapshot? Crystallization labels and
topology are excluded from representation training and checkpoint selection.
This replaces the previous smoothness/crystallization objective as the active
pilot, while preserving those earlier results as a separate protocol.

The first comparison is positions versus positions+measured relative velocities,
each with H=0,12,48 ps. Two separately trained H=48 repeated-anchor controls
distinguish actual history from the extra temporal computation. All models use
the same source splits, center/anchor/target tuples, seed, width, heads and 3,000
optimizer updates. Comparisons are exploratory and restricted to one seed.

The native MACE has two interleaved spatial/temporal blocks, each retaining 16
scalar, 16 vector and 16 rank-two channels. An invariant 128-dimensional state
is pooled once, after atomic reasoning. H=48 includes all 65 stored frames;
attention is causal with actual time offsets and constant age support. Atom
membership can change at every frame. Total observed radius is 17 A, including
all graph context; the 5--7 A target region remains fixed. No fixed nearest-80
list, full-cell learned features, outside-radius halo or cached learned features
are used. Four-component diagonal-plus-rank-two Gaussian mixtures describe the
joint physical target path at 0.75,3,12,48,96 ps. Known temperature is provided
identically to all prediction heads.

The existing 150 independent-melt Al trajectories give 90/30/30 inherited
train/validation/test sources. One center per source (first ID from the inherited
sorted four-center sample) and three anchors at 299.25,300,300.75 ps produce 450
matched windows. Source sampling is balanced; adjacent anchors are not treated
as independent uncertainty units. This is deliberately much smaller than the
proposed 512-center study. Raw coordinates and velocities are float16: the
matched full-precision audit is outstanding. Shared initial FCC preparation,
distinct melted configurations, array duplicates, transition kernel and
measurement completion are audited without reading crystallization outcome
files. The release records limitations rather than manufacturing missing data.

Initial implementation controls check rigid transformations, translation and
velocity boosts, identity permutation, absent/outside atoms, causal attention,
observed-only identity union, gradients through the oldest frame and all blocks,
exact equivariant attention aggregation, activation recomputation, likelihood
against full covariance, resume equivalence, and analytic hidden-velocity versus
full-state controls. The latter are unit-test fixtures, not new MD trajectories.

The allocation pilot does not establish converged memory length, compressed bit
rate, state sufficiency, kinetic closure, a radius-memory surface or confirmatory
crystallization performance. No simulations are generated. Topology is reserved
for external evaluation. Predictions are scored in fixed physical coordinates,
not primarily by incomparable latent MSEs. See [metric definitions](../../docs/metrics/predictive_memory.md)
and [operating instructions](../../docs/predictive_memory.md).

Reproduction uses [pilot.json](../../configs/predictive_memory/pilot.json) and the
commands in the operating instructions. Outputs are under
`output/predictive_memory/pilot-20260917/`; the paired comparison is written only
after all six primary fits and both repeated-anchor controls complete.

## Predeclared follow-up: training-seed replication

Before the primary history fits completed, a second seed (20260918) was selected
for the velocity-input H=0,12,48 models and the H=48 repeated-anchor control. This
four-fit follow-up tests whether history beyond measured current velocities has
a reproducible benefit. Data, normalization, architecture and 3,000-update budget
are unchanged. It uses [its own configuration](../../configs/predictive_memory/replicate-xv-seed20260918.json)
and output `output/predictive_memory/replicate-xv-seed20260918/`.
Its separate paired report is conditional on this seed; two seeds do not make
the reused test sources confirmatory or remove the precision limitation.

## H200 capacity comparison

The H200 assignment doubles atom-feature channels from 16 to 32 while preserving
the 128-dimensional exported state, two blocks, physical targets, data, seeds
20260917/20260918 and 3,000-update budget. For each seed it fits velocity-input
H=0,12,48 and the H=48 repeated-anchor control. This isolates atom-level capacity
from the history objective and bottleneck dimension. Both seeds have matched
width-16 velocity-input runs assigned to the H100. Recipes live in
`configs/predictive_memory/h200/`; [handoff and execution](../../docs/predictive_memory_h200.md)
describe how to run this distinct capacity experiment. No H200 result is yet
available; width 32 is an experiment setting rather than an established improvement.
