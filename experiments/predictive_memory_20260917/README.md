# Predictive memory in partial atomic observations

The [17 September consolidated report](../../output/predictive_memory/research-summary-20260917/RESULTS.md)
includes the completed local pilot, user-reported H200 width results, frozen-head
diagnostics, and the first completed stronger-present-loss seed. The original
two-seed pilot does not establish a consistent history advantage; the stronger
objective's first-seed result is promising, with replication pending at capture.
Remote rounded means and evidence limits are preserved in
[h200_reported_results.json](h200_reported_results.json).

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

## Completed H100 pilot and optimization follow-up

All eight original fits and four second-seed xv fits completed at 3,000 updates.
The first seed's test NLL gain for xv H=48 over snapshot was 0.00275
(95% source interval 0.00060 to 0.00475); the second seed gave -0.00462
(-0.01930 to 0.01223). H=48 versus repeated anchor gave 0.00098
(-0.00165 to 0.00351) and -0.00504 (-0.00954 to -0.00117), respectively.
There is no reproducible history advantage in this pilot. Source intervals do
not cover training-seed variability. See the retained per-seed comparison reports.

A validation diagnostic of the velocity-input models found that replacing every exported state with its
training mean did not worsen the fitted future head, even though it worsened
present reconstruction. A current-physical-packet ridge predictor attained about
0.690 validation future MSE, versus about 0.825 for temperature-only ridge and
0.85 for the neural mixture mean. This motivates checking optimization and
information retention before enlarging the memory/radius sweep. The diagnostic
is observational evidence about these fitted heads, not proof that their input
histories contain no predictive information.

The follow-up holds data, capacity, physical targets, conditions and source
splits fixed. A two-by-two design crosses present-loss weights 0.05 and 1.0
with both existing seeds, at a common 12,000-update budget.
Each recipe includes xv H=0,12,48 and a separately trained H=48 repeated-anchor
control: 16 fresh fits. This distinguishes a longer training budget from stronger
present-information training. Every recipe still selects its checkpoint by
validation physical-path NLL; no test-driven early stopping or new smoothness
term is introduced. The initial 3,000-update experiments remain separate records.

Alongside paired future NLL, retained validation curves show convergence of
present and future errors. Frozen-head constant-state interventions and newly
fitted linear readouts quantify whether the learned head uses the state and
whether accessible predictive signal remains in it. Linear current-packet and
temperature controls establish reference errors on the same physical outcomes;
they are not replacement encoders. No claim of full-history sufficiency or
kinetic closure follows from these controls. This follow-up was designed after
examining the pilot and continues to use exploratory test sources.

Recipes: `configs/predictive_memory/optimization/{original,present1}-seed*.json`.
Outputs: `output/predictive_memory/optimization-{original,present1}-seed*/`.
Use the existing `train`, `compare --modalities xv`, and `diagnose --modalities xv`
module commands in the [workflow](../../docs/predictive_memory.md).

## Prospective precision and temporal-resolution study

New simulations were authorized after the existing-data optimization queue was
launched. That queue retains its original data and protocol. Fresh independent
lineages with paired float32 and full-box float16 observations will allow a
controlled comparison against centering in float32 before local float16 storage.
The scientific question is whether a measured history gain survives this change
in observation precision, rather than merely averaging rounding noise. Finer
cadence additionally permits short-history/short-horizon tests below 0.75 ps.

Reserve the new test lineages before inspecting future outcomes. Develop the
precision audit and sampling protocol on the new training/validation lineages;
lock the matched comparison before evaluating reserved tests. The new shorter
trajectories do not match the old pilot's 300 ps anchor, so a cross-release score
comparison alone cannot establish an improvement. Production details and the
separate fresh-source contract are in the [simulation record](../../docs/simulations/predictive_memory_precision_20260917.md).
