# Saved-prediction overfitting and feature audit

Audit the three completed crystal-set and three completed interface models from
al64-random-20260928. Verify saved prediction checksums and checkpoint identities.
Do not rerun the encoder, change historical metrics, choose new checkpoints or
create online diagnostic training runs.

Fit-gap tables use the selected checkpoint's saved train/selection/calibration/test
predictions. Within each role, assign half mass to each available fixed/uniform
population and equal source mass within population; condition and renormalize
for named visibility/population subsets. These conditional weights may differ
from the original per-subset macro-source table, so the audit has its own contract.
Report distance marginal NLL, RMSE of the capped posterior mean, MAE of the capped
posterior median, Brier for d<=20 A, its prevalence, censored mass, and conditional
direction NLL on valid directions. Direction is omitted for the distance-only fit.
Geometry/population shifts and different prevalence can contribute to apparent
train/test gaps; a gap alone is not proof of memorization.

Learning curves compare logged minibatch objective averages per nominal epoch
with the full source-weighted validation objective. The training trace samples
one minibatch every 16 updates; it is not an exhaustive selected-checkpoint train
score. Separate fit-gap tables supply that selected-checkpoint distance score.
Selected epoch is the minimum recorded validation objective among epochs 12–16.
No AP, test-set selection, or additional tuning is performed.

Feature spectra use saved local scalar 128, context scalar 128 and vector 16x3
features over the complete named role's fixed/uniform rows. Empirical row-weighted
covariance defines d95/effective/participation rank exactly as in crystal_vector.
Constant component count uses std<1e-6; RMS norm uses all scalar/Cartesian components.
Vectors use the contracted channel covariance for rank. All features must be finite.
Concentrated variance is not by itself proof of overfitting or sufficient state.

Nearest-training-feature checks fit per-channel standard deviations on training
rows only (floor 1e-6), then use Euclidean distance to a fixed 4096-row training
reference. Query 512 disjoint training rows and 512 random selection/test rows,
seed 20260928. Report nearest-distance median and p95. Counts are sample diagnostics,
not evidence of independent samples or a calibrated OOD detector.

Source/independent-melt ancestry must not cross roles. Feature input audit traces
forward() and the graph producer: coordinates, patch inverse indices and offsets
only; constant atom attributes and geometry masks. Labels, explicit time and
temperature, material IDs, source IDs and split roles are absent from forward().
Interface visibility uses the same radius<8 graph support. Deliberate label-side
population selection is recorded separately from input leakage.

Retrospective unseen-interface alarms use each preserved path's prefix before
first interface visibility or crystal entry, with two consecutive original
observations and all original paths retained as the denominator. See the
[unseen experiment definition](crystal_interface_unseen.md). This differs from
counting only which previous full-path alarms happened to be invisible.


## Explicit task-head refactor

The training/model refactor separates typed patch and spatial-context trunks from
task heads and expands training statements. Mathematical objectives, populations,
weights and selectors retain their definitions. Joint/rich-patch initialization
and state names are preserved; distance/control fresh initialization changes
when unused head construction is removed and receives a versioned architecture
identity. Historical continuations use their frozen sources. W&B wall-time stays
local and fixed baselines stay in summary; metric calculations are unchanged.
See [implementation and compatibility evidence](../code_cleanup_implementation.md#training-and-model-follow-up).
