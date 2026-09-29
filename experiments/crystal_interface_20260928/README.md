# CIV-MACE128: learning distance to the crystal interface from either side

**Completed:** all three fits and evaluations. [Results, including the matched
crystal-set study](RESULTS.md) report interior distance, direction, warnings and
embedding diagnostics separately.

Follow-up: [liquid-only external-crystal localization](LIQUID_DISTANCE.md) and its
[interim training-drift diagnosis](LIQUID_TRAINING_DRIFT.md).

Question: can a geometry-only, jointly trained MACE encoder and equivariant spatial
predictor locate the crystal edge from within crystal as well as from liquid?

The target is unsigned distance to the crystal-side interface atom layer. Interior
distance is positive; only layer members have zero target. Direction points toward
the nearest interface atom from either side. The interface touches a connected
non-crystal region of >=64 atoms through the existing periodic 3.6-A Al neighbor
graph. Filtering small disordered components avoids treating every isolated PTM
defect as an outer edge. Large internal pockets can still define an interface.
This is an explicit atom-layer definition rather than a continuum dividing surface.
See the [precise definitions and limitations](../../docs/metrics/crystal_interface.md).

| Treatment | Prediction objective | Embedding regularization |
| --- | --- | --- |
| distance_only | Censored distance likelihood + proximity log loss | None |
| distance_direction | Distance + conditional direction + proximity likelihood | None |
| distance_direction_vcreg | Same joint likelihood | Scalar/vector VCReg |

All three use the original completed snapshot CD-MACE128 parent, fresh optimizers,
one matched seed, 16 epochs, width/export 128 and random batches of 256. Shared
encoders process all 25 patches, with the same vector-message spatial context and
32-A maximum support. No time, temperature, velocity, species, material identity,
phase label or interface label is a model input. Selecting checkpoints uses
validation predictive likelihood, never AP or test performance.

The fixed Al64 all64 source contract is unchanged. Preserve original fixed and
scan queries and existing uniform fitting queries. Add outcome-blind uniform
calibration/test centers to evaluate interiors, which are absent from the fixed
liquid-at-risk track. Report fixed, uniform and scan populations separately, and
split interface metrics by crystal interior, layer and exterior. The principal
interior result is held-out uniform-source distance error and calibration.

The existing CDV-MACE128 crystal-set experiment remains a separate task. Direct
absolute metric comparisons across the two target definitions are not matched
encoder comparisons. This study changes the target while holding the three new
treatments matched to one another; a later common readout can compare representations.

[Recipe](../../configs/crystal_vector/interface_al64_20260928.json) ·
[Execution](../../docs/crystal_interface.md).
