# Joint spatial crystal distance and direction

Historical fixed-quota protocol, stopped at the user's sampling revision on
28 September. Partial checkpoints and original W&B identities are preserved.
The [replacement comparison](../crystal_vector_20260928/README.md) uses independent
random batches and fresh fits from the same original encoder initialization.

CDV-MACE128 tests whether direction supervision and variance/covariance
regularization improve the exported state of a jointly trained spatial encoder
and vector-message context predictor. One snapshot; no temporal inputs.

Three matched treatments: distance+direction+VCReg (primary), distance+direction,
distance only. Initialize the same completed snapshot CD-MACE128 spatial weights;
train shared MACE, typed export and context predictor together. Vector messages
was chosen from the earlier validation distance likelihood comparison.

Use the fixed Al64 train/selection/calibration/test ancestors and all64 evaluation
rows. Add the existing outcome-blind 16 uniform centers per fitting/selection
frame. Counts: train 63,251; selection 29,667; calibration 16,848; fixed test
45,291. Existing scan paths bring total prepared rows to 177,929 across 150
sources. No new MD data. This first matched experiment is Al; the initialization
was previously fitted on Al/Mg/Ti/Ta. It does not establish new-material transfer.

Every batch of 256 contains 64 crystal interiors, 64 liquid centers within 8 A,
64 at 8–20 A and 64 beyond 20 A, including censored cases. Importance correction
preserves the original equal-source half-fixed/half-uniform predictive objective.
All arms share draws, data, 16 nominal epochs and the epoch-12 minimum selection.
An epoch is 248 replacement-sampled updates, not an exhaustive data traversal.

All use the same new rotation-covariant distance-only context covering; historical
lab-stencil results remain external references. Outputs describe current nearest
established crystal, not future onset. Directional uncertainty and undefined
directions are explicit. Selection is predictive likelihood, never AP or VCReg.

[Metric definitions](../../docs/metrics/crystal_vector.md) ·
[Recipe](../../configs/crystal_vector/al64_20260927.json) ·
[Execution](../../docs/crystal_vector.md).
