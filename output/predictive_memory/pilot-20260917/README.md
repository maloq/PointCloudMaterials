# Predictive memory: initial allocated-H100 experiment

Launched detached on node53 in allocation 991900 on 17 September 2026 (Paris).
This is a new physical-prediction protocol, separate from the earlier
crystallization/smoothness fits. No simulation trajectories are generated.

Eight fits: positions and positions+velocities, each at 0/12/48 ps of actual
observed history, followed by separately trained 48 ps repeated-anchor controls.
All use 3,000 updates, the same source/center/anchor tuples and seed 20260917.
The queue then writes the paired source-bootstrap comparison automatically.

- [Scientific protocol](../../../experiments/predictive_memory_20260917/README.md)
- [Workflow and resume](../../../docs/predictive_memory.md)
- [Live queue status](technical/allocation-status.json)
- [Frozen launch plan](technical/allocation-plan.json)
- [Paired comparison, after completion](comparison/README.md)

Data audit: 150 existing independent-melt Al sources; 90/30/30 inherited
train/validation/test sources; 450 matched windows. Array content checksums and
source dynamics checked. Existing full-box float16 precision remains a limitation.
The first center of each inherited sorted four-center list is used; this is not
512 centers per source and is not a fresh uniform one-center sample.

Validation: 48 scientific and regression tests passed, plus a complete real-data
training/validation/checkpoint/export smoke run. Full 65-frame H100 timing with
16 tensor channels used about 16 GiB and 0.24 seconds per training step on the
benchmark source. This is a timing pilot, not an established scientific gain.

The launch plan has a 03:10 Paris cutoff, ahead of allocation expiry at 03:28.
Fits preserve optimizer, sampler and random states for exact continuation.
