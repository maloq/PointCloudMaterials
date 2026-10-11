# Unrelaxed response geometry experiments

Two recipes implement the October3 requested response-only comparison:
[existing frozen models](../../configs/analysis/response_geometry_clusters_20261003.json)
and [expanded collection and fits](../../configs/simulation/response_geometry_20261003.json).
All training and evaluation inputs are unrelaxed. Primary clustering diagnostic is
many-to-one agreement at fixed K7; occupancy, effective cluster count, seed stability,
ARI and one-to-one agreement remain supporting diagnostics. No descriptor/crystal
labels enter encoder training or checkpoint selection.

The first job compares six existing likelihood-selected checkpoints (value-only and
response-weight1, three seeds) in native z128, training-covariance-normalized z128 and
predicted256-dimensional future features. Fits use the identical74,880 training
observations; all24,960 held-out observations retain their identity. Three fixed
clustering seeds are reported with no best-seed selection. No MD or online W&B runs.

The new simulation collection retains all135 original inputs and adds630 training
inputs: seven rank-selected center/frame pairs for each of90 training sources. Final
counts720/15/30 train/selection/test. Eight stochastic training futures and two
initial perturbation directions per parent; selection32 values, test32 values plus
8 disjoint response streams. Original training seeds are reused for an independent
execution comparison. Every completed batch publishes to STORE before scratch purge.
No complete new position trajectories are stored; query values, derivatives, seeds,
initial native-precision environments and provenance are retained.

A single MACE simulation provides legacy256 change features, rich256 change features
and rich256 endpoint-plus-change features. Rich observations are generic smooth
radial/angular quantities through l6 in two radial windows. Their normalization
uses training inputs only. A24/36A convergence gate on four training inputs and a
new-target common-noise finite-difference check precede collection.

Eight treatments ×three seeds (24 fits): legacy90 value/response, legacy720
value/response, rich-delta720 response, rich-joint720 value/response, and rich-joint720
response with future-distribution distance supervision. All receive400 optimizer
updates of256 samples;90-parent fits sample with replacement,720-parent fits without.
Microbatch8 is an explicit higher-derivative memory deviation. Native128-channel/
128-dimensional geometry-only architecture and nonlinear prediction head are held
fixed. The distance term uses independent-shot noise correction, retains negative
estimates, excludes identical parent pairs, and does not use phase labels.
Validation feature Gaussian NLL chooses checkpoints. Different feature-target NLLs
are not directly compared. Scientific training stays W&B online in the existing
project, with stable resumable IDs; final clustering updates those runs by API.

In conda pointnet-torch214:

```bash
python -m src.research.response_geometry.clusters launch --config configs/analysis/response_geometry_clusters_20261003.json
python -m src.research.response_geometry.queue prepare --config configs/simulation/response_geometry_20261003.json
python -m src.research.response_geometry.queue submit --config configs/simulation/response_geometry_20261003.json
```

The frozen comparison requests one GPU. Expanded collection/training requests two
GPUs and launches one process per GPU, with shared parent locks and archived batch
resume. A24h allocation can requeue twice at expiry, preserving optimizer/RNG state.
Scientific/numerical failures stop the pipeline; they never trigger a weaker recipe.
Execution is frozen before submission. technical/launch.json, per-stage receipts and
lane logs report progress; all recorded failures remain alongside subsequent attempts.

Output links: output/response_geometry/frozen-spaces-20261003 and
output/response_geometry/unrelaxed-expanded-20261003, backed by STORE experiments.
Simulation scratch/archive and IDS cache are explicit in the portable recipe.
Feature vectors used for clustering stay in memory; saved predictions, assignments,
centers, checkpoints and metrics are retained. Numerical definitions and implementation
hashes accompany CSVs via the metric publication workflow.

[Clustering definitions](../metrics/response_geometry_clusters.md) ·
[Training definitions](../metrics/response_geometry_training.md).

## Additional existing GPU

The October3 node58 RTX PRO6000 helper ran in allocation1019795 beside the two
L40S collector lanes in job1020301. It called the frozen collector with unchanged
seeds, precision, batch sizes and scientific identity, sharing parent locks and
the STORE archive. It considered parents in reverse order and yielded within128
parent indices of the coordinator or with fewer than15 allocation minutes left.
Collection created no W&B runs.

`technical/execution-v1/helper-1019795.json` records the completed helper;
`technical/data-complete.json` records all765 sealed parents. The one-off helper
was removed from the live checkout on 2026-10-11 and remains in commit `7d91467d`.
Its checksum, launch receipt and log remain under `technical/execution-v1/`;
the frozen simulation and training recipe are unchanged.
