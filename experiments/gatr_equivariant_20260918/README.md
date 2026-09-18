# Are GATr's internal directions persistent or spatially ordered?

Completed audit of the same frozen Al-only v6 GATr checkpoint 1216 as the
[trajectory comparison](../trajectory_stability_20260918/README.md).
[Full findings](../../output/gatr_equivariant/al-v6-node07-20260918/RESULTS.md),
[interactive 3D viewer](../../output/gatr_equivariant/al-v6-node07-20260918/explore.html),
[summary figure](../../output/gatr_equivariant/al-v6-node07-20260918/plots/overview.png).

## Findings

- Directions are rotation covariant (relative RMS residual at most 1.7e-5), but
  reorient rapidly at the saved 0.75 ps cadence: about 78° mean turn and 38%
  above 90° for the training-selected point triplet before the final MLP.
  Earlier block-1 directions give a similar mean turn. This does not resolve
  continuity between saved frames.
- Weak-vector filtering retains 99.6% of adjacent pairs; local cage-rotation
  correction leaves the mean turn essentially unchanged. FCC→FCC pairs also
  have very weak directional memory.
- Neighbor alignment is weak and short-range: primary P1≈0.064 below 4 Å,
  approximately zero at larger separation. Earlier block-1 nearest-neighbor
  alignment is stronger, still short-range. Shared neighborhoods remain a
  possible source of the short-range correlation.
- Final returned multivectors are discarded. Erasing directional coefficients
  before the final MLP has no detectable z effect on the control cohort, while
  earlier erasure changes z. Do not optimize discarded directions and assume
  this smooths the exported invariant descriptor.

## Protocol

Forty fixed atom identities across ten held-out Al MEAM trajectories (400,
450,500,510,520 K), 801 frames each, 0–600 ps. Same raw observations as the prior
audit. Training reference: 420 observations from five distinct training sources.
Three internal stages, eight multivector channels and four SO(3) vector triplets.
Representative channel chosen by training RMS norm, not test persistence.

Spatial sampling adds 22,381 local encodings: two 128-atom patches and 64 seeded
uniform centers in each of 70 snapshots, deduplicated by identity. Measures
signed direction and unsigned-axis alignment, exact phase-preserving shuffle
expectations, nearest-80 overlap and phase-restricted pairs. Geometric baselines
include 7 Å/full-support density dipoles and a shape-tensor principal axis.
Source bootstrap stratified by temperature; no pair-level independence assumed.
The primary feature is not a predicted velocity, physical point or crystal axis.

[Metric definitions](../../docs/metrics/gatr_equivariant.md) specify masks,
normalizations, uncertainty and interventions. Historical test use makes this
an exploratory analysis, not a new blind benchmark. Computations used A100 on
node07 and conda `pointnet-torch214`; no retraining.

## Reproduce

Use a valid allocation on node07 exposing an A100:

```bash
conda run -n pointnet-torch214 python -m src.research.gatr_equivariant \
  --config configs/analysis/gatr_equivariant.json
```

The default runs temporal/spatial extraction, intervention controls and reporting.
Stages `--stage temporal`, `--stage spatial` and `--stage report` are available.
The standalone controls command is `python -m src.research.gatr_equivariant.controls
--config configs/analysis/gatr_equivariant.json`. Results retain the frozen
checkpoint, source receipts, metric implementation hashes and numerical controls.

Follow-up research: train a physically targeted, symmetry-aware covariant readout
and norm-aware temporal consistency; test noncollapse and angular sensitivity.
Direct-z smoothing of the current invariant embedding remains a separate task.
