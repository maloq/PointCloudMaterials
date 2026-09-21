# BCR-v1 implementation verification

BCR is implemented and tested; this report is not a representation-quality result.

The clean MACE encoder exports one signed 128-dimensional invariant code. A separate
O(3)-equivariant decoder receives only that code, noisy patch geometry, species,
center/padding masks and noise level. The primary objective is boundary-weighted
unit-noise reconstruction. No JEPA, TDA, physical or variance/rank auxiliary loss
is used in the primary BCR arm.

## Completed checks

- 15 regression tests pass on an allocated GPU: decoder input isolation, nonzero
  encoder gradients, export/reload, rotations/inversion/permutations, FCC cubic
  symmetries, padding/loss scaling, Gaussian corruption, collision gradients,
  skew-cell images, boundary re-extraction, accumulation, exact CPU resume,
  CUDA parity, root bootstrap/matching and source-balanced evaluation.
- All six modes completed four fresh-noise GPU training updates with finite
  parameters: BCR, unconditional, frozen random, node denoising, matched VICReg,
  frozen matched VICReg. The frozen-control checkpoint loader and reconstruction
  evaluator were exercised end to end. These four-update runs are not comparisons
  of converged controls.
- Real-data optimization test: two complete radius-8-A Al patches, fixed artificial
  corruption, 200 optimizer updates. Noise NMSE fell from **1.01785 to 0.75258**
  (26.1%). Encoder gradient norm was **7.02e-5** at initialization. Repeated-export
  and rotation discrepancies were approximately **1.5e-8**, below the predeclared
  FP32 tolerances (atol 1e-6, rtol 1e-5).

The tiny cache has eight patches from two frames of one 400 K Lee2003 MEAM
trajectory. Coordinates are native float32, training d0=2.60606 A, reference
count=124.375, and conservative center-relative rounding bound=7.63e-6 A. All five
proposed noise levels pass the ten-times-uncertainty guard. All samples share one
prepared-liquid parent; there is no held-out generalization claim. Matching correctly
reports zero cross-root swap coverage on this fixture, rather than faking a gain.

The final verification receipt pins encoder/decoder configuration, tested source
hashes, data identity and test-suite hash. Larger training refuses a stale receipt.
FP32 eager is released; BF16/compile deliberately require separate parity validation.

## What remains a scientific experiment

A source-lineage-frozen, full-size matched study has not been launched. No G1
conditioning benefit, G2 information advantage, G3 scientific robustness advantage,
or crystallization improvement is claimed. Archived encoders are not input-matched
controls for the revised complete-radius/readout contract. Independent high-precision
roots and source exposure must be audited before final test construction; previously
consulted crystallization tests remain development evidence.

Training/evaluation instructions: `docs/bcr.md`. Scientific record:
`experiments/bcr_v1_20260921/README.md`. Recipe: `configs/bcr/real_overfit.json`.
Verification table: `tables/verification.csv`, definitions: `tables/METRICS.md`.
Machine evidence: `technical/gate.json`, `technical/correctness-tests.log`,
`technical/overfit.json`, `technical/control-smoke/results.json`.
