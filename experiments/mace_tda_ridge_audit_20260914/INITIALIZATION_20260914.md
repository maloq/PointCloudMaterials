# Topology readouts before encoder training

Frozen random MACE features support surprisingly strong relaxed-topology readouts.
After choosing ridge regularization on validation sources, the original MLIP
checkpoint has only a 1.15% mean error advantage over three random initializations;
its paired source-bootstrap interval includes zero. VICReg finetuning improves the
MLIP readout by 7.64%. Adding TDA supervision improves it by another 0.089% relative
to VICReg. These statements concern this 256D encoder-space linear probe and the
previously examined simulation cohort.

## Matched protocol and results

The test contains 4,608 local neighborhoods from six independent simulations.
Training uses 13,824 neighborhoods from 18 other simulations; validation uses
4,608 from six more. Every encoder sees the same single noisy anchor cloud of 80
atoms and returns 256 raw pooled scalar features. Targets are the same 144D
relaxed MEAM topology descriptors. Readouts fit training labels only and use the
same training-only feature standardization and H0/H1/H2-balanced target scales.

| Frozen encoder | Original ridge alpha 1 | Validation-selected ridge | Selected alpha(s) |
|---|---:|---:|---|
| Random MACE, three seeds | 0.048916374 | 0.036271123 | 1e-7, 1e-7, 1e-7 |
| Original MLIP MACE, one checkpoint | 0.037652692 | 0.035854029 | 1e-6 |
| MLIP + VICReg, three seeds | 0.034453996 | 0.033113714 | 1e-8, 1e-4, 1e-7 |
| MLIP + VICReg + TDA, three seeds | 0.034449982 | 0.033084214 | 1e-8, 1e-4, 1e-6 |

Lower balanced MSE is better. The training-mean baseline is 0.839816500.
Shuffling the ridge training-target pairing gives group means 0.8466–0.8532.
Thus even the random baseline's result requires a trained, correctly paired
topology readout. It is not a zero-shot topology prediction by random weights.

| Comparison | Error reduction at alpha 1 | Reduction with validation-selected alpha | 95% source-bootstrap interval, selected alpha |
|---|---:|---:|---:|
| Random → MLIP | 23.026% | 1.150% | −0.422% to 2.639% |
| MLIP → VICReg | 8.495% | 7.643% | 6.672% to 8.693% |
| VICReg → VICReg+TDA | 0.0117% | 0.0891% | 0.0354% to 0.1466% |

Bootstrap draws resample the six held-out simulations after averaging each
method's available model seeds. They are conditional on these fitted models;
they do not quantify variation over other MLIP pretraining runs, all possible
random initializations or unseen simulation regimes. The small tuned TDA effect
is consistently positive under this conditional bootstrap, while the fixed-alpha
TDA interval includes zero. Neither observation establishes general equivalence.

## Interpretation

MACE's graph construction, distance basis, angular features and invariant
many-body operations already express local geometry. Random learned weights mix
those geometric quantities into features. A supervised ridge readout can learn
their relationship to topology without any topology training of the encoder.
The random control retains the same architecture, element table, cutoff, and
neighbor-count normalization scalar (61.964672446250916); it does not remove all
geometric or data-informed architectural priors.

The regularization comparison matters: alpha 1 suppresses informative directions
of random features much more strongly. Its apparent 23% MLIP advantage shrinks
to about 1% after validation tuning. The geometric architecture plus the trained
readout explains much of the observed recoverability; these results do not support
attributing that recoverability mainly to explicit TDA supervision. VICReg still
improves this assay, but descriptor reconstruction alone does not establish better
crystallization classification or transition-time forecasts.

## Controls and reproducibility

- The original MLIP checkpoint is byte-identical to the installed MACE download
  cache for `mace-small-density-agnesi-stress.model`, with SHA256
  `d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9`.
- Random models invoke the native MACE constructor with seeds 20260910–20260912.
  All 28 learned parameter tensors differ from MLIP; no learned weights are
  transferred. Unused atomic energy offsets and energy scale/shift are neutralized.
  Each complete random model is retained. Same-seed construction reproduces the
  exact state; a different seed changes it; the original model remains unchanged.
- All frozen controls use the trained encoders' exact numerical settings,
  including compensated BF16 radial layers. Native full-FP32 versus accelerated
  outputs differ by at most 9.86e-6 in relative L2 on the three input cross-checks
  per model. Full extraction leaves each model state unchanged.
- All 92 prepared input checksums, original coordinates and target arrays,
  source splits and training scales are revalidated. The six trained feature
  arrays come from the preceding exact-checkpoint audit. Their original alpha-1
  errors reproduce, and their initializing MLIP checkpoint hashes match.
- The initial grid 0.001–1000 placed all validation minima at its lower boundary.
  The extended grid is 1e-9–1000; all final selected values are interior. Selection
  uses validation only. A float64 SVD solver avoids squared conditioning at small
  penalties. At alpha 1 it agrees with independent normal equations to a maximum
  absolute prediction difference below 8.9e-13 across all ten encoders.
- Eleven targeted tests pass, covering random initialization, independent/SVD
  ridge math including near-duplicate features, topology block weighting, GPU
  checkpoint loading, and result-layout/metric contracts.

Run recipe and command are in the [protocol](README.md#where-topology-information-comes-from).
The initial full-FP32 / narrow-grid run remains preserved at
`output/mace_tda_ridge_audit/initialization-20260914`; the final comparison is
[initialization-matched-20260914](../../output/mace_tda_ridge_audit/initialization-matched-20260914/README.md).
That output includes exact predictions, model artifacts, per-seed metric CSVs,
frozen metric definitions and implementation source, and three PNG/PDF figures.
