# Single-frame VICReg / TDA ridge audit

The exact six retained checkpoints and the original 18/6/6 simulation split are
declared by the audit recipe. Every checkpoint is reloaded strictly and each of
the 23,040 prepared single-frame anchors is encoded again. No cached embeddings
are used as model inputs. Historical test arrays are comparison evidence only.

Two distinct feature spaces are evaluated: the 128-dimensional VICReg projector
used in the historical topology report, and the 256-dimensional encoder output
used by future-embedding forecasting. The trained TDA head supervises the former.
Every ridge probe, including the VICReg-only encoder's probe, fits TDA labels from
training sources. This supervision of the readout is separate from encoder training.

`mse` is the independent float64 ridge result. Training feature columns are centered
and scaled by their population standard deviation (constant columns use scale 1).
Training targets are centered, with an unpenalized intercept. We solve
`(X.T X + I) W = X.T Y` at alpha 1, then predict held-out raw 144D descriptors.
The repository procedure independently standardizes features, transforms targets
with the stored training transform, fits sklearn Ridge at alpha 1, and restores raw
units. Linear target rescaling commutes with ridge; finite precision may differ.
`repository_mse` is its result; `solver_mse_difference` is the absolute score difference.

`H0`, `H1`, `H2` are raw descriptor block MSE divided by squared training block
scale, over coordinates [0:16], [16:80], [80:144]. `mse` averages those three
blocks equally, not all 144 pixels equally. Scales and target means are independently
verified against training-only rows and checksummed original target files.

`*_shuffled_training_targets` fits a probe after randomly permuting training target
rows, leaving evaluation labels unchanged. `training_mean` predicts the raw training
target mean. Both test whether the measured low error requires paired information.
`split` identifies validation or test; all readouts fit exclusively on training rows.

Paired uncertainty averages row errors over the three fixed model seeds, groups by
the six test simulations, then resamples those simulations together 4,000 times.
`relative_reduction = 1 - mean(TDA source errors)/mean(VICReg source errors)`;
positive favors TDA. `ci95` is its 2.5/97.5 percentile interval, conditional on the
three fitted seeds. This previously examined test set is exploratory; an interval
including zero does not establish equivalence.

Gradient checks run the original training loss with identical weights, inputs,
RNG and restored buffers, repeating real targets twice before changing the target
pairing. CUDA gradient reductions may vary numerically: the disabled-TDA gradient
relative L2 difference must be below 1e-4 and the loss relative difference below
1e-6 (absolute tolerance 1e-7). The unchanged-target repeat measures that drift.
Autograd must report the target tensor as unused when TDA is disabled, and a
nonzero target derivative when enabled. This graph check is independent of CUDA
parameter-gradient rounding. The initial diagnostic showed unchanged-target
relative gradient drift 1.7e-5, motivating the explicit 1e-4 tolerance. They check both
label independence with TDA disabled and nonzero TDA gradients in encoder, projector
and head when enabled. No optimizer step updates any checkpoint. Gradient magnitudes
describe this diagnostic batch, not an epoch-average training contribution.

Historical score reproduction checks identity/order of test rows, sources, contexts
and targets; it records fresh-versus-saved projector discrepancies. Scalar errors
are independently recomputed from saved prediction arrays. Current loss, feature
selection and dataset-item functions are AST-compared with each run's frozen source.

The main fresh evaluation uses `highest` float32 matrix precision. The original
standard analysis sets `high`, permitting TF32 for the projector; MACE's explicit
local precision scopes remain unchanged. The additional precision stage reapplies
the saved projector to fresh encoder features using the original `high` setting,
then repeats repository ridge evaluation and the saved trained TDA head readout.
It records discrepancies against original projector arrays and prediction scores.

## Initialization controls

The `initialization` stage keeps the same audited coordinates, targets, splits,
training scales, and 256D raw encoder output. It compares one frozen original MLIP
checkpoint, three fresh random native MACE initializations, and the six previously
audited trained encoders. The original checkpoint's architecture is extracted with
MACE's `extract_config_mace_model` and instantiated with the native constructor;
no learned tensor is transferred for the random control. Every parameter tensor
must differ from MLIP. Atomic energy offsets and energy scale/shift are neutralized;
energy readouts are not used by any encoder. Architecture, analytic geometric bases,
element table, cutoff and the original neighbor-count normalization scalar are
shared. Native versus accelerated inference is checked on three real inputs,
and the frozen model state must be unchanged after extraction.

`alpha` is the ridge penalty; the primary comparison fixes it at 1. The sensitivity
analysis selects each encoder's alpha by minimum validation balanced MSE over the
recipe's declared grid, with first-in-grid tie breaking. Test scores never select
alpha. No validation rows are added to the readout fit. `selected_mse` reports the
test error with that validation choice. `alpha1_mse` is the original protocol's
test error. Group scores average three seeds, except MLIP's single fixed checkpoint.
Confidence intervals use the source bootstrap above after averaging each group's
available seeds independently; they describe source uncertainty conditional on
these model instances, not initialization or training-population uncertainty.

`shuffled_train_mse` permutes training targets before fitting alpha-1 ridge.
`feature_participation_rank = (sum(s**2))**2 / sum(s**4)` uses singular values of
training-standardized encoder features; `constant_channels` counts exact zero
training standard deviations. These are descriptive feature diagnostics. The
initialization plots show seed points, group means, and equally normalized topology
blocks. Scores measure descriptor reconstruction, not crystallization accuracy.

Initialization sensitivity uses a float64 SVD ridge path to avoid squared conditioning at small penalties. At alpha 1 its predictions must agree with the independent normal-equation solver (rtol 1e-7, atol 1e-9). The completed initial grid placed every validation optimum at its lower boundary 0.001; the declared extension reaches 1e-9. All final controls inherit the exact inference performance settings (including compensated BF16 radial layers) from the audited trained checkpoint configurations; native parity remains a full-FP32 cross-check. The first full-FP32 control extraction and narrow grid are retained as separate preliminary results.

## Direct comparison without a trained readout

The `direct` stage uses only held-out rows of the verified raw 256D features from
the three random MACE encoders and original MLIP checkpoint. It fits zero
parameters: no ridge, feature standardization, PCA, target scaling, calibration or
hyperparameter selection. Euclidean and cosine embedding distances are both
reported. Each target block uses its raw Euclidean distance independently, so no
cross-block scale needs to be fitted. Summary scores equally average the three
blocks and available model seeds. The original three seeds remain separate in
the detailed table; MLIP is one fixed checkpoint.

`spearman` is the Pearson correlation of average ranks of embedding distances
and raw TDA-block distances on the same unordered pairs. It measures geometric
agreement, from -1 (reversed distance ordering) to 1 (identical ordering). Global
evaluation chooses 100,000 distinct unordered pairs once without replacement
with the declared seed, shared by all models and both distances. Within-frame
evaluation uses all 32,640 unordered pairs among each frame's 256 structures.
The 18 within-frame scores are averaged equally; this is not a pooled correlation.
No correlation p-values are reported because pairs share structures and sources.

`neighbor_overlap` averages `|N_embedding(i) intersect N_TDA(i)| / 10` over all
query structures. Each set contains the closest ten other structures in the
eligible pool; self is excluded. Nearest-neighbor searches use every eligible
structure, even when global correlation uses sampled pairs. Exact distance ties
use stable original row order, and both embedding and target boundary-tie row
fractions are recorded. `chance_neighbor_overlap = 10/(pool_size-1)` is expected
overlap for independent uniformly chosen sets: 0.00217061 globally and 0.03921569
within a frame. Scores are fractions in CSVs and percentages in plots.

`shuffled_spearman` and `shuffled_neighbor_overlap` apply one fixed random
permutation to embedding row identities within the eligible pool, keeping TDA
identities fixed. The permutation is shared by models and distances; these are
descriptive negative controls, not permutation-test significance levels. Global
pool size is 4,608 from six held-out simulations; each within-frame pool fixes
simulation, time and temperature. A mean score can hide seed/block/frame variation;
the full table retains those breakdowns. No TDA prediction is generated, so ridge
MSE and crystallization accuracy are not metrics for this stage.

## Exact 144D topology descriptor

The checksummed producer is `src/analysis/liquid_structure.py:persistence_image`.
It constructs a GUDHI alpha complex on the supplied physical-coordinate cloud,
computes H0/H1/H2 persistence over field 2, removes infinite-death intervals, and
takes square roots of nonnegative filtration values to express birth/death radii
in Angstrom. Intervals with death radius above 3.5 Angstrom are excluded.

- H0 (coordinates 0:16): sum of Gaussian bumps at component death/merge radii,
  evaluated at 16 equally spaced points from 0.7 to 2.5 Angstrom; width 0.10
  Angstrom; divide by N-1. These are sampled smooth values, not integer counts.
- H1 (16:80) and H2 (80:144): separate 8x8 surfaces. Birth radii use eight points
  from 0.7 to 3.0 Angstrom, and lifetime = death minus birth uses eight points
  from 0.025 to 1.1 Angstrom. Gaussian widths are 0.15 Angstrom in birth and
  0.09 Angstrom in lifetime. Each interval contributes its lifetime times the
  product of the two Gaussian factors. Sum, divide by N-1, and flatten the grid
  in NumPy row-major order. H1 summarizes loops/tunnels; H2 summarizes cavities.

For this cohort N=80, so normalization is by 79. Targets use float16-stored
offsets after full-cell Lee2003 MEAM FIRE relaxation to force tolerance 0.01 eV/A,
keeping the original hot-selected atom identities. Encoders receive the original
unrelaxed anchor coordinates. The target is continuous geometric topology, not
PTM/FCC/BCC labels, crystal probability, or future transition time.
