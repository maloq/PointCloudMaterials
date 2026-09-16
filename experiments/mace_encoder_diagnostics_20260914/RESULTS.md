# Forecast MACE encoder diagnostics — September 14, 2026

The exact forecast encoder contains strongly generalizable TDA information. The
results argue against describing it as “just overfit.” Its response to identical
inputs is numerically stable; its temporal fluctuations contain measurable local
topology changes. Patch membership and unresolved stochastic evolution both matter.

The tested checkpoint is `anchor_vicreg`, seed 20260910, epoch 02, SHA-256
`118ca618678ecc9b324ed8b2a93647f53fdd91976b1b7945ce253400e6a4495f`.
The output is the **256-channel MACE embedding before the VICReg projector**.
TDA supervision was disabled and this checkpoint has no TDA head. Therefore this
study fits new training-only linear readouts; it does not claim that a preexisting
TDA predictor was trained or evaluated.

## TDA generalization and overfitting

Sampled 5,760 patches from 30 independent Al MEAM melt lineages: 3,456 training
patches from 18 sources, 1,152 validation patches from six sources, and 1,152 test
patches from six other sources. Each source supplies 64 centers at each of three
frames. Source preparation seeds are disjoint and checked against the simulation
producer. Feature/target transforms use training data; validation selects ridge
regularization; test data does neither.

“Overall skill” below is the reduction in balanced H0/H1/H2 error versus the
training-mean target. “Local skill” removes each source/frame mean from predictions
and targets separately when scoring. This prevents an accurate frame-level mean
from being mistaken for accurate atom-to-atom variation.

| Target | Training balanced MSE | Test balanced MSE | Overall test skill, source-bootstrap 95% CI | Local test skill |
| --- | ---: | ---: | ---: | ---: |
| Observed-cloud TDA | 0.01651 | 0.01855 | 97.96% [97.69%, 98.28%] | 93.92% |
| FIRE-relaxed TDA | 0.03222 | 0.03608 | 95.60% [95.09%, 96.14%] | 85.97% |

Local training skill is 93.70% and 84.72%, respectively: there is no collapse
on held-out sources. Test error is roughly 12% higher than training error, so
these results do not assert an absence of any fitting gap. They establish useful
generalization under the measured protocol.

The result is not confined to one homology block. Observed local H0/H1/H2 R² is
97.15% / 95.00% / 88.99%; relaxed local H0/H1/H2 R² is
86.01% / 86.18% / 85.68%.

MACE reduces test error by **90.05% [89.43%, 90.68%]** for observed TDA and
**73.35% [71.72%, 75.42%]** for relaxed TDA versus a trained geometry baseline
using radial distances, q4/q6 and nearest-shell observables. Relative to a readout
whose training targets were shuffled within each source/frame, the reductions
are **89.24%** and **67.12%**. This shuffle retains frame-level trends, making it
a stronger control than a global label shuffle. Global shuffled-label and
temperature-only controls also perform much worse. Null shuffles use one recorded
random realization; intervals resample sources, not shuffle seeds.

The six test sources have been examined in earlier research. This is exploratory
source generalization within these Al MEAM conditions, not a new blind test,
unseen-material transfer result, or proof that TDA identifies physical phases.

## The requested diagnostics

All embedding error ratios use the training-only normalizer saved in a completed
forecast checkpoint. The reference signal is the measured mean squared 0.75 ps
increment of 144 tracked test patches over 17-frame paths: standardized MSE
**0.343407**. Ratios are **squared-error ratios**, not amplitude ratios.

| Diagnostic | Measured result | Interpretation |
| --- | --- | --- |
| Repeated inference, reordered batches, batch sizes 1/17/64, rotations and point permutations | All error/signal MSE ratios below **7.2 × 10⁻¹⁰** | Implementation variability is negligible at the tested cadence. |
| Small geometric changes with membership fixed | 0.005 Å jitter gives **0.0267%** of the reference change energy; 0.02 Å gives **0.422%** | Small perturbations have small responses. The jitter response grows approximately quadratically at these small amplitudes. |
| Atom-matched MD motion and neighbor retention | One-step matched displacement RMS **0.708 Å**; mean retention **93.35%** of the 79 neighbors. Spearman correlations with embedding-change magnitude: motion **0.453**, replacement fraction **0.463** | Both motion and support changes are associated with changes in the representation. These are descriptive associations, not independent causal estimates. |
| Controlled boundary membership changes | Replacing the outermost **1 / 2 / 4** atoms with the next outside neighbors, while other atoms stay fixed, gives **7.54% / 15.62% / 36.78%** of reference embedding change energy | A finite patch substitution can cause a substantial embedding jump without interior motion. Actual observed TDA also changes, by **2.61% / 4.91% / 11.92%** of its reference increment energy. |
| Precision and storage round trips | Global float32→float16 positions: embedding error **2.39%**, TDA-label error **2.35%** of their reference change energies. Local-offset float16: **0.000178%**; embedding float16: **0.00761%**. Compensated radial arithmetic versus FP32: **0.00000689%** | Global coordinate quantization is noticeable but does not dominate this measured 0.75 ps signal. Local coordinate, embedding storage and radial arithmetic errors are much smaller. |
| Temporal TDA and physical quantities | Observed-TDA increment skill **81.46% at 0.75 ps**, rising to **84.45% at 12 ps**; PTM labels change in **1.26%** of adjacent pairs | Much of the fluctuation encodes actual changing topology while discrete structural labels remain steadier. This tests decoding observed changes, not forecasting them. |
| Siblings and temporal correlations | Same-momentum siblings reach **0.83×** the ordinary-step change energy by **0.3 ps** and **1.03×** by **0.6 ps**. Ordinary embedding level correlation is **0.827 at 0.75 ps**, **0.773 at 12 ps** | A slowly varying level coexists with rapidly variable local structure. The one-parent stochastic ensemble supports unresolved physical evolution as a source of variability. |

The exact one-step embedding decomposition is
`total = motion with original IDs + membership change at the next geometry`.
Its motion energy is 98.33% of total energy; membership energy is 48.04%; their
signed cross term is **−46.37%**. They sum to 100%. Therefore “48% of all noise
comes from patch changes” would be an incorrect interpretation: the terms partly
cancel. Keeping initial neighbor IDs for the whole path also does not solve the
problem: its 12 ps embedding increment MSE is **1.675**, versus **0.444** with
instantaneous neighbors, as the original atoms move out of the local patch.

At 0.75 ps, embedding-change magnitude has Spearman correlation **0.316** with
absolute PTM RMSD/margin changes, **0.355** with q4, **0.278** with q6 and **0.447**
with observed-TDA change magnitude. Density alone is much less related (**0.055**).
These are moderate associations, not an identification of every fluctuating channel
with a unique physical quantity. Relaxed-TDA readout changes are retained, but no
intermediate full-cell quenches were computed to validate their temporal changes.

After subtracting each short track's own mean, the embedding lag correlation is
only **0.062 at 0.75 ps**, versus 0.827 after centering on the global training mean.
This separation is consistent with a persistent background and much less persistent
local fluctuations. Longer-lag track-centered estimates become negative; their
17-frame centering window biases them, so they do not determine a physical decay time.

## Precision and sibling limits

The high-precision comparison starts from an original retained float32 shooting
branch, not from casting an already-float16 trajectory back to float32. Global
rounding changes matched center-relative offsets by **0.0312 Å RMS**. With original
neighbor IDs held fixed, its embedding error is **0.626%** of the ordinary-step
signal, compared with 2.39% after neighbor reselection. This is evidence that
quantization can act through patch membership as well as coordinates.

The 144 precision examples have no PTM label flips, but **all are PTM Other**;
this does not validate classification precision near a crystal threshold. PTM
best-fit RMSD changes by about **0.0102 RMS** where both matches exist. Missing
templates occur in 11 reference and 12 rounded examples and are explicitly
undefined, not counted as perfect RMSD-zero matches. The precision comparison
and the ordinary temporal denominator are different cohorts, so 2.39% is a
calibrated sample comparison, not a universal quantization bound.

The sibling ensemble contains eight momentum groups with two thermostat streams
each, all at one parent configuration. Same-group comparisons use eight branch
pairs; different-group comparisons use 112 pairs. Pairs, centers and times are
not independent parent lineages, so no population confidence interval is claimed.
All 16 sampled centers across all branches/times remain PTM Other. The divergence
is evidence about liquid local-structure variability under the recorded stochastic
NVT protocol, not crystallization branching or deterministic chaos.

## Practical conclusion

Retain this encoder as a credible measurement of local topology. The evidence
points toward **a mixture of real fast structural variation and finite-patch
membership sensitivity**, with much smaller implementation noise. It does not
justify treating all unsmooth latent directions as numerical artifacts or
memorization. If the scientific forecasting target is slow crystallization,
evaluate predictions in the relevant physical readouts alongside raw embeddings;
good decoding of current TDA does not establish predictability of its future.

A smooth-boundary representation would be a reasonable separate ablation, but
freezing neighbor membership indefinitely is not supported by these tests.
Further sibling ensembles should cover independent parents and crystal-transition
centers before drawing population conclusions about irreducible forecast error.

## Artifacts and reproduction

See the [protocol and commands](README.md), active configs and source module linked
there. The full run is at
`${storage:analysis}/mace_encoder_diagnostics/forecast-seed20260910-20260914/`.
`plots/diagnostics.png` and `.pdf` provide the six-panel figure. `tables/` contains
TDA train/validation/test and per-source scores, paired comparisons, numerical
controls, membership interventions, temporal lags, continuous observables,
storage precision and sibling divergence, with frozen `METRICS.md` definitions.
`technical/` retains selected coordinates/IDs, embeddings, labels, readouts,
normalizer, stage source snapshots, implementation hashes and execution logs.

The focused diagnostics and result-layout suite passed **10 tests**. Actual-data
verification matched 24 recomputed TDA labels bitwise and all 144 checked PTM
labels to the established assay. Replaying 72 embeddings across three stored
forecast sources matched the cache at the expected float16 rounding error
(99.85–99.98% of channels landed in identical float16 bins). The 16 siblings had
identical initial positions; all eight same-momentum pairs had bitwise-identical
initial velocities, while different momentum groups differed. Timelines and
initial boxes also matched the declared protocol.
