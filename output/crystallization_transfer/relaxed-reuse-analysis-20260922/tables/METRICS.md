# Relaxed-reuse forecast figures

The input is the completed eight-fit archived-data experiment. No fitting,
relaxation or test-set recalibration occurs. All arms have identical test indices,
original MD event labels and common future physical/latent targets. Natural-test
precision-recall, AP and reliability use equal total weight per test source;
12 ps AP must reproduce the original metrics exactly. Timing bars are conditional
on detection and must be accompanied by raw missed-positive fractions.

Fixed-event curves select one onset and same-source surviving control at every
requested lead bin. For lead L, select the latest available archived origin with
actual lead in [L,L+12) ps. Every event uses distinct observed origins; no temporal
interpolation or new inference is used for these CDF curves. The same event/control
cohort is retained at all leads. Event selection uses labels and availability only.
Matched AP has balanced50% prevalence and equal total source mass. Restricted-time
MAE uses the entire96 ps predictive distribution, placing surviving probability
at96 ps; it includes all cases regardless of alarm detection. The existing
structured-context event evaluator produces1000 whole-source bootstrap replicates;
intervals describe source uncertainty, not training-seed uncertainty. Actual leads
and source/event counts are retained in exported metrics.

Physical example forecasts replay selected frozen checkpoints and original
normalizers on the same archived inputs. Deterministic event CDF replay must
agree within2e-5. Mixture envelopes are pointwise model-sample percentiles, not
coverage guarantees. Example events come from a fixed random sample of distinct
sources without examining forecast scores.

UMAP and StandardScaler fit training sources only. Up to24 windows per source
are sampled identically across encoders; test coordinates use transform. Maps
are independent and their axes are not aligned. Future labels only color the
plot; they do not fit the map.

Real atom illustrations retain the same80 observed-nearest identities in both
coordinate panels. Coordinates are center-relative, periodic, and use the same
rigid camera. Displacement arrows and coordinate panels use the same physical scale,
without magnification. Full-context diagrams use each checkpoint's actual support (original
full-radius versus relaxed nearest80 candidates). Quenched full cells inherit
archived float16 precision; no synthetic lattice or relaxation is introduced.

For plotted PR/calibration risks and the AP replay check, reproduce the original
short/coarse hazard-logit conversion and clipping before reconstructing cumulative
risk. This matters for tied low-risk diffusion outputs; bypassing the evaluator's
clipping can change tie rankings and fail to reproduce its reported AP. Raw saved
CDFs remain the inputs to the fixed-event analysis and event-CDF illustrations.


Table export: 2026-09-21T22:53:38.595858+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
