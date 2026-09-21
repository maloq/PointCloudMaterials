# BCR-v1 metrics

The primary training objective is per-environment noise MSE: sum of C2 clean-radius
weights times squared three-vector noise error, divided by three times noncentral
weight sum. Padding and the distinguished center receive zero weight. Environments
are averaged equally under root/source/time-block balanced sampling. There are no
physical/TDA/temporal or representation-regularization losses in BCR.

C2 support is 1 below 0.8 R, 0 above R, and 1-10u^3+15u^4-6u^5 in between.
Shell diagnostics are unweighted interior r<=0.65R, middle 0.65R<r<=0.8R, outer
0.8R<r<R; absent shells are undefined, not zero. Coordinate RMSE in angstrom is
sigma sqrt(noise MSE), per Cartesian component. Vector RMS is sqrt(3) times this.

G=(other NMSE - true NMSE)/max(other NMSE,1e-30), retaining negative values.
Other may be an independently trained unconditional decoder or a matched code swap.
Swaps are partial one-to-one derangements across distinct roots; unmatched anchors
are excluded from BOTH paired terms. Exact temperature and training-fitted quintile
bins of count/density/q6 define strict matching; condition/count-only and unrestricted
cross-root matches are reported separately. Coverage is a fraction of anchors.
Corruptions/shuffles average per anchor, then anchors average per root; 1,000 paired
root bootstrap draws give a percentile interval. One root has no interval. These
intervals exclude training-seed uncertainty. Raw reconstruction summaries also retain
per-anchor values. Noise banks are deterministic and independently configurable.

Structural targets: 12 smooth radial bins plus length/count/density statistics;
q4,q6 and normalized Wigner-3j cubic contractions; 8 radial Gaussian channels at
l=0,2,4,6 with all symmetric cross-radial Gram entries. These are independent
measurements, not statistically independent information. Feature and per-target
scaling fit training anchors only. Standardized RMSE pools equally scaled targets;
R2 uses evaluation-population centered variance, undefined for constant targets.
q6<0.35 defines a fixed noncrystalline diagnostic, not a ground-truth phase label.
Ridge regularization is selected from 11 log-spaced values 1e-6 through 1e4 on
root-disjoint development data; two-layer 128-wide SiLU MLP trains 500 updates.
Recall@20 uses only other-root candidates, with training-standardized rich geometry
as reference; raw and train-standardized code distances are separate measurements.

Rank is centered covariance participation ratio (sum lambda)^2/sum lambda^2,
with trace, spectrum, channel means/stds and mean norm retained. No rank selection.
Perturbation sensitivity divides raw code change by a fixed training-calibrated
pair-distance scale; raw changes and scale remain visible. Fixed membership and
radius re-extraction are distinct assays. Low-amplitude tests below declared native
coordinate uncertainty are excluded explicitly.

Tiny real-data overfit and synthetic G0 tests establish implementation correctness
only. No G1/G2/G3 scientific success follows from lower training loss.


Table export: 2026-09-21T12:08:02.793169+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
