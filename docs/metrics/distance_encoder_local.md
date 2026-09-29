# CD-MACE128 local-only evaluation

CD-MACE128 is the original joint distance/early-warning encoder; CD-MACE128-VC
adds variance–covariance regularization. Names are aliases for exact checkpoint
hashes. Intermediate checkpoints are explicitly provisional; final evaluations
use the completed 12-epoch checkpoint. They never feed back into model selection.

All predictors receive one local 128-dimensional scalar embedding. The encoder
uses the current observed nearest 80 atoms, radius 8 Å, edge cutoff 5 Å and two
interaction blocks. No context-patch features, velocity, temperature or explicit
time enters any predictor. The fixed Al64 all64 source/sample roles are retained.

The joint distance head is evaluated directly. A fresh local MLP is trained for
16 epochs on exactly the previous mixture of fixed at-risk and uniformly sampled
centers, each with half the training/selection mass and equal sources within each
half. Feature normalization uses only the focal embeddings of the original train
rows. Distance likelihood selection begins after epoch 12; it does not use AP.
This focal normalization differs from historical context exporters' all-patch
normalization; both new encoders use the identical focal protocol. Distance,
probability-threshold, missed/false alarms and visibility definitions are inherited
from `spatial_distance` and `spatial_confidence`, with contracts at each export.

Frozen linear and MLP transfer probes learn the existing five-bin onset hazard
from the single local embedding on original fixed train rows only. They use 16
epochs, batch 256, learning rate .001, equal-source weights, and validation hazard
NLL selection from epoch 12. Calibration uses calibration sources only. Results
at 3/6 ps (12 ps secondary) include raw/calibrated proper scores and raw AP as a
diagnostic. There is no temperature/time covariate and no encoder update. These
test transfer to future onset, separately from present crystal distance.

Representation spectra use source-weighted raw embedding covariance, reporting
participation rank, entropy rank and d90/d95/d99 on all fixed rows (descriptive),
test rows and test rows with no locally visible PTM crystal. These are linear
spectral dimensions, not nonlinear intrinsic dimensions.

Noise uses four outcome-independent test rows per source. Neighbor perturbation
RMS is .001/.005/.01/.03 times mean center-to-12-neighbor spacing, with the central
atom fixed. Report actual RMS in Å and relative to that spacing. Rebuild support
and edges on the fixed candidate set; no new neighbors are retrieved. Embedding
RMS and p95 changes divide by sqrt(2 tr(Cov_train)), using original train sources.

Temporal movement uses eight uniformly sampled frame origins per each of the 30
test sources, all 64 tracked centers, and their exact following 0.75-ps frames.
Frame draws are independent of labels. Report source-weighted RMS jump normalized
by sqrt(2 tr(Cov_train)), pooled state covariance rank, and the uncentered second
moment spectrum of dz/0.75. Movement d95 is the number of linear directions
containing 95% of increment energy. It includes drift. Overlapping sampled pairs
are correlated and do not count as independent trajectories. Coordinates and
identities follow the verified dense observed release; no interpolation.

Larger rank is not itself evidence of a better representation. Compare likelihood,
calibration, transfer probes, within-liquid spread, movement and noise together.


## Mechanism queue extension (2026-09-26)

Shared evaluation code now supports explicit initial/adapted checkpoint provenance and an optional256-unit probe. This family retains its existing checkpoint and default probe settings.


Tracking revision (2026-09-26): diagnostic frozen readouts and per-checkpoint
evaluations keep their logs and results locally. Associated final scores update
a recorded scientific training run through the API, without creating or
restarting runs. Scientific training remains online. This changes logging and
validates identity/hash before cached readout reuse; objectives, selectors,
metric calculations and historical exported definitions are unchanged.
