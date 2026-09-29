# Joint MACE distance and early spatial detection

The shared trainer also supports the separately defined `distance_encoder_history`
protocol. Its exact MD-history joins, temporal predictor, warm-start and changed
visibility semantics are documented in that family's contract. Snapshot calculations
below are unchanged; historical exports keep their frozen producer hashes.

The encoder and head are both optimized. The initial encoder is the earlier
observed Al onset-likelihood model; the distance head starts fresh. This is
crystallization-supervised training, with no physical reconstruction objective.

Distance labels follow the prior confirmed-component protocol: nearest atom in
a ≥64-atom crystal component whose lineage has already been confirmed by the
current observation. Reference construction uses full-cell PTM and past history;
neither the labels nor history enter the encoder. Initial frames without the
full confirmation history are omitted by a fixed frame rule. Static structures
are excluded because they lack this reference history. The label source, raw
frame mapping and graph/chunk checksums are recorded per source. All existing
dynamic structural centers and subsequent 3-ps-spaced frames are retained.

Coordinates and distances are multiplied by the established fixed material
length factor, relative to Al. Predictions and training thresholds therefore use
Al-equivalent Å. No material ID, scale, temperature or time value enters the
encoder/head. Native Al distances equal physical Å. Conversion for other
materials requires dividing outputs by the known preprocessing factor outside
the model. Cross-material thresholds must not be labelled native physical Å.

The head predicts a zero-inflated lognormal distance law. Let q be its point
mass at zero and F its full CDF. Zero observations use -log q; distances in
(0,64) use negative log density per Al-equivalent Å; distances ≥64 (including
infinity when no reference exists) use -log survival at 64. Denote this L_d.
The early term is a weighted sum of Bernoulli log losses for d≤R, with
R=(8,12,20,32) and fixed weights w=(.05,.15,.40,.40):

`L_early = sum_R w_R * BCE(1[d <= R], F(R))`

The training/selection objective is `L_d + 2 L_early`. The greater weights on
20/32 prioritize spatial warning before reference crystal reaches the local
8 Å patch. This is not temporal prediction. Positive/negative labels are not
reweighted; log CDF/survival use stable log-normal tail calculations. Each
component is a proper probability score; there is no AP/ranking/first-alarm
reward. A more aggressive loss cannot create information absent from the local
input. False alarms, misses and reliability remain essential held-out checks.

The matched CD-MACE128-VC variant adds a training-only VCReg term on the exported
128-vector (no projector). For normalized positive row weights `a`, define
`C = sum_i a_i (z_i-mean)(z_i-mean)^T`, with `mean=sum_i a_i z_i`.
The population covariance uses the same material weights as the supervised
objective, normalized within the actual global batch; padded rows have zero weight.
`V=mean_j relu(1-sqrt(C_jj+1e-4))` and
`C_off=sum_{j!=k} C_jk^2/(128*127)`. Add `0.05 V + 0.05 C_off`, ramped linearly
over the first 3,113 updates (one full epoch). The embedding scale is fixed at 1.
This is an adaptation of [VCReg](https://arxiv.org/abs/2306.13292), using weighted
population covariance rather than its unweighted sample covariance. No augmentations,
invariance term, physical reconstruction, species input or extra encoder forward
is introduced. Autograd-enabled gathering computes moments on all 4,096 rows;
DDP averaging cancels the replicated-loss gather gradient factor. Validation and
checkpoint selection retain the predictive objective alone. W&B logs the
predictive objective, regularizer, total and its two components separately.
Global feature spread need not preserve useful within-liquid information;
held-out probes and representation diagnostics determine whether it helps.

Every training row appears once per epoch. Each material has equal objective
mass, with uniform rows within material, implemented by weights
N/(number_of_materials * material_row_count). This does not claim equal
independent ancestry counts. External branches can share preparation ancestry.
Selection uses equal-source weighting on the frozen 15 Al selection sources;
there is no held-out external-material validation claim. All benchmark train,
selection, calibration and test roles retain the fixed Al64 contract.

The declared fit runs 12 epochs, selecting by the combined validation likelihood
from epoch 12 onward (hence epoch 12 in a completed 12-epoch run). A timed-out
partial checkpoint is explicitly incomplete. Global batch 4096 is distributed
over two GPUs, 2048 per GPU; the final batch pads with zero-weight rows, and
distributed gradient averaging accounts for the actual global row count.
Likelihoods use float32; the trainable encoder/head use bfloat16 autocast with
float32 master weights. Coordinate storage is float32. Runtime throughput and
memory logging describes scientific training, not a separate online benchmark.

`validation.csv` reports the combined objective, distance NLL, early log loss,
RMSE of the capped mean distance and Brier scores at 20/32 in normalized units.
The logged runtime throughput includes training synchronization and checks;
warm compilation can slow the first interval. Epoch is examples visited divided
by the declared full training population, not an arbitrary update-budget label.

After training, the selected joint head is evaluated on all original fixed Al
observations and the unchanged controlled scans. Frozen feature export for this
comparison uses float32, as in the previous context baseline. The early-weighted
loss trained the encoder; the subsequent vector/harmonic context probes retain
their previously declared distance-NLL objective and fixed training population.
`distance.csv` scores source-weighted continuous NLL, MAE of the capped median,
RMSE of the capped mean, and CDF Brier scores. Infinity is capped only for point
errors, never treated as an exact distance in the likelihood. Density NLL cannot
be compared numerically to historical categorical NLL. Confidence/path metrics
use the separate frozen `spatial_confidence` contract and report every threshold.


Tracking revision (2026-09-26): diagnostic frozen readouts and per-checkpoint
evaluations keep their logs and results locally. Associated final scores update
a recorded scientific training run through the API, without creating or
restarting runs. Scientific training remains online. This changes logging and
validates identity/hash before cached readout reuse; objectives, selectors,
metric calculations and historical exported definitions are unchanged.

Implementation revision (2026-09-27): the shared trainer also supports full-history checkpoint initialization and explicit material subsets for the separate material-adaptation protocol. Historical calculations and frozen exports are unchanged.
