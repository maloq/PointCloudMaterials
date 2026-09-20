# Equivariant neighborhood JEPA metrics (v1)

Scientific protocol: `neighborhood_mace_v1`, independently encoded local snapshots,
128 invariant channels and four channels each of l=1,2,4,6 (120 tensor components).
All tensor contractions use orthonormal real component-normalized harmonics in FP32.

## Objective

The shared encoder receives gradients from both prediction inputs and targets.
No teacher, stop-gradient, latent averaging, or future query positions are used.
Material/potential groups define differentiable training-batch invariant means
and scales; inference moments are recalibrated from 64 fixed training anchors per
group and their current/next center snapshots. Variance floor is 1e-6. These head
moments are not part of the exported snapshot encoder.

Physical/TDA decoders additionally receive a learned material/potential group
embedding: head normalization removes domain offsets, but the raw-unit target
means remain domain dependent. SIGReg receives no such group embedding.

Physical85 and instantaneous TDA144 targets and normalization originate from the
training-only center current/next endpoints. Errors average equally over four
physical blocks (32 radial,32 pair,16 angular,5 moments) and three topology blocks
(H0 16,H1 64,H2 64), then over rows. The objective contributions are physical,
0.25*TDA,0.1*bond, prediction_weight*invariant_prediction,
prediction_weight*equivariant_prediction, sigreg_weight*regularizer and
future_weight*(future_physical+0.25*future_TDA) for temporal prediction only.

Invariant prediction MSE uses normalized z. Equivariant prediction averages MSE
within each l block, divides each by the fixed scale 0.1 squared, then averages
blocks. It never independently whitens m components. Neighbor/time queries have
equal mass; future-center prediction is one additional query. Physical future
loss reads the predicted center z, not an encoded future input. Bond loss is the
existing fixed 12-neighbor q4m/q6m component MSE times 12, averaged over orders.
Bond, physical and TDA anchors supervise current and next centers in every arm.

SIGReg uses the installed LeJEPA Epps-Pulley implementation (256 slices,t_max=3,
17 quadrature points) on the 64-dimensional invariant projection of current
centers only, computed within each material/potential group and weighted by group
sample count. It is not applied to tensor coordinates. VICReg controls substitute
(25*invariance+25*variance+covariance)/51 on a current/future-center or
current/first-spatial-neighbor projected pair. Logged loss components already
include their coefficients; their sum is `loss/total`. W&B contains no custom
GPU-memory or training-seconds panels; operational durations stay in JSON logs.

## Selection and output tables

Validation uses all 480 fixed native-Al selection anchors from 15 source
trajectories. First average within source, then equally across sources. `physical`
and `tda` are **current-center** reconstruction errors. `selection_score` is
physical+0.25*TDA. Other materials have training data but no held-out validation
in this release. `training_mean_baseline` uses material/potential-specific training endpoint target
means. It therefore tests information beyond domain identity.

For temporal predictors, `future_physical` is the fixed physical85 error decoded
from the predicted next-center invariant state. `physical_persistence` uses the
observed current physical target as the future prediction. These span the actual
recorded per-source lags, not a shared lag or a crystallization benchmark. Their
selection population is Al only. Spatial/control arms have no forecast metric.

Models are selected on the fixed present-information score, never moving latent
loss or test outcomes. One control and two predictive settings are promoted only
if they beat the training-mean baseline. Promotions restart with the longer
cosine schedule. `selected_step` is the actual chosen update; `total_updates` is
the completed budget. Epochs are sampled equivalents (anchors drawn/training
anchors), not guaranteed complete shuffled passes. There is one seed and no
training-seed interval. Metric exports snapshot these definitions and hashes.

Physical reconstruction alone does not establish good frozen downstream dynamics,
and low latent prediction loss alone does not establish information preservation.


Table export: 2026-09-20T10:40:53.386704+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
