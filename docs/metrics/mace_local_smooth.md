# Direct temporal regularization of frozen local-state features

Protocol `mace_local_smooth_v1`, stage A of the smooth-manifold research proposal.
This compares snapshot nonlinear maps of the completed velocity checkpoint's
256-dimensional structural block. MACE is frozen; velocity information is not
an additional input to these maps. There is no teacher-feature retention,
forecast target, history, curvature loss or tangent-rank constraint in this stage.

## Inputs and independence

Reuse the exact 27,000 cloud observations / 13,500 same-center pairs from
`mace_local_phase_space_v1`. All 1,125 retained records contribute. Source train,
validation and test assignments remain unchanged; related legacy shooting
descendants stay in train. Prepared features reproduce the saved original test
embeddings and normalized targets before fitting. Feature/reference rows and
targets are never shuffled independently. The old test sources have informed this
research and are labelled **development_test**, not fresh confirmatory evidence.

Physical targets and training scales are the exact velocity protocol's first 160
columns: 16 group observables and 16/64/64 instantaneous H0/H1/H2 descriptors.
There is no relaxed-TDA target in this dataset. Raw qbar6 (column 4) below the
configured threshold 0.30 selects individual low-order observations. A low-order
temporal pair requires both endpoints to pass. This is a proxy, not a PTM phase.

## Objective and fitting

Maps are 256 -> 128 -> 64 -> d, with SiLU hidden activations; d is 8, 16 or 32.
All variants use a d -> 128 -> 160 physical readout. The uncompressed reference
uses the actual 256 features and initializes its readout from the completed
checkpoint's structural head. It is refined under the same physical fitting
budget, and epoch zero is an eligible checkpoint. Compact readouts/maps are newly
initialized; same seed and dimension give identical starting weights across
temporal penalties. The initial map output is divided by its training-only
within-context channel standard deviations, kept fixed subsequently.

Optimization uses complete training batches, AdamW, configured learning rate and
weight decay 1e-4, gradient clipping 5, and configured epochs. The temporal weight
is zero for 50 warmup epochs, ramps linearly over 50 epochs, then stays fixed.
Exact values are in the recipe. Checkpoint/resume identity includes config, data
and implementation hashes; every saved last checkpoint is a completed epoch.

For each observation, subtract its source-record/physical-frame context's weighted
feature mean *inside the loss only*. Pool weighted residual outer products into a
population within-context covariance C. Training weights are the previous
protocol's inverse balance-group sample counts, normalized to epoch mean one.
Compute C_all and C_low, where the latter separately centers only low-order rows.
These context means are not inputs to the deployed map. Four tracked centers per
record/frame are sampled, so these are finite-sample local variance estimates.

For each eligible physical lag <=0.8 ps and each of all/low-order pairs, compute
weighted mean squared Euclidean increment divided by `2 trace(C_subset)`.
Normalize pair weights within each nonempty lag bin, then average bins equally
and average the all/low-order results equally. The precise nonempty bins and
counts are exported. Different observed lags are not treated as equal time steps.
A missing entire temporal subset or collapsed covariance fails explicitly.

The physical loss averages the four target-family normalized MSEs equally;
each family averages its columns, then source-weighted observations. The map loss
is `physical + lambda * temporal + covariance_weight * calibration`, where
calibration is the average of `||C_all-I||_F^2/d` and `||C_low-I||_F^2/d`.
The reference has no trainable map and omits calibration. Covariance control
discourages collapse and redundancy within conditions. It is not an empirically
established low-dimensional temporal manifold constraint.

Each variant selects the smallest **validation physical mean + its full lambda
times validation temporal score**, including epoch zero. Validation has 0.75 ps
pairs; means use equal sample weights because sources contribute equally. The
calibration penalty is a training regularizer and is excluded from checkpoint
selection. Test observations never select checkpoints. Afterward, validation
selects the smallest within-low-order jump among compact candidates whose four
physical family errors, both globally and within low-order pairs, are each at
most 1.10 times their matched-seed reference.
This compares a declared finite sweep; no candidate passing is a valid outcome.

## Exported comparison

`bond_order` and `instantaneous_TDA_H0/H1/H2` are current-target normalized MSEs:
mean over coordinates and both time endpoints within each source, then equal
mean over the 30 sources. They measure the jointly trained physical readout.
They are not independent probes or proof of retention of unsupervised observables.
Reference head warm start versus new compact heads is explicitly part of this
comparison; the existing reference accuracy is a preservation target.

`rms_jump` = sqrt(mean squared same-center Euclidean increment divided by
`2 sum(var(z_reference))`), with population variance (ddof=0) across the exact
512 training anchor identities retained by the velocity audit. It uses the map's
actual unrescaled outputs. Uniform scaling cancels. `p95_jump` and `max_jump`
describe individual normalized lengths, not confidence intervals. Test lag is
verified as 0.75 ps. Reference spread and absolute increments remain in JSON.

`low_order_rms_jump` and `low_order_p95_jump` select low-order pairs and recompute
reference variance using only the low-order members of those training anchors.
Both selections use the fixed raw physical target threshold. Counts are exported.
Loss normalization uses full-training within-context variance; reported jump
normalization uses fixed training reference anchors, matching the previous audit.
These are explicitly different normalizations.

`low_order_bond_order` and `low_order_instantaneous_TDA_H0/H1/H2` score the same
physical targets restricted to pairs whose two endpoints pass the low-order
selection, averaging each eligible source equally. Counts and per-source errors
remain in JSON. Sources with no eligible pairs do not enter this conditional mean.

`worst_physical_error_ratio` is the maximum candidate/reference error ratio over
the four global and four low-order families for the same seed and split.
`information_pass` checks <=1.10.
`jump_pass` and `low_order_jump_pass` check RMS <=0.10 in their respective
populations. These are proposed experimental gates, not universal physical limits.
The figure shows this physical-error ratio against within-low-order RMS movement.

All seeds and candidates remain visible, regardless of selection. Sampling
uncertainty and independent readout/observable validation are follow-up work;
this stage does not claim a confidence interval from its two training seeds.
The 27,000 embedding/prediction rows and selected epochs are retained per variant.
Future curvature/history stages require consecutive trajectories and remain
separate scientific protocols. Historical metric exports are not rewritten.

The shared dispatcher also exposes the separate `motion-*` stages; this does not change this historical protocol's calculations.
