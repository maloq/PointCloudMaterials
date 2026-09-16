# Consecutive local-state motion: stage B

**Historical protocol, discarded 16 September 2026.** The implementation is no
longer active. Definitions below and original run exports remain as evidence;
see [retained source and results](../discarded_frozen_encoder_maps.md).

Protocol `mace_local_motion_v1`. Frozen current coordinate/velocity MACE features
feed a snapshot state map using only the coordinate-derived first 256 channels.
No history input, future prediction, relaxed-TDA target or MACE fine-tuning is
implemented in this stage. This protocol is separate from the paired stage-A
`mace_local_smooth_v1`; historical calculations/exports are preserved.

## Consecutive data and provenance

Use the original velocity experiment's inventory, preparation-level splits,
source-balancing groups, tracked center IDs and target normalization. Select one
window of nine consecutive recorded eligible frame indices per source, centered
nearest the median eligible frame. Use the four previously sampled center IDs.
No interpolation, random reshuffling of atoms or newly inferred velocities.
The converted legacy stream contains scattered pairs, so it is explicitly
excluded from this sequence protocol. All 1,114 binary records and ten native NPZ
records otherwise contribute (1,124 records / 4,496 sequences / 40,464 observations).
All measured velocity data are Al. Smoke recipes deliberately use a named subset.

Binary arrays/timelines/atom IDs are validated by `ShootingBinaryTrajectory`;
legacy NPZ uses its producer's `positions_A`, `velocities_A_per_ps`, diagonal
`cell_vectors_A`, and `step` fields. Source timestep is 3 fs. Preserve native
storage precision. Each source unit retains its original manifest or NPZ hash,
hashes of selected arrays, extracted features and labels, and exact frame/center
selection. Labels are the velocity protocol's existing 169 columns; training uses
the first 160 (16 group observables, 16/64/64 instantaneous H0/H1/H2 descriptors).
The existing train-only target normalization is reused without refitting.

The held-out sources have 0.75 ps cadence. Finer legacy cadences remain training
data, with their actual time increments. Previously inspected test sources are
**development_test**, not a new blind confirmation set. Related legacy descendants
remain together in training. No source ID or absolute time enters the encoder.

## Models and controlled comparisons

Map: 256 -> 128 -> 128 -> d, SiLU activations, d=32/64 in the main recipe.
Readout: d -> 128 -> 160. Initial state channels are divided by their initial
training within-source/frame standard deviation, fixed afterward. Reference uses
the unchanged 256 structural features and initializes its physical head from the
completed velocity checkpoint. Reference and compact heads train for the same
configured epoch budget. Each seed/dimension/rank has physics-only, direct-slow,
direction-only addition, curvature-only addition and combined controls. "Only"
here refers to additions to direct slowness, not removal of physics supervision.

A shared direction network d -> 32 -> d*r takes only the current state, with
its input detached for gradient purposes. Reduced QR produces an orthonormal
basis B(z) with r=4/8. It cannot see atom/source identity or future observations.
Every control trains this network, including references. The map's direction
penalty uses B detached, while basis fitting uses detached increments and state:
the basis-fitting loss therefore cannot smooth the control state. Map/readout and
direction-network gradient clipping are separate, preventing indirect coupling.

## Losses and sampling

Full training batches; AdamW, weight decay 1e-4, gradient norm limits 5. Source
weights are inverse numbers of records in each original balance group. Equal
samples per source make subsequent within-source weights equal. Physical loss is
the weighted mean of four family MSEs. These are train-normalized MSEs, not errors
in Angstroms or classification percentages.

Subtract each source/frame's weighted mean inside covariance calculations only.
Compute within-context covariance C_all and a separately centered C_low for rows
with true group qbar6 <0.30. Both traces must be nonzero. Covariance calibration
is the mean of `||C-I||_F^2/d` over these populations; reference omits calibration.
This is not whitening of the exported state. qbar6 selects a disordered proxy,
not a PTM phase classification.

For each eligible native lag <=0.8 ps, normalize source weights in that lag bin,
then average nonempty bins equally. Repeat separately for all and low-order
edges, where both endpoints are low. Direct slowness is weighted squared
increment / `2 trace(C)`, averaged over both populations.

For times t0<t1<t2, let h0=t1-t0, h1=t2-t1, v0=(z1-z0)/h0 and v1=(z2-z1)/h1.
Physical acceleration is `2(v1-v0)/(h0+h1)` (state units/ps²).
The training bend is `(v1-v0)*2*h0*h1/(h0+h1)`, in state units. For equal spacing
this equals z2-2z1+z0. It vanishes for constant velocity even with uneven times.
Curvature loss averages squared bend / `2 trace(C)` with source/cadence-balanced
weights, requiring both intervals <=0.8 ps and all three endpoints for low-order
selection. Cadence is the mean of the two intervals; raw acceleration is reported
separately. This finite-lag penalty is not an assumption of deterministic or
differentiable microscopic dynamics.

Use a deterministic uniform sample of at most 1,024 eligible training edges for
the direction loss. Re-normalize all/low source-and-lag weights in that sample.
Direction error is `sum(w ||Delta z-BB' Delta z||²)/sum(w ||Delta z||²)`, averaged
over all/low populations. The denominator participates in the map gradient;
uniform shrinking does not improve this ratio. The separate basis-fitting loss
uses the same ratio with detached increments. Equal lag weighting here describes
the weights: the resulting ratio remains an energy-weighted fraction.

Weak basis consistency uses nearest physical neighbors from a training-only pool
containing up to 16 observations per independent preparation lineage. Neighbors
must have a different lineage. Distance uses the 16 train-standardized physical
descriptors. Penalize `1-||B' B_neighbor||_F²/r`; signs and rotations of the same
subspace do not affect it. This term updates only the direction predictor.
Physical targets are used for this training regularizer, never as encoder inputs.

Main loss: physical + ramp*(lambda_time*slowness + lambda_direction*direction_error
+ lambda_curvature*bend) + covariance_weight*calibration + basis_fit
+ basis_neighbor_weight*basis_consistency. The main recipe uses 100 physical
warmup epochs then a 100-epoch ramp; basis fitting runs throughout. Exact weights
and epoch budgets live in configs. Checkpoint selection minimizes validation
physical mean plus full-weight time/direction/curvature penalties, excluding
calibration and auxiliary basis consistency. No test score selects a checkpoint.
Completed saved epochs resume with scientific config, implementation and input
hash checks. Runtime node/deadline/device settings are excluded from identity.

## Evaluation and acceptance

**Original-pair assay:** map all original frozen paired features and keep their
exact 512 training reference identities. RMS, p95 and maximum jump reuse
`smooth.jump_metrics`: denominator is twice the sum of population channel
variances over training anchors. Low-order pairs and reference anchors are
restricted separately. This assay remains directly comparable to the earlier
0.75 ps audit; sequence sampling is not substituted for that cohort.

**Physical retention:** MSE within source, then equal mean over sources, for each
family, globally and in low-order observations. Sequence scoring uses individual
low-order rows; original-pair scoring requires both endpoints to be low. The
worst error ratio is the maximum candidate/matched-seed-and-rank-reference ratio
across all 16 combinations (two cohorts x two populations x four families).
Information passes at <=1.10. Jump passes only when both original-pair overall
and within-low-order RMS are <=0.10. A failure to find an accepted model is valid.
Among information-passing nonreference candidates, the smallest validation
original-pair low-order jump selects the candidate; passing the motion criteria
must still be assessed separately, not inferred from that selection.

**Sequence motion:** report each native lag and population separately. Normalize
using each map's same original training reference spread. RMS is sqrt of the
equal-source mean normalized squared increments; p95/max use individual pairs.
Bootstrap whole sources 500 times for the RMS interval. Bend RMS is calculated
analogously from equal-cadence triples. Acceleration RMS uses raw state units/ps².
Velocity-change ratio is source-mean squared change between adjacent velocities
divided by source-mean average squared velocities. Learned-direction explained
energy is 1 minus source-mean residual energy / source-mean increment energy on
held-out sources. Projector change is mean `1-||B_t' B_next||_F²/r`.

**Independent local-direction assay:** fit uncentered local displacement bases
using only training increments near each held-out current physical state. Use
up to 16 candidate increments per training preparation lineage, then the nearest
32 distinct lineages (at least 16 required). Never fit the basis to the held-out
increment. Report fractions of held-out increment energy captured by 4/8
directions on a deterministic sample of 128 held-out edges, and rank-4 subspace
overlap between alternating, disjoint training-lineage subsets. The proposed
90% target is a diagnostic, not a demonstrated intrinsic dimension. Physical
neighborhoods are an evaluation tool, not deployed encoder inputs.

**Additional observables:** nearest distance, mean/std radius of the 79 neighbors,
and positional-covariance anisotropy `(largest-smallest eigenvalue)/trace` are not
training targets. Independently fit ridge readouts with weighted training-only
input/output scaling, choose alpha on validation, report source-mean normalized
MSE on validation and development test. This complements jointly trained heads.

**Membership/displacement:** nearest-79 retention is intersection size /79.
Atom-matched RMS displacement uses previous-frame neighbor IDs, center-relative
periodic coordinates at both observations, excludes the center, and handles
minimum-image displacement. Compare jumps with unchanged versus changed
membership, retaining sample counts. No inference of causality from association.

**Storage precision:** on the first configured native-float32 preparation sources,
cast absolute positions and velocities through float16 before rebuilding groups.
Retain coordinate/velocity RMS quantization errors, structural-feature squared
increments, and per-target squared changes. Native float16 data has no recoverable
high-precision reference; this audit does not manufacture one or establish its
complete precision floor.

These runs do not yet establish transition detection delay, spatial coherence of
clusters, or symmetry of a later history/fine-tuned encoder. Earlier backbone
symmetry verification remains distinct. Those assessments and causal history
comparisons follow only when the snapshot tradeoff warrants them.
