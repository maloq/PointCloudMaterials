# Giving equivariance a structural purpose

Proposal, September 18, 2026. No training or architecture change has been applied.
Based on the [frozen GATr audit](../../output/gatr_equivariant/al-v6-node07-20260918/RESULTS.md).

## Correct the interpretation first

The audit establishes rapid changes of internal vector directions and weak
spatial vector alignment. It does **not** establish that equivariance is useless
for the invariant task, or that a useful crystalline representation should
contain a persistent arrow. Intermediate geometric-stream erasure changes z;
the final returned multivector is discarded. Smoothness and utility are separate
from the already verified rotation law.

A permutation-invariant vector readout of a perfect FCC neighborhood must be
zero: rotations by 180° about x and y leave the neighborhood unchanged, while
the only vector fixed by both rotations is zero. Thermal symmetry breaking can
then give the small vector a rapidly fluctuating direction. Likewise, a cubic
neighborhood's second moment is isotropic, so its principal axes are undefined.
Forcing a nonzero fixed arrow would conflict with the symmetry of the target.
This argument assumes no external direction or atom-index symmetry breaking.

Under proper rotations about the center, a PGA multivector decomposes into four
scalar and four vector triplets. The grade-4 pseudoscalar is **not** an angular
momentum l=4 representation. Increasing the number of center multivector channels
does not by itself produce cubic-orientation tensors. We must preserve angular
information on neighbor tokens and form higher-order features before pooling.

A mathematical sanity check on node07 used the exact 12-bond FCC first shell:
standard Q1 and Q2 are below 7e-17, Q4=0.19094065 and Q6=0.57452426. Generic
rotations change the q4m/q6m coefficients while preserving their norms; cubic
symmetry rotations leave the coefficients unchanged. This is a representation
check, not evidence from a trained modification. Its record is
[`fcc-symmetry-sanity.json`](../../output/gatr_equivariant/al-v6-node07-20260918/technical/fcc-symmetry-sanity.json).

## Recommended first architecture

Keep GATr's contextual per-neighbor features and the invariant z128 interface.
Add an explicit SO(3)-equivariant bond-orientation branch with l=4 and l=6,
initially four channels per order (4×9 + 4×13 = 88 real coefficients).

For invariant token scalars s_j and center-relative unit bond directions u_j:

```
H[l,c,m] = sum_j w(r_j) a[l,c](s_j, s_center, r_j) Y[l,m](u_j)
           / sum_j w(r_j),                      l in {4,6}
```

The learned weights a are scalars. Their output for each channel is shared
across m, preserving H_l(RX)=D_l(R)H_l(X). Exclude the center; use a fixed smooth
radial taper over native input neighbors, with cutoff chosen from training
radial distributions. Do not introduce hard nearest-k membership for the new
smooth target. Record the taper and harmonic basis as a new target protocol.
The existing nearest-12 bond-order protocol remains unchanged.

Supervise a small channel-mixing readout of H_l against the **full q_lm
coefficients**, computed from the same smooth bond weighting. The existing
physical q4/q6 targets are scalar norms and do not specify orientation.
`src/analysis/liquid_structure.py:bond_order` already computes complex q_lm
internally but exports scalar summaries. Use an explicitly versioned real basis
for the new head, target, tensor contractions and rotation tests.

Expose H4/H6 as the orientational embedding. Feed rotational invariants such as
`G[l,c,d] = sum_m H[l,c,m] H[l,d,m]` into the z readout alongside its present
scalar features. This preserves invariant z and gives the new branch an explicit
path to physical/TDA losses. Simply attaching a head to the discarded output
would not establish this connection. Forming l=4/6 only from pooled center
vectors cannot recover information that cubic symmetry has already removed.

This is a small hybrid GATr/tensor readout, not an unchanged native GATr output.
It deliberately supplies angular geometry. A head that merely reproduces a
known descriptor is not proof that the learned backbone adds value; the
deterministic-harmonic baseline below is essential.

## Objective and smoothness

First establish physically accurate covariant outputs, then add a small temporal
term. A proposed objective is:

```
L = L_existing + lambda_bond L_full_q_lm + lambda_time L_covariant_time
L_covariant_time = mean_l ||H_l(t+dt) - D_l(R_t) H_l(t)||² / S_l²
```

R_t is a detached proper fit of persistent-neighbor cage rotation; it is used
only in the training loss and evaluation. Snapshot inference still needs one
frame. Include unaligned laboratory-frame persistence in evaluation as well.
Use weak weights where cage fitting is poor or physical bond tensors change
substantially, so rearrangements and grain boundaries are not forced to remain
constant. Start with short verified time gaps; no future input at inference.

S_l is a fixed training-reference scale for the whole irrep block, never a
per-m coefficient whitening or a denominator the model can shrink. Use finite
tensor MSE without normalizing tiny vectors to unit length. Supervised nonzero
bond targets anchor amplitude; also monitor irrep energy, invariant feature
spread and physical/TDA accuracy to detect unused or collapsed extra channels.
Avoid an arbitrary nonzero-vector requirement at symmetric centers.

Do not apply an unconditional nearest-neighbor alignment loss: it can erase
grain boundaries and orientational defects. First require agreement with local
physical tensors. Spatial comparisons should distinguish same-grain interiors,
boundaries and unclassified environments, with labels only for evaluation.

The current mixed-triplet implementation already regularizes curvature of z128.
That does not give an exported covariant embedding a physical target. Preserve
that distinction and its separate training protocol.

## Controlled pilot and success criteria

Use the audited Al checkpoint and its original training/selection ancestry for
a separate matched continuation. DATASETS.md was consulted; no new simulation
is needed for these coordinate-derived targets. Verify source availability and
prepared triple schemas before execution. Use selection sources for tuning;
the trajectories already inspected remain exploratory test data.

| Arm | Change |
|---|---|
| A | Matched existing-objective continuation |
| B | Learned l=4/6 branch, full coefficient supervision, invariant connection to z |
| C | B plus covariant temporal regularization |
| D | Deterministic smooth q4m/q6m branch connected to z with matched readout capacity |

Same input schedule, physical/TDA labels, continuation budget and data splits.
Calibrate new-loss gradient magnitudes on training batches before a small
selection-only strength sweep. Freeze sampling, target definitions and selection
criteria before comparing test results. A new architecture/target revision must
not mutate the live snapshot encoder or the frozen audit's producer files.

Success requires more than smoother curves:

- Correct tensor rotation law and scalar readout invariance.
- Full q_lm accuracy, orientation retention modulo crystal symmetry, temporal
  persistence and same-grain coherence at physically meaningful amplitudes.
- Sensitivity to fixed-radius angular distortions; preservation of boundaries
  and genuine transitions; no collapse or excess transition lag.
- Physical/TDA utility relative to matched A and deterministic D. Geometric
  branch erasure must have a measurable task effect; a matched ablation/retrain
  is stronger evidence than out-of-distribution erasure alone.
- Comparison to direct q4m/q6m as the orientational descriptor. If learned H
  adds no utility beyond these inexpensive features, retain them explicitly
  and use GATr for complementary structural information.

## Basis and scope

[GATr](https://arxiv.org/abs/2305.18415) supplies the geometric transformer;
[tensor field networks](https://arxiv.org/abs/1802.08219) give the spherical-harmonic
construction of higher-order equivariant features. Full bond harmonics and their
rotational invariants follow the
[bond-orientational-order framework](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.28.784).
The particular hybrid head, losses and pilot are proposals for this repository;
their benefit has not been measured.

## Measured follow-up: information in the current export

The subsequent [conditional-information test](../gatr_conditional_information_20260918/README.md)
holds radial structure and density approximately fixed. Frozen z128 adds little
bond/angular information in 609 strictly matched spatial pairs, while SOAP
provides a strong positive control. Apparent small forecasting gains largely
recur with redundant radial inputs; no robust angular increment remains after
current-order conditioning. Forecast probes also fail to beat a prevalence-only
Brier baseline, limiting conclusions about future information. This motivates
the proposed angular supervision and connection to z, but does not validate
that proposed architecture or its temporal objective.
