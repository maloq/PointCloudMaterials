# Orientation-preserving spatial prediction

**All ten fits completed on 2026-09-25.** Compare four context predictors on the
same current observations, using a shared patch encoder and learned combination
of patch-level predictions. This is the crystallization-supervised branch.

The question is whether relative orientation and surrounding structure improve
predictive information beyond symmetric scalar geometry attention. Train and
select by source-weighted event negative log-likelihood (NLL). AP at 3 ps is the
main ranking diagnostic, AP at 6 ps secondary; neither selects checkpoints.

| Predictor | Inputs and interactions |
| --- | --- |
| Symmetric invariant | Scalar embeddings, radial node geometry and pair-distance attention |
| Vector messages | Scalar embeddings plus MACE vector fields, relative directions and radial message filters |
| Tensor attention | Scalars, MACE l=1/l=2 fields, invariant attention, spherical harmonics and fixed Clebsch–Gordan couplings |
| Harmonic hierarchy | Scalars and l=1/2/4/6 fields, orientation-preserving regional aggregation at 10/20/48 Å around every patch |

The vector model borrows scalar/vector mixing from [PaiNN](https://proceedings.mlr.press/v139/schutt21a.html).
The tensor model borrows irrep attention from [Equiformer](https://arxiv.org/abs/2206.11990).
The hierarchy combines nested regional summaries with the principle of averaging
bond harmonics before contraction from [Lechner–Dellago](https://arxiv.org/abs/0806.3345).
These are compact adaptations, not reproductions of the published architectures.
Smooth attention support enters the normalization as well as the values; the
importance of this distinction is discussed in [EquiformerV3](https://arxiv.org/html/2604.09130v1).

## Matched comparison

Use all 31,609 existing prospective observations from 150 independent Al sources.
Preserve 90 train, 15 selection, 15 calibration and 30 historical test sources.
Do not generate new trajectories or quenches. Compare observed and relaxed
coordinates separately: two shared encoder fits followed by four predictors per
domain, ten scientific training runs total, one seed (20260924).

Each domain gets one fresh width-128, 128-export native MACE encoder (634,496
parameters), fitted for 4,096 updates by the existing onset-likelihood trainer.
Select its minimum selection NLL checkpoint, freeze it, and apply exactly the
same weights to all patches and all four predictors in that domain. Context
predictors use width 128, 16 field channels, two blocks and 4,096 updates each.
Batch and microbatch are 512 for encoder training; predictor batch and microbatch are 512.
The encoder is not jointly fine-tuned with the context predictor in this study.

At every focal atom, select 25 representative atoms: the focal atom and 12
cuboctahedral queries at each of 10 and 20 Å. Unique minimum-cost atom assignment
allows at most 4 Å query displacement. Select the observed nearest 80 IDs for
each representative, preserve those IDs under relaxation, then crop each view
at 8 Å. Spatial MACE uses two message-passing blocks, 5 Å edges and no halo.
The maximum observation reach is approximately 32 Å from the focal atom.
The 48 Å hierarchy radius is between patch centers; it does not add atoms.

All predictor nodes use one shared event head. Each produces a six-category
distribution: onset in (0,0.75], (0.75,3], (3,6], (6,9], (9,12] ps or survival
beyond 12 ps, **for the focal event**, not the neighboring atom's event.
A shared learned combiner takes each patch's probabilities and distance to the
focal location, then mixes those distributions. The mixture is trained jointly
by NLL, starts with uniform weights, and preserves normalized probabilities and
monotonic cumulative risk. No central embedding bypass, index embedding or
manually privileged central vote is present. The focal location is still known
through relative geometry. The local patch encoder's center indicator has the
same meaning in every patch.

Inputs use one current frame, positions and Al species. No velocities, temperature,
simulation age, explicit time covariates, descriptors, PTM features or future
frames enter the predictor. Quenches start from that same current full periodic
cell. Original MD sustained onset supplies supervision; relaxed crystallinity
does not define the label. Time is used only to construct labels and provenance.

## Interpretation limits and evaluation

The matched invariant baseline adapts the old symmetric geometry attention.
It removes the old central residual, shell priors, history and metadata inputs.
Historical symmetric-context scores are therefore not matched baseline results.
Equivariant variants also receive additional information: MACE l=1/2 features
before scalar export, and the hierarchy additionally receives fixed present
coordinate l=4/6 bond fields. This tests complete predictor/input designs,
not a pure equal-information or equal-parameter architecture ablation.

Only training sources fit feature normalization. Scalars use channel mean/std;
tensor fields use channel RMS over all components, without component centering
or anisotropic scaling. NLL training uses exact importance correction for event
oversampling. Report test NLL, AP3/AP6, Brier/log loss, recall at a calibration-set
false-positive threshold and source-bootstrap intervals. Calibration is fitted
only on calibration sources. Paired differences against the symmetric control
resample whole test sources. One seed cannot quantify training-seed uncertainty.
Test sources have been used in previous research; this is not a new untouched test.

Rotation/reflection tests cover jointly transformed coordinates, nominal queries
and tensor fields with fixed atom correspondence. The box-fixed stencil and
discrete representative/nearest-80 assignment are not globally SO(3)-invariant
resampling operations. A rotated or noisy structure can change selected atoms.
Absolute archive float16 coordinates retain their quantization; extraction does
not recover pre-quantization precision. All four variants use the same new cache.

This comparison cannot measure 0.75 ps trajectory stability: its retained focal
anchors are 12 ps apart. Neither low event NLL nor equivariance establishes
temporal smoothness or coordinate-noise robustness. Dense trajectory/noise
audits remain separate analyses, not fabricated from these sparse anchors.

Recipe: [comparison config](../../configs/equivariant_context/comparison_20260925.json).
Commands and resume procedure: [operations](../../docs/equivariant_context.md).
Results: [completed comparison](/work/PERSO/vmorozov/analysis/equivariant_context/node59-b512-v2-20260925/RESULTS.md).
Relaxed tensor attention improves test event NLL from 0.143385 to 0.139248
against the relaxed symmetric control (paired source interval excludes zero).
Observed harmonic hierarchy improves AP3 from 0.151607 to 0.375376 and AP6
from 0.372756 to 0.479996, with paired intervals excluding zero, but does not
improve event NLL. Relaxed vector messages reach AP3 0.386266; their NLL gain
remains uncertain. These are one-seed results on 30 previously examined test
sources; AP remains an evaluation diagnostic. See the report for all eight
models, source uncertainty, calibrated scores and input/interpretation limits.
