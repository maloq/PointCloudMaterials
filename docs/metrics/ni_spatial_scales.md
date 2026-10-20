# Spatial scale of Ni latent prediction

This family evaluates an explicit spatial-input extension of the native-time
[Ni kinetic protocol](ni_kinetic_latent.md). It retains that protocol's local
244 measurement targets, train/test ancestry, normalization, prediction horizons,
energy scores, ordering diagnostics, frozen phase readout, nucleus definitions,
probe budgets and whole-source uncertainty. The differences are specified here;
both descriptions and shared implementations are bound in the metric contract.
Historical runs retain their own frozen source and exported definitions.

## Matched inputs and unchanged population

Four conditions, three seeds 20261025/26/27 each: `local`, `r8`, `r12`, `r20`.
Every encoder receives 331 numbers: the unchanged normalized central 244
measurements and 87 regional measurements. Local supplies exactly 87 zeros.
The other conditions use all observed atoms within 8, 12 or 20 Angstrom of the
same central atom, with minimum-image periodic distances and no atom-count cap.
The central atom itself is excluded from the regional summaries. Full-ball
membership is reselected at each actual observation; the central identity is
fixed through time. The recorded box determines periodic images but is never a
feature. No explicit temperature, absolute time, simulation age, species, phase,
PTM classification or nucleus labels enter the encoder or transition network.

All 14,400 native windows remain: 3,840 train/16 ancestors and 10,560 test/44
ancestors. Each has four observations through time zero and 60 future
observations at exact 0.1 ps spacing. The original/liquid-liquid/growing-front
enrichment and its interpretation limits remain unchanged. The separate birth
assay retains all 19,800 rows, 19,652 unique inputs, original roles and exact
lead times, with no new neighborhood filtering. Regional histories use the
same four physical offsets. Their membership differs from the original birth
local-history control, which holds the present 80 atom IDs fixed backward.
Birth encoders use only the present observation, so no mismatched historical
membership is supplied to the native transition model.

Shared extraction covers 87 relevant source histories, 52,578 source/frame
groups and 999,971 unique (source,frame,center) observations. The requested
1,000,208 observations include exact duplicates shared between windows/cohorts.
These are not independent MD continuations. All source cells have 32,000 atoms;
only observed regional features are retained. No new MD, relaxation, TDA
calculation or physical derivative teacher is introduced.

## Regional measurement definition

Regional density uses the existing `SmoothDensity` with four radial channels,
angular orders l=0..6, radial centers [0.15,0.4166667,0.6833333,0.95], radial
width 0.12 and density divisor 12. Positions are divided by the selected radius
R. A quintic taper is one through 0.8R and vanishes at R. Four radial masses and
ten upper-triangular radial-pair powers for each of seven angular orders give
74 numbers. Harmonic moments are summed before their Gram products, giving
linear atom-count cost. Angular power divides each Gram block by 2l+1.
The descriptor does not normalize away the number of atoms in the region.

Motion contributes 13 numbers from the shared `VelocityMeasurements.kinematic`:
three shell-wise radial means, radial second moments and tangential second
moments, plus affine dilation, squared shear, squared rotation and residual
second moment. Shell centers are [0.35,0.60,0.85]R, widths 0.1R, with the same
compact taper. Affine fitting has fixed ridge 1e-4 Angstrom squared. All
velocities are relative to the center in Angstrom/ps; uniform boosts cancel.
The four quadratic bond-power rates from the original local17 are deliberately
absent here. The local17 remain in the unchanged central244 vector.

Source minimum images and subtraction use float64; GPU regional arithmetic is
float32. Invalid padded entries are explicitly moved outside compact support
with zero relative velocity, contributing no density or shell weight. Complete
support and half-box validity are checked, with measured atom counts recorded.

For each radius/channel separately, native training observations alone fit raw
mean/std (floor 1e-4), then asinh and transformed mean/std (floor 1e-4).
No test or additional birth-only training sources fit these transformations.
All 64 observations of the original 3,840 native train windows participate.
There is no clipping. Local-control context is defined as zero without regional
statistics and can train before regional extraction finishes. Its checkpoint
records this explicit definition; context checkpoints retain the fitted
regional normalizers and selected radius. Central normalization remains bitwise
the existing sealed 244-target normalization.

## Fixed training and autonomous generation

Every condition uses the same hybrid objective and architecture: a
331->256->256->32 encoder, regularized nonreversible VAMP-2 with alternating
0.5/1 ps pairs, and group-balanced central244 reconstruction with weight one.
Context features are input information, not an enlarged reconstruction target.
Consequently all reconstruction losses and decoded proper trajectory scores
refer to identical local targets at every spatial scale. The local zero-input
condition retains the same parameter layout; its regional input weights receive
no data-driven gradient.

The three stages each last 12 epochs: representation (2,304 updates), fresh
frozen-encoder decoder (2,304), and eight-component latent-increment MDN (2,700).
All conditions have 7,308 updates, batch/microbatch 1,024, AdamW maximum LR 0.001,
one-epoch warmup, cosine to 1% LR, weight decay 1e-5 and gradient clipping at 10.
Neural operations are float32; VAMP covariance/Cholesky are float64. The fixed
final checkpoint is evaluated; no test selection, early stopping or AP objective.
All 12 scientific runs use online W&B IDs with prefix `niscales-`. Readouts and
preflight remain local; final metrics update the original scientific run.

The MDN receives the four most recent latent states. Its training targets are
encodings of actual observations, including their contemporaneous regional
context. Autonomous forecasts encode only the four initial observed inputs and
then evolve latent states recursively. No future saved regional feature, atom
position, velocity, neighbor list or box is supplied. The decoder predicts only
the central244 measurements. This is a context-dependent local trajectory
surrogate, not a jointly evolving many-atom configuration or a connected nucleus
generator. Observed-future context appears only in the explicitly retrospective
`encoded_MD` control.

## Evaluation and contrasts

Native retention uses all 10,560 current test observations. The forecast panel
and random seeds are unchanged from the kinetic comparison: 464 fixed rows,
44 ancestors, 32 sampled paths, 1/3/6 ps horizons. Density/velocity/BO/TDA group
MSEs, continuous raw order, phase Brier, path/increment/lag-product energy,
coverage, temporal/ensemble power, operational high-order liquid occupancy and
local persistent crystal entry retain the precise shared definitions. The
train-only phase probe now sees the 32 latent features produced from 331 inputs;
its budget and class threshold are unchanged. Exact existing PTM annotations
are reused with checksum binding. Raw inverse overflow and nonfinite autonomous
forecasts retain explicit failure accounting, not survivor-only estimates.

Nucleus readouts retain latent32, decoded244, actual central244+latent32, and
actual central-history976+latent32. Per-arm controls additionally fit the actual
current331 measurements and actual four-frame context-history1324. Common
controls fit prior, structural227, central244 and central-history976 once. All
use the same fixed 600-tree/depth4/LR0.04 predictive CatBoost Logloss protocol,
seed20261025, existing case-control weights, 32/64-atom targets and 1/3/6 ps leads.
The direct-context controls repeat the same fixed diagnostic seed across model
seeds; these repeats are not independent stochastic probe fits. No calibration
map or AP-driven selection is fitted.

Primary matched contrasts are each of r8/r12/r20 minus local, for every declared
native scope and nucleus feature space. Additional frozen contrasts compare
direct context331/history1324 against central controls and compare central-plus-
latent readouts against direct-context readouts. The latter tests compression
and accessible information under the fixed probe; it does not give the latent
readout extra raw context. Comparisons group by `arm.scale`, not the shared
`arm.kind=hybrid`. Source-paired differences, 1,000-draw whole-source intervals,
seed spread, incomplete-seed counts and nonfinite handling use the shared
comparison implementation. Repeated windows and neural samples never count as
independent source trajectories.

A larger region can contain existing crystals or broader cell-ordering progress.
Improved prediction alone therefore does not establish a prestructured-liquid
mechanism or a causal influence radius. The native present-liquid stratum only
constrains the existing local80 support, not the full 20-Angstrom region. No
new filtering is introduced to conceal this distinction. This campaign measures
the value of spatial scale; coupled spatial generation is a separate experiment.
