# Does the surrounding Ni region improve a local kinetic state?

Frozen comparison, October 15, 2026. This study changes the observed spatial
support while keeping the central prediction target, model capacity, training
budget, source ancestry and evaluations fixed. It follows the
[kinetic-latent study](../ni_kinetic_latent_20261015/README.md). Implementation and
integration checks establish that the comparison can run; scientific training
and evaluation completion are determined by their individual receipts. Results
from the completed frozen comparison are recorded at the end of this document.

The physical question is whether an 80-atom neighborhood misses surrounding
organization that matters for its subsequent evolution. A larger observed
region could carry orientational organization, strain or motion relevant to
nucleation. It can also reveal an already crystalline neighbor or advancing
front. These explanations must be distinguished before interpreting a gain as
evidence for a subtle liquid precursor.

## Matched observations and notation

Let i identify the initial snapshot, t_i its saved time, and j the fixed central
atom. The integer k indexes observations relative to that snapshot, with
Delta = 0.1 ps. Write Y_ij,k = Y_ij(t_i + k Delta) for the unchanged 244 central
measurements: bond order40, TDA113, smooth density74 and velocity17. Native
sequences contain k = -3,...,60 and follow the same atom while reselecting its
nearest-80 neighborhood at each observation. Neither absolute time nor elapsed
time is an encoder or propagator input.

For a radius R, C_ij,k^(R) is an additional 87-vector measured from all observed
atoms inside the periodic ball of radius R around j, excluding the center.
Positions and velocities are relative to that center. The three radii are
8, 12 and 20 Angstrom. Membership is reselected at every observation and has
no nearest-atom cap. This is an additional regional measurement, not a change
to the definition of Y.

| Regional measurement | Channels | Definition |
| --- | ---: | --- |
| Smooth density | 74 | Four radial sums and rotationally invariant radial-channel power products through angular order 6; coordinates scaled by R |
| Shell motion | 9 | Radial mean, radial second moment and tangential second moment in each of three shells |
| Affine motion | 4 | Dilation, squared shear, squared rotation and residual second moment from the shared local affine-flow fit |

The density uses four radial centers from 0.15 to 0.95 in units of R and width
0.12; compact support tapers from 0.8R to R. Motion shells are centered at
0.35R, 0.60R and 0.85R, with width 0.10R. Regional TDA and bond-order derivatives
are not recomputed. The observation is a compact, rotationally invariant summary;
it does not retain the full regional atom coordinates or connectivity.

Let A denote the existing train-fitted central transformation and A_R the
regional transformation fitted separately for each radius. Both standardize,
apply asinh, then standardize again, using only native training observations
and scale floors of 1e-4. Define u_ij,k = A(Y_ij,k). Each encoder receives

\[
a_{ij,k}^{(R)} = [u_{ij,k},\ A_R(C_{ij,k}^{(R)})]\in\mathbb R^{331},
\qquad z_{ij,k}^{(R)}=E_\theta(a_{ij,k}^{(R)})\in\mathbb R^{32}.
\]

The local control replaces the last 87 entries with exact zeros. It has the
same 331-input architecture and parameter count as the radius treatments.
No radius identifier, temperature, simulation age, absolute time, box vector,
species or phase/birth label is supplied as a feature. Boxes are used only to
construct physical periodic neighborhoods.

## Fixed data and split

The comparison retains all 14,400 native windows: 3,840 training windows from
16 ancestors and 10,560 test windows from 44 ancestors. Former selection roles
are pooled into test and retained as provenance. The 4,800 original windows,
4,800 liquid-persistent windows and 4,800 growing-front windows remain unchanged.
The latter two groups were selected using their futures, so their frequencies
are not natural event incidence or equilibrium occupancy.

The separate birth assay retains all 19,800 rows, representing 19,652 unique
current states with four saved observations. Its declared source contract has
28 train and 62 test ancestors; actual assay rows cover 21 train and 42 test
ancestors. The central birth-history producer holds the present neighborhood
membership fixed backward in time, unlike native reselected neighborhoods.
For that reason the learned birth encoder uses only the present observation;
history is used only in explicitly named readout controls. Regional balls are
reselected independently at each saved birth-history observation.

Regional collection reads 52,578 distinct source frames across 87 relevant
existing sources, deduplicating overlapping requests into 999,971 unique
source/frame/center observations. Each observation is measured at all three
radii. These counts describe additional feature extraction, not new physical
continuations, independent samples or new source trajectories. There is no new
MD, interpolation, additional neighborhood filtering or change to birth labels.

All normalizers, representation fits and probe fits use their declared training
ancestors only. Fixed budgets, final checkpoints and probe settings are chosen
before test evaluation. Test data do not select models, radii, hyperparameters,
thresholds or checkpoints. Source identity and exact native timelines are
verified against the frozen parent inventories.

## Representation, decoding and generation

All four treatments use the same hybrid nonreversible VAMP representation.
The training objective is the equally weighted sum of central measurement
reconstruction error and the negative regularized VAMP-2 score, with that score
divided by latent dimension. Lagged pairs alternate between 0.5 and 1.0 ps;
the covariance ridge is 0.001. Reconstruction weights retain the parent
measurement-family weighting. Both ends of a training pair use the treatment's
observed context, but reconstruction always targets central244.

Training has three stages, each lasting 12 full epochs:

1. Fit the encoder and a provisional central decoder with the hybrid objective.
2. Freeze the encoder, initialize a fresh decoder, and fit central reconstruction.
3. Freeze both and fit an eight-component Gaussian mixture density predictor
   of the next latent increment from the last four latent states.

Encoder, decoder and propagator have two hidden layers of width 256 with SiLU
activations. The encoder exports 32 values. The decoder predicts only the
244 normalized central measurements. Latent increments are standardized using
training latent statistics; each Gaussian component has diagonal covariance,
and the mixture weights and parameters depend on the four-frame history.

Three fixed seeds, 20261025/20261026/20261027, produce 12 independent pipelines.
Batch and microbatch are 1,024. AdamW uses maximum learning rate 0.001,
weight decay 1e-5, one epoch of warmup, cosine decay to 1% of maximum learning
rate and gradient-norm clipping at 10. Neural arithmetic is float32; VAMP
covariance and Cholesky calculations use float64. Each full pipeline contains
2,304 representation, 2,304 decoder and 2,700 dynamics updates, 7,308 total.
There is no derivative teacher or newly integrated simulator trajectory in this
comparison; the target dynamics come from the existing native NPT histories.

To generate continuation b, encode the four observed inputs at k=-3,...,0.
The propagator samples a latent increment, adds it to the previous latent state,
and repeats this operation for 60 steps. The decoder maps each future latent
state to central measurements. All context after the initial four observations
must be carried by the autonomous latent state: no saved future context is
injected. Training uses observed histories for one-step likelihood fitting;
evaluation exposes autonomous multi-step error. The 32 sampled paths per input
are neural realizations, not 32 independent physical MD continuations.

## Evaluations and controls

The parent evaluation definitions are retained, with paired radius-minus-local
comparisons for each seed and identical held-out rows. Tables retain source
packets for whole-source bootstrap uncertainty; missing or nonfinite source
values do not silently become an analysis of surviving cases. See the
[metric contract](../../docs/metrics/ni_spatial_scales.md).

Continuous liquid-order retention uses the current observation of every native
test window. It measures qbar6/coherence, wbar6 and TDA reconstruction, including
source-balanced errors and within-source association. The liquid diagnostic
subset has a noncrystalline central PTM label and local crystalline fraction
at most 0.05. A high-order diagnostic bin uses a threshold fixed from training
liquid observations. This subset and bin describe evaluations only; they do not
filter the birth assay or define a new physical phase.

Autonomous evaluation uses the same fixed source/regime panel as the parent
(464 windows from 44 held-out ancestors), with at most four windows per
source/regime and 32 particles through 6 ps. Whole-path energy scores,
increment scores, lagged products, spread, temporal correlation and order dwell
measure connected paths rather than only separate marginal future values.
Controls are the deterministic conditional-mean rollout, persistence and
retrospective encoding/decoding of actual MD. The last control uses actual future
regional observations and is an information/reconstruction diagnostic, not a
forecast.

A frozen train-only logistic probe maps latent states to central crystalline
probability. Generated local entry and phase Brier scores are compared with
actual PTM observations, using the same probe for all four path controls.
The logistic regularization is fixed (C=1), as is its 0.5 classification
threshold; fitting uses at most 60,000 source-balanced native training states.
Three consecutive crystalline future frames define a local entry event. At
0.1-ps cadence, the earliest confirmed entry is at 0.3 ps; those three frames
span 0.2 ps and do not establish continuous unseen persistence. This local
event is separate from full-cell connected-nucleus birth.

Actual nucleus readouts retain the existing 32-atom and default 64-atom
connected, liquid-origin births with 1.5-ps persistence, at exact 1/3/6-ps leads.
The test set has 135 size-32 events from 42 ancestors and 81 size-64 events from
38 ancestors. Each assay has fixed 20% case-control prevalence, so probabilities
and AP are interpreted for that assay, not an unconditional physical birth rate.
Frozen CatBoost probes use 600 trees, depth4, learning rate0.04, L2=10 and
64 borders with log-loss fitting; there is no early stopping, calibration fit
or AP-based selection.

| Readout space | Dimension | Information available |
| --- | ---: | --- |
| Latent | 32 | Encoded present treatment input |
| Reconstructed measurements | 244 | Present latent decoded to central measurements |
| Central measurements + latent | 276 | Actual central244 plus present latent32 |
| Central history + latent | 1,008 | Actual four-frame central976 plus present latent32 |
| Regional measurement context | 331 | Actual present central244 plus regional87 |
| Regional measurement history | 1,324 | Four actual treatment inputs |

Shared central controls are the training prior, structural227,
central measurements244 and four-frame central history976. Every radius is
compared with local in each of the six readout spaces. Additional contrasts
compare actual context with central-only information, central-plus-latent with
central-only information, and central-plus-latent with actual context. These
distinguish newly available regional information from how much of it the
32-dimensional encoder preserves. The zero-context local arm is deliberately
included in these controls as a consistency check.

## Interpretation

A useful radius should improve more than one aspect of the held-out record:
retain liquid organization, predict continuous path changes, or add information
to the actual nucleus assay beyond central measurements. Better classification
alone does not establish a better generative simulator. Better reconstruction
alone does not establish slower or more physically meaningful coordinates.

Even a reproducible radius gain could be explained by existing outer crystals
or growth fronts. A claim about noncrystalline prestructured liquid needs a
separate, declared stratified audit of regional phase composition and matched
source/present-order controls. No post-hoc removal of birth rows is part of
this experiment. Failure of a compressed regional summary also would not prove
that physical surroundings are uninformative.

The generator evolves independent local measurement paths. It does not jointly
generate atom positions, conserve global quantities, enforce agreement between
overlapping neighborhoods or create a connected full-cell nucleus. Local
crystalline entry is a useful minimum baseline for latent generation, with
actual nucleus prediction evaluated separately. Neither establishes a
nucleation rate, free-energy barrier or faithful atomistic simulation.

The [frozen recipe](../../configs/ni_latent_sim/spatial_scales_20261015.json)
defines this comparison; [execution and provenance](../../docs/ni_spatial_scales.md)
describe collection, the detached queue, receipts and reproduction commands.

## Completed comparison, October 15

All 12 fixed-budget pipelines, 87 source feature packets, native evaluations
and birth readouts completed. The queue ran from 14:49 to 15:25 CEST, about
35.7 minutes. An initial pre-worker Slurm failure was replaced; no scientific
stage failed. Completion does not imply physical validity of generated values.

Default 64-atom nucleus birth AP, using a frozen predictive classifier on the
present 32-dimensional latent, averaged across three encoder seeds:

| Observed context | 1 ps lead | 3 ps lead | 6 ps lead |
| --- | ---: | ---: | ---: |
| Local80 | 0.794 | 0.779 | 0.675 |
| Local + 8 Å | 0.797 | 0.772 | 0.673 |
| Local + 12 Å | 0.813 | 0.800 | 0.704 |
| Local + 20 Å | 0.851 | 0.836 | 0.770 |

The 20 Å minus local 6-ps AP difference is +0.0955, with whole-source paired
95% interval [0.0533, 0.1324], holding fitted seeds fixed. NLL improves from
0.3242 to 0.2922 and Brier from 0.0957 to 0.0852. The 32-atom assay also improves:
local versus 20 Å AP is 0.792/0.869, 0.755/0.836 and 0.677/0.730 at 1/3/6 ps.
All AP values concern the fixed 20%-positive case-control assay. These are
actual birth readouts, not births produced by generated trajectories.

Direct observed 20 Å measurements already give 64-atom AP 0.871/0.851/0.772.
Thus broader measured context carries the benefit; this comparison does not
establish an added benefit from motion learning over those direct inputs.
Existing outer crystals, fronts or broader source state remain possible
explanations. No neighborhood filter or label definition was changed.

Liquid retention is mixed. On 5,937 locally liquid test states from 44
ancestors, within-source qbar6 R² is 0.734/0.683/0.758/0.657 for
local/8/12/20 Å. The corresponding coherence R² is 0.768/0.711/0.810/0.705.
The modest 12 Å gains occur in all three seeds, but local retains lower
aggregate bond-order and TDA reconstruction errors. This does not identify
a distinct prestructured-liquid phase or demonstrate that it causes birth.

Autonomous trajectory quality does not show the same radius benefit. Mean
6-ps path energy score (lower is better) is 0.702/0.963/0.750/0.739 for
local/8/12/20 Å. The 20 Å treatment modestly improves the 1-ps score, has no
clear 3-ps change and worsens the 6-ps score by 0.0373 [0.0130, 0.0614].
It reduces false local crystalline entries, but the strict present-liquid
forecast panel contains only one actual entry and cannot establish reliable
generation of liquid-origin nucleation.

All normalized rollouts remain finite, yet 11 of 12 runs contain some raw
inverse-transformation overflow. Mean 6-ps invalid path fractions are 0.286%,
3.45%, 0.777% and 0.067% for local/8/12/20 Å. Finite raw outputs can also be
unphysical: a current-state local decoder produces wbar6 near -26,942 for one
non-liquid observation whose actual value is -0.00796. Physical decoding and
long-rollout stability therefore remain unresolved.

Evidence is in the existing exported [comparison tables](/store/PERSO/vmorozov/experiments/ni_spatial_scales/radius-20261015-v1/analyses/comparison/tables):
`birth_summary.csv`, `birth_paired.csv`, `retention_individual.csv`,
`retention_paired.csv`, `trajectory_individual.csv` and `trajectory_paired.csv`.
This summary introduces no new fits, selectors, metrics or exclusions.
