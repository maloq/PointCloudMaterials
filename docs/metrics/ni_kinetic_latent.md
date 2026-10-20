# Native-time Ni kinetic representations and local ordering

This family binds the frozen 2026-10-15 kinetic experiment, including its training,
native trajectory diagnostics, and separate nucleus readouts. It does not revise
historical derivative-training metrics.

## Population and inputs

The native bank contains 14,400 overlapping atom-centered sequences from 60 Ni
MD histories, each with 64 observations at exact 0.1 ps intervals, from -0.3 to
6 ps relative to its prediction origin. There are 3,840 train windows from 16
ancestors and 10,560 test windows from 44 ancestors. Former selection rows are
test; original roles are retained. The central atom identity is fixed along a
sequence; the nearest 80 neighbors are reselected at each observation. The
original, liquid-liquid, and growing-front groups each contain 4,800 windows.
The latter two were selected using their futures: separate regime results are
required, and their mixture does not estimate natural nucleation incidence.

Encoder inputs are the current 244 measurements: bond order 40, TDA 113,
density 74, velocity 17. No temperature, simulation age, absolute time, box,
species, phase or birth labels enter encoder or propagator. Training-only raw
mean/std (std floor 1e-4), asinh, then training-only transformed mean/std (floor
1e-4) define normalization. There is no quantile clipping. Active channels are
determined on train; each of the four measurement groups has total weight 1/4.
All 244 columns are active in this release. Normalizers use all 64 training
observations, never test. Inverse transformation is the corresponding sinh.

## Fixed scientific training

Six representations, each with seeds 20261025/26/27: PCA, linear VAMP, structural
VAE, neural VAMP, VAMP plus present-measurement reconstruction (hybrid), and VAMP
with independently sampled future partners within the same source (shuffle).
All output 32 continuous features. Neural encoders have two 256-wide SiLU layers.
VAE latent posterior log variance is bounded to [-12,8]; its reconstruction loss
is group-weighted MSE and its mean coordinate KL has weight 0.001. Downstream
features use posterior means, not sampled VAE posteriors.

Neural VAMP uses centered C00, C11, C01 and separate Cholesky whiteners, without
reversibility or covariance symmetrization. Objective is minus squared Frobenius
norm of the whitened cross covariance, divided by 32. C00/C11 have absolute
ridge 0.001; covariance and Cholesky use float64, neural operations float32.
Lag 5 or 10 frames (0.5/1 ps) alternates per update. Present positions are 3..53
inclusive. The hybrid adds group-weighted present reconstruction with weight 1.
The shuffle preserves each source's marginal population but breaks local time
pairing. Linear VAMP pools the same two lags and uses the established linear
implementation's relative ridge 0.001 times each largest covariance eigenvalue;
it is a linear control, not an exact neural-ridge ablation. PCA fits current
training covariance. Linear representations have no iterative training epochs.

Neural representation training lasts 12 epochs over 3,840*51 pairs, 2,304 updates.
Every method then receives a fresh 32->256->256->244 decoder, with its encoder
frozen, for the same 12 epochs/2,304 updates. This decoder is distinct from the
VAE/hybrid decoder used during representation training. Finally an eight-component
diagonal Gaussian mixture predicts the standardized latent increment conditional
on the last four latent observations. Latent location/scale are fitted on train
only (scale floor 1e-5). Component scales are softplus plus 0.02. The transition
MLP has two 256-wide SiLU layers. It minimizes mixture NLL divided by 32 for 12
epochs over 3,840*60 transitions, 2,700 updates. This NLL is in each model's own
standardized latent coordinates and is not a cross-representation proper score.

All iterative stages: AdamW, LR 0.001, one-epoch warmup, cosine decay to 1% LR,
weight decay 1e-5, gradient norm clip 10, batch and microbatch 1,024. This explicit
batch deviation supports full-batch covariance estimates. Each stage uses a
fixed final epoch; no test selection, inner holdout, calibration, early stopping,
AP objective or AP-dependent promotion. Neural pipelines have 7,308 updates,
linear pipelines 5,004. Training online W&B is one resumable run per pipeline;
diagnostic readouts remain local and update the recorded training summary only.

`learning.csv` contains stage/global updates, epoch, objective, LR and gradient
norm, with stage-specific reconstruction, VAMP, KL, latent variance or NLL.
An omitted objective is logged zero, not a scientific zero-response evaluation.
`representation-rank.json` is training covariance participation rank
(sum eigenvalues)^2/sum squared eigenvalues. `latent_spectrum.csv` instead reports
the entropy effective rank exp(-sum p log p), with covariance eigenvalues from
the train-only phase-probe sample. These are explicitly different definitions.

## Crystal annotations and frozen readout

Existing PTM outputs are reused with their original cutoff 0.1; crystal types
are 1/2/3. Local crystal fraction uses the same nearest-80 support and Ni length
normalization as the original measurement producer. The central tracked
component size is retained; component zero means no tracked component >=4,
not necessarily a liquid atom. New labels are evaluation annotations only.

A train-only logistic readout of standardized latent coordinates is fitted on
up to 60,000 train states with equal source allocation, C=1, maximum 1,000 LBFGS
iterations. Its threshold 0.5 is frozen. No phase labels train the encoder or
MDN. Generated latent phases use this imperfect readout; observed phases use
actual PTM. Encoded-saved-MD controls quantify readout/representation error.

`retention.csv` uses all 10,560 current test observations. Group normalized MSE,
raw ordering MSE, within-source R2 and Spearman retain continuous information;
the ordering quantities are qbar6, l6 coherence mean, qbar4 and wbar6. R2 is
1-MSE/within-source target variance; undefined variance/correlation is blank,
never zero. Phase Brier, log loss (probability clipped to [1e-7,1-1e-7] for logs)
and threshold accuracy are separate diagnostics. Present liquid means observed
noncrystalline center and local crystal fraction <=0.05. It is an evaluation
subset, not a new filter on nucleus assay rows.

Each statistic first averages windows within ancestors, then gives ancestors
equal weight. The 1,000-draw whole-source bootstrap reports 2.5/97.5 percentiles.
Within-source R2/Spearman use only ancestors where defined and report the count.
`retention_sources.csv` retains source contributions for paired comparisons.

## Generated trajectories and temporal structure

The forecast panel is fixed by hash of seed and row identity: at most four test
rows per ancestor/regime, independent of outcomes within those predefined
regimes. This release has 464 rows from 44 ancestors. Each has 32 autonomous
neural trajectories of 60 native steps. These are neural samples, not additional
physical MD continuations. Evaluation horizons are 1, 3 and 6 ps. There is only
one observed physical future per input, so conditional distribution scores must
be assessed across sources; they cannot prove an individual conditional law.

Controls are recursive conditional-mean transitions, persistence of the actual
current measurement, and encoding/decoding the actual saved future. The last is
retrospective reconstruction, never a forecast. Forecasts begin from the four
actual observed measurement states. No generated measurements are clipped to
physical ranges or replaced by observed futures.

For paths A,B with T times and group/channel weights w summing to one, distance
is sqrt(mean_t sum_d w_d (A_td-B_td)^2). Energy score is mean sample-to-truth
distance minus half the mean distance over distinct ensemble pairs. This uses
the unbiased distinct-pair estimator; for one deterministic path it reduces to
distance. `path_energy` scores the connected path. `increment_energy` applies the
same score to consecutive increments, anchored at the actual current
measurement, standardized by training RMS increments with floor 0.05.
`increment_correlation_energy` scores per-channel increment lag products at
0.1 and 0.5 ps, averaged over times within the horizon, retaining their amplitude.
These scores are comparable across representations in the same measured space.

`endpoint_mse` uses the ensemble-mean final observation. `temporal_power` is mean
within-particle time variance; `truth_temporal_power` its observed counterpart.
`ensemble_power` is mean variance across particles. `coverage90` is the weighted
fraction of true time/channels within empirical ensemble 5/95 percentiles; a
deterministic control has a zero-width interval. These diagnose collapse and
overdispersion alongside the proper path score.

Continuous raw ordering is standardized by training raw mean/std and its lag
product is averaged at 0.1/0.3/0.5/1/3 ps over the full 6 ps path. The table keeps
predicted and observed products and their squared difference. There is no
per-path centering/rescaling that would conceal amplitude collapse. A diagnostic
high-order liquid state exceeds both training-liquid 80th percentiles of qbar6
and coherence while remaining phase-readout noncrystalline (actual PTM for
observations). This is an operational ordering bin, not an established distinct
phase. Occupancy and longest occupied-bin duration are compared to observations;
duration is consecutive sampled frames times 0.1 ps, not continuous residence.

Local crystal entry is completion of three consecutive crystalline future
frames, with time recorded at the third. Earliest entry is 0.3 ps; the span
between its first and third samples is 0.2 ps. It does not establish uninterrupted
behavior between samples or a connected nucleus. Ensemble entry probability,
observed event frequency and Brier are reported among initially noncrystalline
centers, and separately among the stricter present-liquid subset. Probability
Brier at individual future observations is also retained. `trajectory.csv`
uses source-balanced means/intervals; `trajectory_sources.csv` retains paired
source contributions. Regimes are reported separately as well as pooled.

Raw sinh overflow is explicitly invalid, never clipped. Invalid fractions and
failure receipts accompany missing raw metrics; comparisons must not silently
drop invalid paths or compare only surviving trajectories. Finite normalized
scores and present-state retention remain available. Nonfinite autonomous latent
or normalized trajectories are a recorded generation failure, not a good score;
independent nucleus readouts and other fits continue.

## Actual nucleus prediction

The unchanged assay has 19,800 rows (19,652 unique current inputs), 6,840 train
and 12,960 test. Declared ancestry is 28 train/62 test; actual qualifying assay
rows represent 21 train/42 test ancestors. No additional neighborhood filtering.
Default nuclei have 64 atoms; 32-atom nuclei are a separate task. Existing birth
definitions, 1.5 ps persistence, 20% case-control prevalence and leads 1/3/6 ps
are retained. Test has 81 default64 events from 38 ancestors and 135 size32 events
from 42 ancestors. Native representation-training ancestors do not cross birth
test ancestry. Original catalogue roles are preserved as provenance.

The learned encoder sees present 244 measurements only in this assay. Birth
history retains the original four-time fixed-neighbor-membership support,
different from the reselected native bank. It is used only in explicit actual
history controls/concatenations, not supplied to the newly trained propagator.
Readout spaces: latent32, reconstructed244, actual current244+latent32, and
actual measured history976+latent32. Controls: train prior, structural227,
actual current244 and actual measured history976. All transforms fit on train.

Every nonconstant readout is fixed CatBoost: 600 trees, depth4, LR0.04, L2=10,
64 borders, predictive Logloss, seed20261025, no eval_set, early stopping or
calibration fit. Case-control weights and existing rows are unchanged. Scores
are weighted NLL, Brier, AP and AUROC; AP is diagnostic only. Reliability tables
use ten fixed probability bins and never fit a calibration map. The bootstrap
resamples whole ancestors 1,000 times with the shared birth assay implementation.

`birth_individual/summary/paired.csv` retain individual fits, seed means/population
SD, and matched-seed left-minus-right contrasts on identical test row identities.
Native comparisons likewise pair identical source/scientific scopes and seeds.
Whole-source intervals average paired bootstrap differences across available
seeds; they are conditional on fitted models, not independent-seed population
confidence intervals. Missing fits retain explicit counts/failure receipts and
are never represented as completed three-seed results. This experiment tests
local latent ordering and separately detects full-cell-labelled births; it
does not generate a whole-cell connected nucleus or estimate a physical rate.
