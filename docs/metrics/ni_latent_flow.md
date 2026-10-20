# Native Ni trajectory distributions, memory and latent representations

This family evaluates the fixed 2026-10-18 trajectory experiment. Its central
measurement normalization, native population, ordering definitions and nucleus
assay inherit the [kinetic protocol](ni_kinetic_latent.md). Regional inputs, when
declared by the arm, retain the [spatial protocol](ni_spatial_scales.md). This
document defines the new evaluation adapter and its limits; historical outputs
retain their own recorded definitions and implementation hashes.

## Physical observations and scientific population

The central target is the original 244 measurements: BO40, TDA113, smooth
density74 and velocity17. The source bank has 14,400 fixed-central-atom windows,
3,840 train windows from 16 ancestors and 10,560 test windows from 44 ancestors.
Each window has 64 actual observations at 0.1 ps spacing. Relative to the
forecast origin, four observed frames are at -0.3,-0.2,-0.1,0 ps and 60 target
frames are at 0.1 through 6 ps. Four input observations therefore span 0.3 ps,
not 0.4 ps. Local nearest-80 membership is reselected at each native observation.
No intervening femtosecond states are observed or invented by this experiment.

All declared training ancestry is preserved; former nontraining roles map to
test with original roles retained as provenance. Central normalization is reused
unchanged. Any new transforms, representation training, diagnostic phase probes
and ordering thresholds use training ancestors only. Budgets and final-checkpoint
rules are fixed before test evaluation. There is no AP objective, test selection,
calibration fitting or early stopping.

These source histories are NPT and include outcome-enriched original,
liquid-liquid and growing-front groups. Their pooled population is not an
equilibrium ensemble or a natural-incidence sample. Neither reversibility,
detailed balance, fluctuation-dissipation consistency, physical rates nor an
equilibrium free energy follows from fitting them. A slow representation or a
reversible parameterization cannot establish those assumptions. No temperature,
simulation age, absolute time, species, PTM labels or nucleus labels are model
inputs. Physical offsets organize the observations and labels only.

## Native measurements and distribution scores

The native assessor is injected with the new model and data providers. It retains
the existing 10,560-current-state retention assay and outcome-independent forecast
panel: up to four identity-hash-ranked rows per test ancestor/regime. The full
release gives 464 forecast origins and 44 ancestors. The number of neural paths
is frozen in the recipe; 32 paths per origin means 14,848 neural paths, not new
independent physical MD continuations. These neural samples share one saved
physical target per origin.

For generated paths A,B of T frames, distance is
sqrt(mean_t sum_d w_d (A_td-B_td)^2), with each of the four measurement families
having total weight 1/4 over active columns. The energy score is mean distance
from each sample to the observed path minus half the mean distance between
distinct sample pairs. It is evaluated at 1,3,6 ps, using the actual common
244-dimensional normalized target, not a representation-specific latent distance.

`path_energy` scores connected trajectories. `increment_energy` scores their
successive increments, including the first increment from the actual observed
state at time zero; each channel is divided by its training RMS increment with
floor 0.05. `increment_correlation_energy` scores time-averaged increment products
at lags 0.1 and 0.5 ps. These are proper energy scores for the corresponding
path/derived-observable distributions; the derived scores alone do not identify
the full trajectory law. Endpoint MSE of the ensemble mean, temporal power,
ensemble power and marginal 90% interval coverage are separate diagnostics.

Controls are persistence of the actual observed state and retrospective
encoding/decoding of the actual saved future (`encoded_MD`). The latter has
future information and is never interpreted as a forecast. The deterministic
MDN control is `conditional_mean`: recursively applying the one-step conditional
mean, not the exact multistep conditional expectation. The deterministic flow
control is `zero_noise`: one zero-noise representative, not an ensemble mean or
conditional expectation. The common assay does not report teacher-forced NLL.
A flow-matching loss is not a density evaluation and must never be relabeled NLL.
`likelihood-availability.json` records this explicitly.

## Ordering and local phase diagnostics

The shared definitions remain unchanged: actual PTM types 1,2,3 identify
observed crystalline centers; a training-only logistic readout assigns an
imperfect PTM-like label to generated states. It uses up to 60,000 training
states, C=1, maximum 1,000 iterations and threshold 0.5. For a direct-measurement
model, the same readout is fitted to its 244-dimensional state; latent models
use their declared learned representation. Reported effective rank is the
entropy rank of the corresponding training covariance spectrum.

Continuous ordering retains qbar6, l6 coherence, qbar4 and wbar6, with raw errors,
within-source correlations and train-standardized lag products. A high-order
liquid diagnostic requires qbar6 and coherence above their training-liquid 80th
percentiles and a noncrystalline phase label. This is an operational bin, not a
claim of a distinct liquid phase. Occupancy and longest occupied run are scored.
The sampled duration convention is run length times 0.1 ps; it does not establish
continuous residence between observations.

Local persistent crystal entry completes three consecutive crystalline future
observations. Its earliest completion is 0.3 ps, and the first-to-last span of
those three observations is 0.2 ps. Entry probability/Brier and observation-wise
phase Brier retain the existing definitions and initially noncrystalline/present
liquid strata. Independently generated local trajectories cannot establish a
spatially connected 32- or 64-atom nucleus or a whole-cell nucleation event.

## Complete path validity and finite but impossible measurements

`validity.csv` and `validity_sources.csv` account for every requested sampled
path, deterministic path and saved-MD reference at every horizon. There is no
clipping, sample replacement, rejection/resampling or survivor-only averaging.
The audit records nonfinite latent-path, normalized-path, raw-path and raw-
coordinate fractions. Raw values use the original inverse sinh transformation;
overflow remains visible.

Necessary measurement bounds are checked directly from their producers:
q, qbar, neighbor q means/stds and coherence std lie in [0,1]; coherence means
and quantiles lie in [-1,1]; radial density masses and diagonal angular-density
powers are nonnegative; velocity second moments, squared shear and squared
rotation are nonnegative. A fixed absolute 1e-5 tolerance permits arithmetic
roundoff. Cross-radial angular powers, signed bond-power rates, signed dilation
and third-order bond invariants are not incorrectly constrained to be positive.
No new relaxation, PTM or nucleus filter is introduced.

`constraint_violation_path_fraction` counts paths with any violation, and
`constraint_violation_state_fraction` counts violating sampled states. Their
denominators include all requested paths/states. Nonfinite values have separate
counts; `raw_invalid_or_constraint_path_fraction` is their union. These are
necessary conditions only. Passing does not prove existence of a corresponding
atomistic configuration, correct forces, energy/pressure conservation or kinetics.

Excursions beyond the minimum/maximum of each normalized channel over all native
training observations are reported per coordinate and path. These are training-
support diagnostics, not mathematical physical bounds. Nonfinite values retain
their separate counters; all samples remain in the denominator.

If autonomous states become nonfinite, the shared assessor records a generation
failure and does not fabricate proper scores. The new adapter continues only the
validity accounting over the remaining scheduled origins, unless the declared
deadline requires resumption. It does not score just the surviving paths. A
complete wrapper has `technical/flow-evaluation-complete.json`; a shared
`evaluation-complete.json` alone can precede this final all-path accounting.
`validity-partial.pt` supports checkpointed continuation against the exact model,
data identity, panel and random seeds.

## Independent full nucleus assay

The original 19,800 assay rows, 19,652 unique current inputs, case-control
weights, ancestry and exact leads are retained, without neighborhood filtering.
Default nuclei have 64 atoms; size32 is a separate task. The existing 1.5 ps
persistence criterion and 1/3/6 ps lead definitions are unchanged. The available
assay rows comprise 6,840 train and 12,960 test rows; the declared 28/62 ancestral
partitions have 21/42 ancestors represented by qualifying assay rows.

Only arms declaring `birth_evaluate=true` receive new frozen readouts: 12 new
SelfTICA/FMRC encoder fits and nine inherited linear-VAMP, VAE and regional-hybrid
reference fits across three seeds. The evaluated encoder sees the present
observation, and spaces are latent state,
decoded244, actual current244 plus latent, and actual measured history976 plus
latent. Common controls are train prior, structural227, current244 and actual
history976. The birth history keeps the present atom IDs fixed backward, unlike
the native reselected-neighbor history; no generated-future birth feature is
constructed from that mismatched support. Regional arms retain their declared
present input support without adding future context.

Each nonconstant readout uses the same fixed CatBoost predictive Logloss recipe:
600 trees, depth4, LR0.04, L2=10, 64 borders, seed20261025, no eval_set, early
stopping or calibration fit. Weighted NLL, Brier, AP and AUROC are retained; AP is
diagnostic. Ten fixed probability bins provide reliability summaries. Changing
only a propagator with a frozen inherited encoder/decoder cannot change these
present-feature readouts. One representative per encoder and seed is evaluated
for paired comparisons; other propagator variants record
`not_repeated_present_features`. The direct244 arm is covered by the common
measurement control. These present-only assays do not establish a birth gain
from changing the propagator.

## Source-paired comparisons and provenance

Native and birth comparisons reuse the shared paired helpers and group by
`arm.treatment`. Native statistics average windows within an ancestor before
giving ancestors equal weight. Fixed matched model seeds are averaged first;
1,000 whole-source bootstrap draws then provide conditional 95% intervals.
Seed SD is separate, not another independent-source count. Source identities
and scientific scopes must match. Birth pairs additionally verify identical
IDs, ancestors, events, labels, weights and source bootstrap draws.

`validity_individual/paired.csv` apply the same source-paired helper to the
complete validity audit. Native and birth individual/summary/paired tables
retain missing-seed counts. Undefined source contributions are not silently
discarded, and a numerical generation failure prevents a misleading finite
trajectory contrast. Diagnostic readouts remain local; metrics update the
associated existing online scientific run rather than creating new W&B runs.

All numerical exports bind this description, the new implementation, its frozen
recipe and reused metric/data producers through the metric contract. Results
are conditional local prediction and full-cell-labelled detection diagnostics,
not evidence of equilibrium sampling or successful global latent nucleation.
