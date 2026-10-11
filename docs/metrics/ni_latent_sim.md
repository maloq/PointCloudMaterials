# Compressed stochastic local Ni simulator

## Transformer architecture comparison, October 11

`transformer_path_response_20261011.json` replaces only the fine-step MLP
transition with two causal pre-LayerNorm blocks (width64, four heads,
feedforward192, GELU, zero dropout), ordered-slot embeddings and a current-token
drift/noise head. The same16-dimensional compression, four input states,5-fs
integration and100-fs history refresh remain. No physical time covariate is
introduced. Its87,008 transition parameters compare with90,656 for the MLP.
The unchanged dense response bank is verified against its original producer
plan/packet hashes, with the same long/birth banks, scalings and roles.

Training/selection and metric calculations remain the dense-path definitions
below. The final `architecture-comparison-v1` copies matched original per-model
tables, recording their file hashes, then reports `paired_differences.csv`:
each `difference` is the transformer metric minus the MLP metric for the same
seed, time, task and feature space. It has no confidence interval or automated
promotion criterion. The copied metric rows retain their original intervals;
those are not uncertainty intervals on their difference. Figures show means
and ranges of two paired seeds. No AP-based fit, selection or model change.

## Dense-path extension, October 11

`dense_path_response_20261011.json` reuses the sealed 14,400-row expansion and
its original standardization, source roles and nucleus assays. Six fits cross
endpoint value+response, dense values only, and dense values+response with two
seeds. All use 5-fs neural integration and four native0.1ps history observations;
past slots shift every100fs, and only the current code evolves inside that
interval. Drift scales as dt and noise amplitude as sqrt(dt/0.1ps). A long
training rollout still spans1ps and a free evaluation spans6ps. No input
interpolation, teacher reset or explicit time/temperature covariate is used.

New physical continuations reuse the original300 anchors and two shot seeds,
with real 1-fs float64 BAOAB integration/analytic tangents and float32 Ni EAM
forces. Local80 atom IDs stay fixed during each teacher path. Targets at
5,10,...,100fs are the two-shot mean frozen structural code and its exact eSEN
input JVP along the propagated initial Q or V tangent. Source ancestry and
protocol remain separate from the long saved NPT paths. All perturbation
amplitudes and student epsilon0.5 remain those of the endpoint study.

`teacher_half_mse` averages half squared error across the128 standardized
structural coordinates, physical minibatch and selected time slots. For each
direction and selected time, `relative_response` computes .25 times student
response MSE divided by teacher response mean square, floor1e-5; then averages
over times and sums over the two directions. Common neural noise is used for
both signs. These means prevent twenty times from implicitly multiplying loss
weights. Batch16 short groups and four neural particles, loss weights0.1/0.1,
long NLL and selection are unchanged. Dense targets supervise the path freely
predicted from its initial history, not teacher-forced one-step transitions.

`dense_physical_path.csv` reports time-resolved source-balanced held-out values:
`value_mse` is standardized structural squared error of the four-particle mean
against the two-shot physical mean. `persistence_mse` holds the initial code.
`response_q_mse`/`response_v_mse` are per-direction standardized response MSE;
`response_q_power`/`response_v_power` are mean-square targets (also the error of
a zero-response predictor). `persistence_response_*_mse` holds the exact initial
eSEN response constant throughout time; its velocity response is initially zero.
Raw per-anchor/per-time scores are retained. These metrics average anchors
within sources, then sources equally, with1000 source-mean bootstrap draws.
`relative_response_q/v` and their persistence counterparts divide the respective
source-balanced MSE by source-balanced power, floor1e-5. They use no factor0.5;
their zero-response reference is1 when the denominator floor is inactive.
These ratios have no reported interval. Weak target powers remain explicit.
`zero_relative_response_q/v` reports power/max(power,1e-5), so the zero-response
curve remains correct even when a weak early velocity target activates the floor.

`dense-comparison-v1` concatenates frozen tables and plots two-seed means and
ranges, not confidence bands. Long evaluation includes the existing divergence
metrics and per-regime tables. Dense labels never select a model by AP. The
physical short path is newly computed; the sealed long/birth banks are reused.
The definitions below retain the historical100-fs-step experiment semantics.

This first-night experiment tests compression, finite memory, and analytic
simulator-response supervision. It is not an implementation of a published
CoBRAS algorithm, a full-cell autonomous simulator, or an atomistic generator.

Data: the pinned unrelaxed Ni parent supplies60 sources and300 outcome-blind
anchors. Sixteen fixed centers per anchor include its original response center.
Each sequence has64 exact native0.10ps observations from-0.3 through+6ps.
Nearest80 membership is recomputed at every saved frame around the same center
atom ID. Geometry uses the parent's fixed material length normalization.
The frozen, geometry-only unrelaxed eSEN produces128 structural features and128
velocity secants;24 radial/angular descriptors are appended. All280 coordinates
are standardized using only the16 inner-fitting ancestors, with scale floor.01.
Four original training ancestors select checkpoints; original selection/test
sources remain pooled evaluation. No broad spatial arrays are loaded.

Encoder inputs are these280 features, including saved velocities. Predictor
inputs are one or four compressed states at exact0.1ps spacing. No temperature,
age, absolute time, box, material/species or event label enters either network.
Data span the parent's different thermostat settings; learned dynamics is a
condition-free local observation model of that mixture. NPT cell/barostat and
surrounding atomistic degrees of freedom are unresolved, so closure is a tested
assumption rather than a physical guarantee.

Eight arms cross dimension16/64, history1/4, and value/response supervision,
each with two seeds. Total initial dynamic state is dimension times history;
history4/d16 and history1/d64 both carry64 scalars. A shared-size MLP compresses
the frozen observations; a stochastic residual recurrence produces the next
code. One recurrence step means0.1ps. Its diagonal Gaussian noise is learned.
A decoder emits a280-dimensional Gaussian observation distribution. This is
feature reconstruction, not atomic reconstruction pretraining. No relaxed
coordinates or quenched response targets are used.

Training3200 updates, batch/microbatch256, AdamW maxLR.001,200-update warmup
and cosine decay with1% floor. Long neural arithmetic is BF16; scores and
short-response computations are float32. The loss is the average mixture NLL
over all280 standardized coordinates at1/5/10 autonomous steps, plus.2 times
current-feature half-MSE, .1 short-physical-teacher half-MSE, and for response
arms .1 relative tangent loss. Four stochastic particles form each training
mixture; validation uses eight and selects by NLL at the same horizons on a
fixed256-row subset of the four inner-selection ancestors. Validation noise
is fixed. No birth metric affects training/selection.

Short teachers are the parent's100fs original-Q,V fixed-box Ni GPU EAM/FS
BAOAB runs, two noise shots, float64 MD/tangents and float32 forces. They remain
a separate physical protocol from the saved NPT sequences. Values and analytic
responses of the frozen128 geometry outputs are reused exactly. The student
uses paired centered differences with epsilon.5, common latent noise and
original directions (Q RMS.01angstrom,V RMS.1angstrom/ps). For history models,
only the present observation is perturbed; the past is held fixed. Derivative
error sums over two directions .25*MSE divided by mean-square standardized
teacher response (floor1e-5). Value and response arms use the same base teachers.
This is not an equal-cost comparison against extra perturbed-value MD examples.
Held-out physical-teacher value and response errors are measured for every arm,
including value-only models, on original held-out-source teacher groups. They
are exported in fixed16-group batches and do not select checkpoints.

Free-rollout evaluation initializes from actual input history once, then takes
60 sampled transitions without later MD feedback. Scores at0.1/.5/1/3/6ps use
32 particles. Mixture NLL is logsumexp over Gaussian components, summed over
coordinates then divided by their count, reported separately for structure,
motion and24 descriptors. MSE uses the mixture mean. Energy score is mean
Euclidean distance from sampled observation to truth minus one half mean
distance between disjoint particle pairs, divided by sqrt(dimension). Coverage90
is coordinate-wise inclusion in empirical particle5/95 percentiles. Persistence
MSE holds the initial observation constant. Variance ratio compares ensemble
variance to squared observed change, floor1e-4; it is only a descriptive statistic.
Means give each ancestor equal weight;95% intervals resample those ancestor
means1000 times. A single observed future per input is a proper-score evaluation,
not a direct measurement of its complete conditional future distribution.

Nucleus probes preserve every parent32/64 event/lead row and label, adding exact
past history without neighborhood screening. Inputs are concatenated compact
history codes. CatBoost selection uses inner-training NLL then refits on all
original train ancestors. AP/AUROC/NLL/Brier and event weighting/bootstrap follow
ni_esen_response.md. These probes test representations, not generated nucleation
rates or an atomistic PTM classifier of decoded features. No new AP selection.

Empirical lifting is tested for predeclared d16/history4 value/response models
at the first seed. Eight outcome-independent held-out query sources are used.
The decoder returns two real training full-cell states nearest in compressed
history space, compared with two random training states at the same thermostat
setting. Temperature matching only defines the physical diagnostic stratum.
Returned Q,V,box and atom identity are real source states; no atoms are invented.
Each query/candidate gets two fresh full32000-atom fixed-box BAOAB continuations
to.1/.5ps at its original temperature,1fs timestep,1ps friction, float64 states,
float32 forces. Saved cropped physical observations are prequantization float64.
The two-shot reference and four-shot decoded ensembles are compared through
standardized frozen structural/descriptor observations. Empirical energy
distance is2 mean cross-distance minus both within-ensemble means (including
diagonal), normalized by sqrt(dimension); this small-sample statistic is biased.
Mean MSE and relative box-volume discrepancy are also reported. This MD-vs-MD
check measures retrieval lifting, not equivalence of fixed-box MD and the
NPT-trained latent recurrence, nor temporal coherence of atomic backmapping.

Online W&B records scientific fits; evaluation and hardware probes stay local
and update existing run summaries. The absolute8.5-hour deadline includes
preparation, fits, readouts, and lifting. Incomplete stages are explicit and
checkpointed; they are not silently reduced or continued beyond the budget.
Compact birth-feature caches share a six-entry retained pool across both GPU
lanes; checkpoints, probe predictions and final metrics remain permanent.

## Derivative stability factorial, October 10

The separate `derivative_stability_20261010.json` recipe reuses the sealed original
data files, pinned by plan, receipt and content hashes. Sixteen new fits cross
physical-response matching on/off, learned-map Jacobian control on/off, training
rollout 1/6 ps, and two seeds. All use d16/history4; fixed initial sequence anchor
index3 makes the initial-state distribution identical across horizons. All other
training settings remain as above. The 6-ps loss/selection observes autonomous
steps1/5/10/30/60, equally weighted; the 1-ps loss uses1/5/10. Selection uses only
the original four inner-selection ancestors, never held-out birth AP. No new
physical MD or lifting runs are performed in this factorial.

The additional derivative is the exact autodiff Jacobian of a 10-step (1ps)
learned stochastic history-to-history map, with all noise draws held fixed.
It includes state dependence of both drift and diffusion. Two power iterations
on J-transpose-J estimate a high-gain unit direction. The final JVP carries
parameter gradients (reverse-over-reverse); the direction search is detached.
This estimates a spectral norm from below, not a certified global bound.
Use float32 and a32-history subbatch, independent of the batch256 likelihood.
Half start from actual training observations; half start after10/30/50 autonomous
steps, rotating deterministically. These histories are detached: the penalty
trains the transition map, not the spatial encoder or bottleneck through the
state-location producer. No held-out state is used to generate training points.

The penalty is mean squared positive part of log(gain/3), weight1.0. Here gain
is the Euclidean norm of the full four-state output perturbation divided by
the unit input perturbation. Simply copying the latest state to all four slots
already has gain sqrt(4)=2; the cap3 allows1.5 times this neutral copying gain.
We do not force physical sensitivities to zero or constrain all maps to contract.
Learned latent units and unresolved local dynamics mean this is a heuristic
regularizer, not a physical Lyapunov-exponent estimate or a stability guarantee.
`jacobian_penalty`, `jacobian_gain_mean/max` refer to this training subbatch.

Additional held-out rollout metrics use the same32 particles, ancestor weighting,
and bootstrap as the original scores. `training_mean_mse` predicts the stored
training feature mean (zero in standardized coordinates). `particle_failure_rate`
is the fraction of component mean predictions with per-coordinate mean squared
error above100 or nonfinite error. `input_failure_rate` applies this threshold
to the ensemble-mean MSE. `nonfinite_particle_rate` checks component means before
averaging; `predictive_variance` averages sampled predictive variance over output
coordinates. These operational failure thresholds do not define a physical
phase transition. Nonfinite values remain failed/undefined proper scores; they
are not dropped or silently replaced with finite values. Accuracy, proper scores,
coverage and spread must accompany stability, to expose collapsed predictions.
The comparison figure uses a logarithmic MSE axis and seed-range bands.

## Threefold phase-enriched collection

`regimes3x_20261010.json` repeats the16-arm factorial on14,400 sequences:4,800
original,4,800 liquid-persistent and4,800 growing-front windows from the same60
ancestors. Every sequence follows one fixed central atom ID across all64 frames;
neighbors are reselected, never the center. The source and role quotas, exact
PTM sampling criteria, native timeline verification and outcome-selected nature
are defined in `experiments/ni_latent_sim_20261010/REGIMES3X.md`. PTM selects rows
only and is not a model input or supervised loss. The original birth bank and
300 physical teacher groups remain unchanged. Normalization remains the original
fitting-source mean/scale, so `training_mean_mse` is specifically the ORIGINAL
training-mean control, not a refitted mean of the enriched mixture.

`free_rollout_by_regime.csv` computes the same per-input scores as `free_rollout`,
then aggregates ancestors separately within `original`, `liquid_liquid`, and
`growing_front`. Each has3,200 held-out rows; actual distinct ancestor counts are
reported. The overall table gives equal weight to held-out ancestors across their
available rows and is not a natural-occupancy score or necessarily the equal
average of the three regime tables. The original4,800 feature rows and their order
are preserved exactly, including original evaluation noise/block seeds. Birth
metrics retain all original definitions. Six disposable compact-feature caches
are shared with the base run across both GPU lanes; immutable checkpoints and
predictions remain permanent.


## Recovery and incomplete-fit reporting

Historical endpoint replay discrepancies are advisory diagnostics, not sample
selection or teacher rejection criteria. They report Q/V-separated relative L2
and maximum absolute differences, plus exact initial-state/basis agreement.
Scientific cohorts, numerical losses and checkpoint selection are unchanged.

An incomplete fit may be evaluated using its best saved validation-NLL checkpoint.
`evaluation-training-status.json` records this status, observed completed updates,
requested updates, selected update and checkpoint hash. These results must be
identified as partial training rather than compared as fully budget-matched fits.
No checkpoint is selected using evaluation metrics or birth AP.

The incremental `available_metrics.csv` copies available numerical table rows
without recalculation. Added fields identify the source table, training status
and completed updates. Its source paths retain their own frozen metric definitions;
the snapshot's contract records the aggregation implementation. Recovery exports
with changed implementation hashes use `analyses/recovery-metrics-<hash>/` beneath
the original analysis, with per-table pointers in `technical/metric-exports/`.
Original tables and their definitions remain intact. Missing stages are listed
explicitly; missing results and nonfinite scores are never replaced with zero.
