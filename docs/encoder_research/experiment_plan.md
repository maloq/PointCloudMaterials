# Experiment and methods-update plan

**Status, 12 October 2026: bounded E0/E2/E3/E4 diagnostics, their transformer/readout follow-up, and the separate six-fit direct-measurement queue are complete. The October 12 Ni continuation is complete: all15 scientific fits and evaluations,78 response readouts and3 history readouts finished by08:15 CEST. See the [completed findings](evidence.md#measurement-overnight-20261012). Completion receipts are authoritative for execution status.** The supplied broader plan was proposed against repository main at **05e3e8ca57f42ea95f63f7f11c0aa05426fa5896**. This page records actual execution alongside designs and gates. It follows the [methods](./methods.md), [evidence register](./evidence.md) and [literature review](./literature.md).

E0 freezes definitions. E1–E7 use existing records, frozen inference or bounded readouts where available; independent branches can proceed after E0. E8 acquires physical information only for a surviving question. E9 implements the remedy supported by those diagnostics. Counts below bound an initial design; they are not a power analysis or a commitment to execute every branch.

<a id="execution-status"></a>

## Authorized density-rate and source-expansion study, October 13

**Complete:** all 36 fits reached the fixed 1,600-update checkpoint; all 36
evaluations, 18 readouts and 751 anchor packets are complete. Last evaluation
finished about 6.01 hours after launch. See the
[results and limitations](evidence.md#density-rates-20261013).

The user authorized the proposed input-rate and training-data experiments, with
a smaller neural step and a project-wide train/test-only default. The new
`density_rates_20261013.json` recipe fixes36 fits of1,600 updates:24 diverse-bank
fits (rate input × response loss ×1/5-fs step ×three seeds), plus12 controls on
80-state/original16 and320-state/original16 banks. The expanded320-state bank
uses28 training ancestors and matches the original320 temperature composition.
All former nontraining roles become test:310 physical anchors from62 ancestors.
Original roles remain provenance; there is no inner selection set, test tuning
or checkpoint promotion. The final update is the primary checkpoint.

The extra74 inputs are the instantaneous fixed-box atomic-motion contribution
to density changes, D_x rho[V], computed from actual Q/V. They condition the
initial token; they do not change the244 future targets. Controls retain equal
parameter capacity. All physical outputs and tangent targets are recorded at
5,10,...100fs for both neural steps; physical integration remains1fs with
float64 states and float32 forces. The existing300 physical anchors are reused;
451 new anchors complete the source/data contrasts and expanded test coverage.

Primary questions are held-out value/path likelihood, sensitivity beyond zero
and persistence, and rate-input × derivative interaction, with source uncertainty.
The18 fixed response readouts use no parameter search. Per-shot target-noise
diagnostics accompany the derivative scores. There is no new long-native fit
or birth evaluation in this authorized queue. [Execution workflow](../measurement_sim.md#density-rates-source-expansion-and-smaller-steps--october-13)
and [metric definitions](../metrics/measurement_rates.md) record the implementation;
launch/checkpoint receipts establish actual progress.

## Executed scope and next decisions

Run: [methods-diagnostics-20261011-v1](../../output/ni_latent_sim/methods-diagnostics-20261011-v1/README.md). All three jobs completed; [protocol](../../experiments/ni_latent_sim_20261010/METHODS_DIAGNOSTICS.md), [metric definitions](../metrics/ni_motion_diagnostics.md) and [findings](evidence.md#ni-methods-diagnostics) distinguish the implemented subset from the wider hypotheses.

| Stage | Executed scope | Decision and remaining work |
| --- | --- | --- |
| E0 | Pinned three completed 3,200-update MLP checkpoints, source roles, actual time grids, feature normalization, physical kernels and random streams. | Covers this Ni audit; it is not a full event-eligibility or restart-context ledger. Historical held-out sources remain exploratory development evidence. |
| E2 | 214 rows, all 40 held-out ancestors, three sampling keys, nested K=4/8/16/32/64/128; full smooth24 mean/variance/sample paths through 6 ps and full280 endpoint scores. All-pairs U energy allows arbitrary within-time particle permutations while preserving endpoint multisets. | Complete for this panel. Thirteen ancestors lack growing-front rows; missing strata remain explicit. A positive scramble gap in path-response coincides with runaway tails, so it does not validate useful dynamics. Full280 path/tail audits and any emission/transition remedy remain separate work. |
| E3 | 28 local Gaussian ridge fits: raw280 and three compact16 checkpoints, 1/4/8/14 exact observations, ordered and unordered-past controls, fixed-model reversal, common shifted origin and .1/.5/1/3-ps targets. | Complete for these readouts; no compact64 comparison. Longer raw histories help longer-offset MSE, while compact ordered histories mostly saturate at four observations. Capacity, regime-specific comparisons and autonomous memory remedies remain open. |
| E4 | Both saved physical shots from one anchor per each of 60 ancestors; 40 held-out anchors for student scoring; all six K values, three sampling keys and epsilon=.5/.25/.125; raw errors, shot disagreement, variance/K accounting and training-batch normalizers. | Complete for measurement/loss diagnostics. None beats the pooled zero-response error. The observed variance/K penalty is small; prioritize observation/teacher compatibility and fitting controls before changing that estimator. Physical finite differences, refreshed-versus-fixed support and rank-sufficient direction tests remain unrun. |
| E1, E5–E9 | Not launched by this diagnostic queue. | Retain their prerequisites and conditional gates below. No new physical MD or neural training remedy has been run as part of this audit. |

Do not infer that a small variance penalty at selected checkpoints proves it had no effect during training. Conversely, it is not evidence for making U-statistic training the primary remedy. The response target remains measurable above the estimated two-shot noise, but its compatibility with the permitted local observation has not been established. No AP-based promotion follows from these diagnostics.

The later [methods handoff](../../experiments/ni_latent_sim_20261010/METHODS_HANDOFF_20261012.md) records the completed transformer and response-readout follow-up. Separately, all six [direct244 fits](evidence.md#direct-measurement-followup) now have completed evaluations. Neither establishes a consistent derivative-training benefit. The bounded follow-up and its expanded overnight execution are below; the broader E0–E9 descriptions remain the reference framework.

<a id="ni-followup-20261012"></a>

## Ni continuation, 12 October

**Objective:** make simulator-derived motion information useful for future prediction. Keep the direct244 measurements as the primary observation. First distinguish a failed response fitting procedure, limited physical fitting data and a mismatch between the physical intervention and the native forecast task. A new architecture or a stronger derivative weight is not yet supported by the completed comparisons.

### Fixed scope and notation

Reuse `ni-direct-measurements-20261011` and `ni-dense-path-response-20261011`:

- Native data: 14,400 fixed-central-atom sequences from 60 ancestors, with unchanged 3,840/960/9,600 fitting/selection/evaluation rows and 16/4/40 ancestors. Some liquid/front rows were selected using future behavior; retain regime-stratified scores and do not infer natural birth prevalence.
- Physical data: 300 anchors, two shots each; 80/20/200 anchors from the same 16/4/40 ancestor roles. Extra query directions or shots are not extra independent sources. No new MD is required for the first stage.
- Inputs: BO40 + TDA113 + density74 + velocity17, with no new spatial encoder, structural replay, temperature, source age, absolute-time, box or species covariates. Length normalization and units remain explicit.
- Future offset is always relative to the declared forecast origin. Native observations are 0.1 ps apart. The physical teacher integrates every 1 fs and records every 5 fs through 100 fs. A 5-fs neural step does not add native ground truth.

For the response experiment, let **H** denote the permitted observed history. Let **R(tau)** denote the 91 smooth measurement responses at future offset **tau** to one specified initial position or velocity perturbation. **R(0)** is computed from the current Q/V and that perturbation, using the measurement function; it is not taken from a future teacher value. Q and V query results remain separately reported. BO/TDA are context/value targets, not claimed to have smooth analytic response targets.

Historical held-out sources have already informed development. Preserve their roles, but call this exploratory evaluation; a later confirmatory claim needs genuinely unused ancestors. Do not obtain that confirmation by resplitting already inspected sources.

### 1. Repair the response assay and measure its data dependence — E4/E7

Fit response change around the known current response:

\[
\widehat R(\tau)=R(0)+\Delta R_\theta(H,R(0))_\tau.
\]

The correction is exactly zero at tau=0. Future offsets index fixed output slots; they are not time covariates supplied to the predictor. Make the response map linear in the supplied perturbation response, while allowing nonlinear dependence on H. Include cross-channel terms: an initial velocity perturbation has zero present structural response but can have a nonzero future structural response through its velocity channels.

Compare zero response, present-response persistence, a fitted global linear transport of R(0), a linear residual readout and a state-conditioned residual readout. Use small, predeclared regularization/kernel grids selected by proper selection likelihood. Direct244 is primary; recreating all nine historical representations is not required. Supplying R(0) changes the information contract: this tests response evolution given a known response, not the sufficiency of H alone.

**Positive controls and limits.** Verify present response and the identity transport in an end-to-end calculation. Exact t0 recovery enforced by a skip connection is a boundary condition, not evidence of learned future sensitivity. Compare a fitted simple transport with the flexible readout to expose failure to recover an easy relation. Evaluate actual future errors at 5/20/50/100 fs, with group/direction raw powers, half-amplitude secants and source-paired uncertainty. Reuse per-shot packets to distinguish shot noise from input-response approximation. The old geometry128 noise estimates do not automatically apply to the new91 targets.

**Fitting-data curve.** Use 4/8/12/16 fitting ancestors, hence 20/40/60/80 physical anchors. Predeclare three nested ancestor orderings, sharing the full16 fit: at most ten distinct subsets. With two residual readout families and separate Q/V fits, this is at most 40 selected frozen readouts, not 40 neural trajectory trainings. Fit each subset's readout transforms without using the other fitting ancestors' response targets; keep selection/evaluation ancestors unchanged and score in common physical/normalized evaluation units. Query pairs, shots and neighbors never cross ancestry roles.

**Decision.** Proposed practical target: at least 5% lower primary future-response error than the better of zero and persistence over 20–100 fs, with 50/100-fs errors reported separately and source-paired intervals. A global linear transport meeting this criterion is already a useful result; a more flexible state-conditioned readout must improve on it to justify its added complexity. Fix group/direction aggregation before fitting. A small or uncertain gain remains unresolved, rather than proving missing information. Improvement confined to an explicitly reported channel family supports only that narrower response task. A flat curve from a failed positive control does not justify collecting more of the same targets.

The ancestor-count curve changes both fitting-state count and source diversity. Before choosing an expansion within the same16 ancestors, add the cheap complementary curve using 1/2/3/5 anchors per each of those ancestors under one fixed nested ordering. This gives16/32/48/80 fitting anchors, with the full80 fits reused: at most12 additional selected frozen readouts. A gain from more independent ancestors alone does not imply that more correlated states from the same ancestors will suffice.

### 2. Test longer observed memory using the existing measurements — E3/E9

Run small frozen readouts on three contexts: four ordered observations; fourteen ordered observations; and the current observation plus mean/standard deviation of the preceding thirteen observations. The last context is invariant to reordering the past; fixed reversal of a fitted model is not a substitute for this comparison. Predict the same measurement targets with the same permitted inputs and a matched regularization budget.

**Exact time contract:** move every model's origin to old t0+1 ps. Four observations then cover old t0+0.7 through t0+1 ps; fourteen cover old t0-0.3 through t0+1 ps. The retained data end at old t0+6 ps, leaving **5 ps**, not 6 ps, of future. Use common 0.1/0.5/1/3/5-ps readout targets and all native future observations for a subsequent rollout. No interpolation, extra future-informed row filtering or unmatched origin is allowed. A future 6-ps comparison would need additional genuine saved observations and its own availability manifest.

Choose one memory candidate by the declared selection likelihood: fourteen ordered observations or the simpler past summary if it is adequate. If selection evidence supports extra history, the scientific contrast is **four observations versus that one candidate, three paired seeds, six fits**, all at the shifted origin. Keep direct measurements, connected objective, maximum forecast horizon, effective batch, exposures, initialization pairing and selector identical. The current unshifted models are historical references, not matched controls for this experiment.

Start with 1,600 optimizer updates per fit, effective batch256, maximum learning rate0.001 with warmup/cosine, and autonomous 5-fs steps. Use common 1/3/5-ps horizon sampling. Benchmark the longer-memory peak allocation first; if accumulation is needed, record it and match the effective batch/exposures across treatments. No derivative loss enters this memory contrast.

**Decision.** Seek a reproducible proper-score gain and useful temporal dependence, not just larger path variance. Report native increments, lag products, source-specific trajectories and the known initial-state control. A readout history gain does not by itself establish successful autonomous rollout or a closed Markov state.

### 3. Learn a physical short-path predictor before coupling its derivatives to long NPT forecasts — E4/S3/E9

If the response-change assay works, fit a **matched short physical task** on the existing 5–100-fs BAOAB bank: same original80 identities, same measurement producer, and the same kernel for both value and derivative targets. Compare values-only with values+physical derivatives using three paired seeds. Score actual derivatives/secants of the predicted smooth measurements, not merely an unrelated response head. Retain both physical shots for predictive value scoring and their mean response for the corresponding mean-sensitivity target.

This is a bounded 100-fs task, not 6-ps response training. Matching support/kernel removes two known differences but does not prove that local observations determine full-cell interventional response. Keep the 200 original held-out anchors, present-response and zero controls, groupwise scale floors, common-noise student secants and per-ancestor uncertainty. Do not increase the derivative weight to compensate for a failed readout or control. No new potential implementation or broad force-validation campaign is proposed.

**Decision.** A successful response readout but unsuccessful trajectory derivative identifies a fitting/model-consistency problem worth addressing. If both succeed and a larger fitting cohort is justified by the curve, acquire a bounded first expansion to 320 fitting anchors from the existing16 fitting ancestors (240 additional anchors), with no additional test ancestors reassigned. This increases fitting states, not independent histories. Broader confirmation would require new independent ancestors. Freeze kernel, directions, shot count, stored precision and measured GPU budget before any physical acquisition.

### 4. Transfer a useful physical response task into the longer predictor — conditional E9

Only after the short response task succeeds, compare the same selected native forecaster with and without an auxiliary physical-response task, across three paired seeds. The two tasks share the learned motion/context computation; the response branch also receives R(0) and predicts its subsequent change. Ordinary forecast inference requires only the declared native measurement history. No crystal-specific loss or structural-encoder retraining is introduced.

Keep the physical intervention head separate from the native NPT forecast target. Equating the derivative of a partial-observation conditional NPT mean with a fixed-box physical intervention is an additional hypothesis, not an identity. Evaluate the true sensitivity of the forecast separately; good auxiliary-head error alone does not establish a physically correct generated trajectory. The existing direct244 predictor already has an exact observed starting point, so adding another initial anchor is not a new treatment.

**Decision.** The response task must improve held-out native predictive/path scores or frozen future readouts across seeds, without unacceptable calibration/support deterioration. Better training response loss alone does not justify adoption. Use the existing Ni first-appearance and persistent32/64-atom tasks later as frozen likelihood-fitted diagnostics; 64 is the existing default establishment size. Preserve their source/lead/center definitions, distinguish these event-centered assays from prospective discovery, and retain AP only as an evaluation diagnostic.

### Overnight execution approved 12 October

The user requested a full-night queue with more experiments. The concrete recipe is
[`measurements_overnight_20261012.json`](../../configs/ni_latent_sim/measurements_overnight_20261012.json),
implemented in the [measurement workflow](../measurement_sim.md#overnight-memory-and-matched-response-study).
It reuses the complete available motion/response cohorts; it does not claim new
independent sources. Longer common histories expose more observed frames per
forecast. No broad spatial-pretraining data enter motion fits.

| Contrast | Scientific fits | Updates per fit | Data and future |
|---|---:|---:|---|
| Ordered history4 / ordered history14 / summarized history14 | 9 (three seeds each) | 2,400 | 14,400 native sequences, common +1-ps origin, 5-ps future |
| Physical values / physical values + derivatives | 6 (three seeds each) | 4,000 | 300 existing anchors, 80 fitting, 100-fs fixed-support BAOAB future |

In parallel, local diagnostics fit 78 response readouts (three families × two
query directions × 13 distinct fitting subsets) and three common-origin history
readouts. The extra transport readout and within-source density curve expand the
smaller initially proposed assay. All original roles remain fixed. Readout
regularization and neural checkpoints use declared selection likelihoods.

Both short physical treatments run as bounded diagnostic experiments independently
of the response-assay outcome. This is an explicit expansion of the initially
proposed single-contrast queue: a failed readout will not block the native memory
comparison or be hidden by changing the physical loss. Promotion to a joint native
forecast with response auxiliaries still requires evidence and is **not** in this
queue. Nor are new physical teacher acquisition, a stronger response weight,
structural-encoder retraining or retrospective birth-AP optimization.

Held-out evaluations use32 particles, source-balanced scores, fixed examples and
within-time particle shuffling. Physical response additionally uses three outer
noise keys and half-amplitude secants. Native full-cohort scores use one key,
with three keys saved for examples. The larger Kmax128 audit of the six historical
checkpoints remains a separate proposed diagnostic, not a completed part of this
queue. [Metric definitions](../metrics/measurement_followup.md) explicitly retain
the different observation grids, targets, norms and shot counts.

The launcher freezes source/configuration/metric hashes, starts two detached GPU
workers, submits a CPU diagnostic job, and pre-submits one two-GPU continuation
reservation after the current allocation. RTX6000PRO/A100/A40/L40S are alternatives.
Native effective batch256 uses accumulation on smaller devices. Every50 updates
and at allocation boundaries it saves resumable state. Each failed stage gets one
same-setting retry, with its failure retained, then independent fits proceed.
Training and evaluation do not depend on a chat wake-up. Launch receipts and
`technical/lane-*.json` in `output/measurement_sim/overnight-20261012-v1` record
actual execution; these instructions alone do not prove a job is running.

All primary predictions and particles remain in scores, including excursions or invalid outputs. Audit support per actual target definition: signed bond-order invariants and signed velocity measurements must not be forced positive. If constraints need repair, use a separately declared appropriate output law or fixed transform with correct zero handling/Jacobian; do not silently clip particles. Report calibration and error by measurement family/regime and use ancestor-paired intervals. Keep numerical sampling variation, fitting seeds and independent physical sources distinct.

The present priorities are residual-response learnability, fitting-data dependence, matched teacher semantics and useful memory. Larger generic transformers, broad new shooting, different structural encoders, distributional-response targets and a U-statistic training rewrite remain conditional candidates rather than the first queue.

<a id="e0"></a>

## E0 — Freeze definitions and verify prerequisites

**Coverage: all hypotheses.** On the execution system, consult the current [dataset registry][datasets] and existing receipts. Locate retained checkpoints, normalization, predictions, full native frames, component/source-state records and physical-shot packets. A frozen repository report does not establish current storage availability.

Produce one manifest defining complete input/context identities, targets and ascertainment, actual physical offsets, source roles, weights, transformations, scientific contrast, practical tolerance, selection budget and metric version. Record independent source counts and which historical holdouts have already informed development.

**Gate:** proceed only with the corresponding prerequisites. Missing full-state records narrow support/identity claims; missing independent sources limit confirmation. Preserve the [repository policies][agents], current cohorts and exact cadences. Do not reconstruct retired production histories, add birth-cohort filters or introduce forbidden conditioning variables.

<a id="e1"></a>

## E1 — Eligibility, aliases and population weights

**Coverage: H4/H5/H6.** Reconstruct the diagnostic ledger from retained event, component and source-state records. Classify archive copies, independent trials, hidden-context differences and kernel differences. Recover actual horizon labels, attempted/unresolved trials and present eligibility.

Report known/unknown weighted mass, event-risk bounds and the four weighting panels on matched rows. Use the three-way eligibility/weight bridge in [H4](./methods.md#h4). Reuse predictions only after complete model-input and calibration-contract equality; identical coordinates alone are insufficient.

**Gate:** close a nonexistent or practically negligible effect only when uncertainty supports that conclusion. Otherwise freeze the alternative diagnostic estimand before fitting. Keep the canonical release unchanged. Unknown labels, unavailable inclusion probabilities and unidentified context mixtures remain explicit.

<a id="e2"></a>

## E2 — Joint paths, particles and support

**Coverage: H1/S1; supplies S4.** Start with **up to 256 source-balanced rows**, **at most three relevant retained checkpoints**, **Kmax=128**, and **three independent outer RNG keys**: at most nine checkpoint/key evaluation passes before expansion. Each Kmax output supplies nested K=4/8/16/32/64/128 prefixes, not six independent reruns.

At native physical times, retain full emitted paths for the fixed 24-observable panel, corresponding means/variances and identities. Stream full-280 endpoint/tail summaries. Archive full-280 trajectories only for a small registered subset.

Measure endpoint ranking, calibration, future-to-future dependence, excursions, decoded-mean Monte Carlo error, emission/process-mean variance contributions and de-standardized negative mass. Use pair-preserving scrambling and a control with demonstrable temporal signal.

**Gate:** identify a material dependence, numerical-resolution or support discrepancy before proposing repair. Weak reference dependence or insufficient sources gives “unresolved.” Expand rows only for a concrete remaining uncertainty. K and outer keys are numerical replication, not additional physical ancestors.

<a id="e3"></a>

## E3 — Common-anchor history and compression

**Coverage: H3; supplies S4.** Use the later +1-ps anchor within existing −0.3 to +6-ps windows. Compare **1/4/8/14 observations**, raw 280-vectors and available 16/64 codes: **up to 12 ordered representation/history cells**, with common **0.1/0.5/1/3-ps futures**. E0 fixes the small readout family, selection budget and number of target/readout fits separately.

Include a predeclared order-restricted past summary or randomized past-order treatment. Fixed reversal after refitting does not remove information. Keep current-frame inputs and population weights matched.

**Gate:** distinguish history value, compression/readout limits and ordering value. Choose the simplest adequate diagnostic input. A successful frozen readout does not establish autonomous simulation; any memory remedy requires E9 path validation. Report the three Ni sampling regimes separately.

<a id="e4"></a>

## E4 — Response reliability, compatibility and support

**Coverage: H2/S1/S2/S3.** Use a small source-balanced panel, with its state count and availability rationale fixed in E0. Re-encode individual-shot feature values/JVPs from physical packets; existing learned-feature targets average shots.

Compare nested **K=4/16/64** mean estimates and repeated outer keys; report raw response errors, powers, batch normalizers and finite-particle variation. Check common-state producer agreement, student secant-size behavior and native fixed-ID versus refreshed support using full frames.

Physical finite-difference checks require paired physical trajectories. Two shots or two directions may leave reliability/rank questions unresolved. The teacher's **1-fs integration**, **5-fs recording** and **100-fs horizon** remain distinct from long native 0.1-ps observations.

**Gate:** distinguish measurement, observation mismatch, compression and fitting before changing response strength. Send E8 only the missing physical comparison or precision requirement. Do not infer a derivative obstruction from rank-insufficient/noisy directions.

<a id="e5"></a>

## E5 — Cross domain and inspect producer semantics

**Coverage: H7/S6.** Use **R0/R1 × observed/relaxed inputs × three existing encoder seeds at one maturity = 12 encoder-domain cells**, each evaluated on fixed structural and native-future assays. R0/R1 isolate alignment within the Epi+variance recipe.

Use common paired rows and targets, per-input training-fitted scaling and common target normalization. Read existing symmetry receipts first; add only unresolved target-appropriate producer diagnostics. R2 is a conditional extension.

**Gate:** determine whether the apparent cost is an assay/domain effect, normalization issue or persistent representation trade-off. No new encoder training belongs to this screen. Preserve immutable frozen buffers; any projector or joint-normalization treatment waits for E9.

<a id="e6"></a>

## E6 — Complete the liquid-partition contrast

**Coverage: H8.** Keep **K=7**, one alignment arm and one retained maturity. The first global/liquid × three-seed contrast has **six quantizer cells**, reusing compatible existing fits.

Extend only when needed to **two populations × two exports × two metrics × three existing seeds = at most 24 total small quantizer fits/cells**. This is not 24 new encoders or 24 additions after the first contrast. Use one common training-fitted diagonal transform per export across both populations.

**Gate:** retain a stable useful partition, change its declared export/metric, or use continuous features. A null first contrast does not automatically exclude export-by-metric interaction. Avoid a broad K sweep and any change to birth-cohort eligibility.

<a id="e7"></a>

## E7 — Readout procedure and maturity

**Coverage: S5.** For one primary treatment use **two retained maturities × three encoder seeds × three matched readout replicates = 18 frozen fits**. Reuse compatible features and fits. Fix rows, sampling mode, update budget, selector schedule and calibration.

Record encoder, head and sample-order identities separately. The three replicates quantify readout-procedure variation; they do not fully factor head and order effects unless one is controlled. Keep contrasts paired across maturity.

**Gate:** establish the precision of rankings and decide whether maturity-dependent adaptation remains relevant. A flat frozen maturity effect does not rule out an adaptation interaction. Three encoder seeds and many bootstrap draws do not create high-powered independent confirmation automatically.

<a id="e8"></a>

## E8 — Acquire only missing decisive physical information

**Coverage: H2/H5/S2/S3, potentially H4.** Require a written finding from an earlier stage: informative/null response directions, independent futures from retained full states, or a short matched support/kernel/finite-difference comparison. Design one physical question at a time.

Specify eligible states, full context and restart semantics, perturbation units, rank target, independent seed policy, native physical cutoff and all attempted/failing/unresolved trials. Generic full-rank derivative screens need **17/65/281 directions** for represented dimensions **16/64/280**; these are rank thresholds, not automatic acquisition budgets. Deliberate null/dependent directions may answer a narrower question more cheaply.

**Gate:** estimate pilot variability and cost before fixing the total shot/direction budget. Preserve source ancestry; new shots do not create new melts. Stop if the required context is unavailable or the scientifically useful precision is infeasible. Never revive deleted production trajectories.

<a id="e9"></a>

## E9 — Test one justified remedy and confirm its effect

**Coverage: the surviving H1–H8/S1–S6 discrepancy.** Compare one remedy with its exact baseline, beginning with a viability pilot and then **three matched seeds** where a fitted comparison is meaningful: normally **two treatments × three seeds = six fit cells**, less new work when valid baselines are reused. Do not launch all candidate remedies.

Examples are a small path-loss/emission-coupling treatment, explicit memory state, bottleneck change, mean-error U-statistic at fixed K, matched response target, horizon-specific risk model, projected alignment loss or one normalization treatment. Unperturbed-outcome scoring does not itself repair derivative response.

The separate S5 interaction has an upper bound of **two maturities × frozen/fine-tuned × three seeds = 12 fits**, reusing compatible frozen controls. If a continuous-time claim remains relevant, timestep sensitivity uses the same fitted drift/diffusion at **1/5/10 fs** with a fixed **100-fs history refresh**, correctly coupled Brownian increments and common physical endpoints.

**Gate:** adopt only if the targeted discrepancy improves to a practical tolerance and required endpoint/path/population checks remain adequate. Otherwise retain the simpler method or narrow its claim. Repeated use of historical holdouts remains exploratory; prospective confirmation needs a genuinely unused source role.

## Implementation and publication

Implement selected updates in the existing [Ni data/evaluation][ni-eval], [dense response][ni-dense], cohort/appearance, [quality][quality-code], spatial partition and [readout][onset-eval] workflows. Freeze new producer/metric identities and retain historical definitions. Validate the relevant existing end-to-end scientific workflow; no new unit-test suite is proposed.

Scientific training retains online tracking through the established project and resumable receipts; frozen diagnostics stay local and associated final metrics update existing run identities. Record failures and stopped branches as evidence.

Publish one consolidated analysis using [analyses, tables, plots and technical receipts][results-system]. Report numerical, source and fit uncertainty separately; functioning controls and sufficiently narrow uncertainty are prerequisites for closing a null hypothesis. Update this plan with the actual decision and evidence link rather than creating another family of small status reports.

[agents]: ../../AGENTS.md
[datasets]: ../../DATASETS.md
[ni-eval]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/evaluate.py
[ni-dense]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/dense_response.py
[quality-code]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/encoder_quality/run.py
[onset-eval]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/supervised_onset/evaluate.py
[results-system]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/docs/research_results_system.md
