# Methods and decision rules

**Status: methods review with a completed first diagnostic stage, 11 October 2026.** The supplied source review is based on **05e3e8ca57f42ea95f63f7f11c0aa05426fa5896**. “Observed” statements below describe the cited implementation; dated audit updates refer to [new completed evidence](evidence.md#ni-methods-diagnostics). Remaining hypotheses and conditional remedies follow the [experiment plan](./experiment_plan.md#execution-status). The [literature review](./literature.md) records external support and its limits.

## Shared scientific contract

Before comparing methods, freeze the observation producer, physical target, actual saved offsets, complete input identity, source ancestry and roles, population weights, training-fitted transformations, selection budget and metric version. Keep encoder inputs, predictor inputs and training-only teachers separate. A completed diagnostic can falsify a hypothesis to useful precision, support a remedy, or leave it unresolved; a better selection score alone does not settle the mechanism.

[Repository rules][agents] retain geometry-only native spatial MACE encoders with a constant atom channel and default width/export 128; new encoder training defaults to batch/microbatch 256 unless a deviation is recorded. Motion observations may include actual velocities under their declared input contract. Temperature, material, simulation age and time remain audit metadata rather than new input channels. AP remains a diagnostic, not a fitting or selection objective. Physical-reconstruction pretraining is discontinued. New Al birth work uses the fingerprinted consolidated 0.10-ps collection; uniformly native 0.30-ps work is a separate protocol. No extra crystallinity/component/distance filters enter the consolidated birth catalogue. Retired 0.75-ps production trajectories must not be regenerated.

Current availability must be checked through the [dataset registry][datasets] before execution. Historical results keep their true inputs, targets and data use.

New Ni relaxations start from the [0.05 eV/Å recipe](../../configs/simulation/ni_relaxation_default.json): this is a maximum absolute force-component tolerance, with float64 minimizer states and float32 GPU force arithmetic. Reuse the [shared relaxation index](../data_storage.md#unified-relaxation-index); historical minima, running recipes and frozen training targets retain their recorded tolerances.

<a id="h1"></a>

## H1 — Joint observable paths

**Observed in the reviewed latent branch.** Ni training scores joint 280-coordinate mixtures separately at each horizon; its original final evaluation reports separate structure, motion and descriptor blocks. Horizon marginals do not establish cross-future coupling; separate-block reporting additionally leaves cross-block dependence unmeasured, although joint-280 training does constrain it within a horizon. That historical evaluator saves metrics rather than reusable full trajectories. Earlier forecasting studies already proposed whole-path controls. [Model][ni-model], [training][ni-train], [evaluation][ni-eval], [earlier protocol][old-forecast].

**October 11 implementation update.** The [completed diagnostic revision](../ni_motion_diagnostics.md) retains smooth24 paths, means, variances and particle identities. It uses all-pairs U energy, so its temporal permutation need not preserve the old estimator's disjoint pairs. Its synthetic sensitivity check passes, but the real path-response model's positive scramble gap is associated with rare runaway tails. The separate [direct244 measurement branch](../measurement_sim.md) already implements connected objectives; it has a different observation law and does not directly validate the latent280 branch.

**Proposed assay.** Preserve row/ancestor, checkpoint, normalization, physical-time and particle identities. On a bounded panel, save the 24 radial/angular observables throughout each emitted path, decoded means/variances, and separate transition/emission RNG identities. Stream full-280 endpoint and tail summaries; save full-280 paths only for a declared small subset.

Measure joint future pairs, such as 0.1/0.2 and 0.5/1 ps; future-to-future increments; lagged covariance; and excursion/dwell statistics using training-defined thresholds and an explicit finite-window convention. These smooth descriptors are not PTM nuclei. Anchor-to-future increments subtract a fixed observed value and remain endpoint transformations. Native long-path truth is spaced at 0.1 ps. The separate physical teacher integrates 100 steps of 1 fs and records every 5 fs, supporting only its 100-fs window. [Observables][observables], [data][ni-data], [dense protocol][dense-protocol].

Use a proper score on fixed joint/path transformations alongside endpoint calibration and support checks. Non-injective transformations need not identify the entire law. The existing disjoint-pair energy estimate is unbiased under independent particles; an all-pairs U-statistic is a variance-reduction candidate. Naively squaring Monte Carlo estimates of variogram moments can introduce another variance penalty. See [scoring foundations](./literature.md).

A temporal particle scramble is a sensitivity control. Move means, variances and associated samples together; for even particle counts preserve the evaluator's disjoint pairs as groups so endpoint energy estimates also remain unchanged. Separate-block permutations preserve separate-block metrics, not necessarily joint-280 NLL. Require meaningful reference dependence before interpreting a failure to distinguish scrambling.

**Conditional remedy.** Adequate endpoints and dependence justify retaining the fitted recipe. Otherwise test a small temporal emission-copula calibration around a frozen checkpoint if appropriate, then one small joint-path score or joint-block likelihood if needed. Preserve endpoint supervision. Stationary correlated Gaussian residuals can preserve conditional marginals while changing their temporal coupling; autonomous use additionally needs coherent residual state and initialization. Score complete emitted paths: decoded means omit emission variation. These controls may localize model contributions without identifying the physical allocation of process and emission noise. A pass supports the tested population and observables, not universal dynamics.

<a id="h2"></a>

## H2 — Response compatible with the permitted observation

**Observed.** Dense response training already uses common random numbers inside plus/minus predictions. New physical states and tangents are float64, with float32 forces and frozen feature differentiation. Response errors are normalized by sampled-batch teacher power. Preserve these controls while measuring their practical numerical limits. [Physical producer][ni-simulator], [dense objective][ni-dense].

Screen, in order: producer/measurement agreement; observation and compression; then fitting and generalization. Record Q/V conventions, centering, perturbation units, changed history slots, atom support, precision, kernel, seed coupling and standardization. Student secant-size curves are available from a checkpoint; physical finite-difference validation requires actual paired physical trajectories. Stored tangents alone cannot establish that agreement.

**October 11 audit.** Re-encoding and student secant checks completed on the declared panel. None of the three frozen models improved the pooled held-out zero-response MSE, and smaller secants changed that error little. Estimated two-shot noise is a minority of the measured response power. These observations narrow the numerical/noise explanation; they do not establish the observation/compression obstruction below. [Results and limits](evidence.md#ni-methods-diagnostics).

**October 12 follow-up.** The fitted state-conditioned readout also fails to recover the known present geometry response, while copying its measured secant succeeds. Its poor future score therefore cannot bound the information in the observation. The proposed [response-change assay](experiment_plan.md#ni-followup-20261012) supplies the known current response explicitly and tests its subsequent evolution, with global transport and fitting-data controls. An exact t0 skip is a boundary condition, not evidence of learned dynamics. The completed direct244 experiment already anchors values at the present and also lacks a consistent derivative advantage; repeating that anchor is not a new remedy. A successful response readout remains distinct from matching the actual derivative of a generated trajectory.

For an infinitesimal diagnostic, let G map the perturbed physical state q (Q/V as appropriate) to the permitted representation with past observations fixed. Let V contain perturbation directions, \(A=D_qG\,V\) their represented derivatives, and R the corresponding teacher responses. Exact local factorization requires \(R=JA\), hence

$$
R(I-A^+A)=0.
$$

A significant residual can obstruct that derivative factorization. It is not a general lower bound on the implemented finite-secant loss. First establish a linear-response/numerical plateau. Two generic directions cannot expose a right-null-space obstruction in a 16-dimensional code: use deliberately null/dependent projected directions or more directions than rank(A). Generic full-rank screens need 17/65/281 directions for 16/64/280 represented coordinates. A small singular value requires a justified tolerance or bounded-gain condition; it is not automatically absent information.

**Conditional remedy.** Repair producer disagreements before fitting. If targets are too noisy, retain shots and allocate additional measurements only to an informative subset. If compression introduces incompatibility, compare one less restrictive bottleneck or response-sensitive compression. If incompatibility already exists before compression, stop imposing that intervention as the native conditional forecast derivative; retain a separate physical probe or define a matched task. If reliable compatible targets still fit poorly, compare one optimization/normalization treatment before increasing capacity.

Report raw error, teacher power and normalizer distributions. If batch-dependent weighting is material, compare a fixed training-estimated direction/time scale with a declared floor. Keep held-out definitions unchanged. Physical interventional response and the derivative of a conditional future mean are different objects; success on one does not establish the other.

<a id="h3"></a>

## H3 — Memory, ordering and compression

**Proposed assay.** Use a common later anchor in existing native Ni windows. An anchor at +1 ps within the original −0.3 to +6 ps window supports histories of 1/4/8/14 observations at exact 0.1-ps spacing and common 0.1/0.5/1/3-ps futures. Name this diagnostic population separately from the original six-ps-at-zero assay. [Sequence producer][ni-data].

Compare raw 280-vectors and available frozen 16/64 codes with a small regularized readout family, common rows/targets and a fixed selection budget. Use learning curves or effective-complexity diagnostics: an identical ridge grid does not guarantee equal useful capacity.

A fixed reversal followed by refitting is an invertible feature permutation; isotropic ridge is invariant to it. Use reversal only on an unchanged fitted model as an explicitly possibly off-support intervention, or restrict ordering through a permutation-invariant past summary or random per-row past permutations during both fitting and evaluation. Retain the current frame separately. Such controls do not uniquely identify denoising; current-minus-past-mean can contain kinetic information.

**Conditional remedy.** If longer history helps both spaces, try a small delay, causal convolution or explicit state summary. If raw history helps but compact history does not, address compression/readout first. If an order-insensitive summary suffices, prefer it. If no history gain is resolvable with useful power, retain the current history length without claiming closure.

A finite-memory/state-space diagnostic needs training-only centering, identical forecast origins and no observed-prefix credit. Do not silently omit the unresolved stochastic force when claiming a stochastic surrogate. Keep original, liquid-persistent and front-selected Ni regimes separate: future-selected windows do not establish natural occupancy. Related memory implementations and their preprocessing qualifications are in [literature](./literature.md).

<a id="h4"></a>

## H4 — Eligibility and outcome ascertainment

**Observed.** The establishment cohort has future eligibility and a frozen persistence definition; onset-by-h ascertainment extends through h+1.5 ps. Onset and confirmation are different clocks. Some late-loss cases can have known short-horizon negatives despite common 7.5-ps exclusion; confirmed positives have an existing exemption. [Cohort producer][embryo-code], [metric definition][embryo-metrics].

**Proposed assay.** Reconstruct a deduplicated event ledger with present eligibility, onset, confirmation, terminal fate/time, administrative end, ascertainable horizons and exclusion reasons. Source size/event records alone may not recover spatial eligibility after early exclusions; retained component/source-state packets are also prerequisites. Missing information remains unidentified. Sequential pre-deduplication counters and missing-follow-up frame counts are not a population denominator.

If consequential, create a separately versioned horizon-specific diagnostic cohort. Preserve the official cohort and distinguish eligibility from changed family weights using three comparisons: official rows/weights; the same rows with diagnostic-pool weights renormalized on that subset; and the known-outcome diagnostic subset with its covered mass.

For normalized full-pool weights w_i and known-label set K_h, event risk obeys

$$
\sum_{i\in K_h}w_i y_i
\le P_h \le
\sum_{i\in K_h}w_i y_i+\sum_{i\notin K_h}w_i.
$$

These are identification bounds, not confidence intervals. Unknown labels are not negatives.

**Conditional remedy.** Retain the complete-case assay with its conditional interpretation if the affected mass/decision is negligible. Otherwise declare the prospective estimand before fitting. Start with cumulative-incidence/risk tables; a discrete-time competing-outcome model is conditional work. Establishment, dissolution/absorption and coalescence follow the declared endpoint. Unresolved identity remains missing outcome information unless an explicitly different operational target defines tracking failure as an event. Administrative censoring remains distinct.

Inverse-censoring weights require conditional independent censoring and adequate positivity. They do not automatically repair mergers, deterministic outcome-based exclusion or unresolved identities. A confirmation-time hazard is a new target, not a relabeling of onset-by-h.

<a id="h5"></a>

## H5 — Copies, independent futures and hidden context

**Proposed assay.** Distinguish duplicate archive views, genuinely independent futures from one complete state/kernel, and different hidden full-cell contexts sharing a local input. Extend existing identities with available Q/V, atom IDs/support, box, ancestry, restart/branch lineage, kernel and seed/trial provenance. Coordinates alone are insufficient for Q/V prediction; Q/V alone is insufficient when the model also consumes history.

Recover actual horizon labels. The appearance evaluator declares 0.1/0.5/1/3/6 ps for ordinary appearance and 3 ps for persistent appearance, with unsupported partitions potentially skipped. Different onset times need not imply different scored labels. [Appearance evaluator][appearance-eval].

For verified independent futures, average losses within each full context/kernel, combine contexts under a declared mixture independent of incidental shot counts, then apply the desired input-group/ancestor weights. Keep kernels separate unless their mixture is explicitly defined. Trial-weighted fitting is also legitimate if it is the intended estimand.

Count-based AIMMD fitting illustrates the distinction: configuration influence scales with resolved shot count. Desired total configuration weight W_g with B_g eligible trials requires grouped mean losses or an adapter W_g/B_g, with empty/unresolved groups handled explicitly. See [implementation and limits](./literature.md).

Retain attempted trials, time limits, failures, unresolved endpoints and acquisition proposals separately from successful counts. New directory names and restarts do not prove independent noise. Missing historical inclusion probabilities cannot be invented.

For repeated Bernoulli outcomes, mean log loss can use the success fraction. Mean per-trial Brier loss is

$$
(p-\bar y)^2+\bar y(1-\bar y).
$$

Scoring only the fraction drops the second term; paired model differences on identical groups/weights remain unchanged, absolute scores do not.

**Conditional remedy.** If aliases are copies, retain deduplication and close the branch. If real repetitions exist, reuse deterministic predictions only after equality of complete inputs, model, ordering/masks and calibration contracts. Unknown natural context mixtures require stratification or an explicitly designed mixture. New shooting is conditional on a remaining precision or identification question.

<a id="h6"></a>

## H6 — Population weights and uncertainty

**Proposed assay.** On identical rows retain the current primary weights and compare four declared populations. For nested families with n_f landmarks, n_a landmarks per ancestor and F_a families per ancestor:

| Population average | Unnormalized row weight |
| --- | --- |
| Equal landmarks | 1 |
| Equal families | 1/n_f |
| Equal ancestors | 1/n_a |
| Equal ancestors, then families | 1/(F_a n_f) |

Normalize within the specified role/population. Use consistent weights for prevalence, baselines, calibration and predictive-score contrasts describing that population. Sampling and loss weights jointly determine the effective training distribution.

Bootstrap ancestors for dependence, keeping contrasts paired. Multiply fixed point-estimand weights by each ancestor's draw multiplicity, or use separate bootstrap-copy identities. Recomputing inverse ancestor counts after pooling duplicate IDs can cancel multiplicity. The resampling unit does not redefine the point estimand.

**Conditional remedy.** If conclusions differ materially across legitimate population averages, report that dependence and choose the primary scientific population before further fitting. Show contribution/support tables when a few ancestors dominate. Numerical RNG replicates, training seeds and source uncertainty answer different questions. More resamples do not create independent physical data. Weight alignment does not make appearance, establishment and commitment the same target.

<a id="h7"></a>

## H7 — Input domain versus representation cost

**Observed.** The quality evaluator's static assay uses fixed relaxed observations, while its predictive domain path changes input and physical target together. Toggling the current domain flag does not provide an input-only comparison. [Evaluator][quality-code].

**Proposed assay.** Cross observed/relaxed input on a genuinely available paired cohort while fixing structural target, native future outcome, rows, centers and roles. Begin with R0 (Epi+variance without alignment) and R1 (same recipe with alignment), one retained maturity and three existing encoder seeds. R2 (paired VICReg) is conditional. [Treatment definitions][mechanism-protocol].

Fit each input-space standardizer on its own training features and keep common target normalization. Do not require unrelated learned spaces to use the same literal scaling array. Record paired-input availability before selection; avoid model-specific intersections. Historical test sources repeatedly used for development remain exploratory evidence.

Native paired fitting already supplies one hot and one cold view; its fixed normalization is estimated on hot training views. Joint hot/cold normalization is a new treatment, not a modification of frozen buffers. [Preparation][native-pretrain].

**Conditional remedy.** If the apparent structural cost vanishes in the matched observed domain, narrow the trade-off claim. If it persists, compare one separation of alignment/loss space from the exported representation. A bounded candidate is direct-export VICReg versus a training-only 128→128→128 projector, keeping backbone, initialization, pairs, reductions and normalization fixed. Evaluate declared exports separately. Do not change the Epi teacher simultaneously. If normalization is implicated, test that factor separately. A robust trade-off may justify distinct representations for distinct targets.

Loss coefficient names are insufficient: record loss space, dimension, covariance/variance reductions, fitted buffers and batch population. Official and native VICReg differ in these details; see [literature](./literature.md).

<a id="h8"></a>

## H8 — Partition population, export and metric

**Proposed assay.** Complete the existing frozen-feature question before searching K. First compare global versus liquid-only K=7 with representation and metric fixed. If consequential, compare the stored z/y exports and one training-fitted diagonal metric. A null first contrast does not exclude an export-by-metric interaction. [Existing study][spatial-results].

The conditional maximum is two populations × two exports × two metrics × three existing encoder seeds: 24 small quantizer fits at one alignment arm/maturity. Reuse compatible global fits. Fit one diagonal transform per export on a common declared training population and reuse it across both population cells; separate transforms would confound population and metric. Keep the historical descriptor target scale primary.

The global sample already has equal sampled rows per source. Liquid conditioning can change those counts. Choose pooled-liquid or equal-source-liquid fitting explicitly rather than alleging an established global weighting bias. [Sampling][spatial-data].

Use the requested many-to-one best assignment for descriptor-group accuracy: multiple neural clusters may map to one descriptor group. Keep one-to-one historical scores distinct and preserve mapping-population and weighting definitions.

**Conditional remedy.** Retain a liquid partition only if held-out descriptor skill and assignment stability justify it. If continuous information survives but partitions do not, use continuous features. If both are weak, revisit representation/readout questions under H7/S5. Continuous readouts are comparators, not information-theoretic ceilings. No inferred cluster is automatically a precursor, and this structural stratum must not become a new birth-cohort filter.

<a id="s1"></a>

## S1 — Finite particles, mean supervision and support

**Observed.** Current dense value supervision averages four decoded structural particle means before squaring error. For conditionally independent means mu_i and fixed cached teacher target y_S,

$$
\mathbb E\!\left[\tfrac12\|\bar\mu_K-y_S\|^2\right]
=\tfrac12\|\mathbb E\mu-y_S\|^2
+\tfrac1{2K}\operatorname{tr}\operatorname{Cov}(\mu).
$$

Restore coordinate scaling on both terms. The same reasoning applies to averaged per-particle secants under existing common-noise coupling, conditional on their batch normalizer. This penalizes variation of decoded means, not emission variance directly. An outer expectation over independent unbiased teacher packets adds teacher uncertainty; a fixed reused packet does not itself perform that average. [Implemented objective][ni-dense].

**Hypothesis.** Mean-loss pressure could shift uncertainty from decoded means toward emission variance while endpoint NLL remains acceptable. With independent centered emissions, marginal variance is Var(mu_t)+E(sigma_t²), while cross-time covariance comes from decoded means. The code permits this mechanism. The completed first audit measures a small variance/K contribution at selected checkpoints—below 0.1% of the corresponding pooled value/response errors—but does not identify its causal influence during training. [Measured components](evidence.md#ni-methods-diagnostics).

**Proposed assay.** Run a frozen checkpoint once at Kmax, then use nested particle subsets and separate outer RNG keys. Changing K/batch shape with the same seed does not preserve paths. Use stable row/particle/time innovations or a single Kmax output; keep process and emission streams separate. Across architectures, identical seeds do not imply corresponding physical noise. Compare scores, mean Monte Carlo error, path dependence and variance decomposition. A finite mixture is a valid forecast; distinguish scoring that issued mixture from approximating an integrated latent law. Jensen effects concern the latter.

**Conditional remedy.** Retain the existing loss if its variance penalty is negligible or desired. Otherwise compare a mean-specific U-statistic at fixed K and other settings. For d_i = mu_i−y_S,

$$
U=\frac{\|\sum_i d_i\|^2-\sum_i\|d_i\|^2}{K(K-1)}.
$$

Apply the existing one-half and coordinate factors. U can be negative/noisy; clipping destroys unbiasedness. If the goal is predictive spread, use individual physical outcomes with a proper distributional/path score, preserving independent derivative validation.

The 24 raw radial/angular observables are nonnegative. Quantify de-standardized negative mass beyond numerical/practical tolerances; Gaussian tails alone do not establish failure. Material violations motivate a positive-support law or fixed transform with the correct density/Jacobian and zero handling. Do not clip/reject particles silently or impose positivity on embedding coordinates. [Observable definition][observables].

<a id="s2"></a>

## S2 — Distributional response

Physical packets retain per-shot Q/V and tangents; learned-feature caches average shots. Re-encode shot-level feature values/JVPs for a small fixed smooth-observable panel. Two shots cannot support a flexible 280-dimensional conditional covariance. [Archive][ni-simulator], [target cache][ni-targets], [dense aggregation][ni-dense].

For independent shots at the same complete initial state, write phi_i for an observable and r_i = D phi_i. Under valid differentiation through a fixed noise law,

$$
\widehat{D\operatorname{Var}(\phi)}
=\frac2n\sum_i\phi_i r_i
-\frac2{n(n-1)}\sum_{i\ne j}\phi_i r_j.
$$

At n=2 this reduces to (phi_1−phi_2)(r_1−r_2), with potentially very high variance. Pair each value with its own response; pooling states first changes the estimand. Naive products of shot means need finite-shot correction.

**Conditional remedy.** Use retained shots for reliability screening and request more independent futures only for an informative subset. Reproducible spread/outcome response may justify supplementing mean supervision with individual-outcome proper scores. Unperturbed-outcome fitting does not enforce its perturbation derivative: retain the response assay independently, and require matched perturbed-condition outcomes or valid derivative supervision before claiming repair. Stop rather than expand targets if the signal remains below measurement uncertainty.

<a id="s3"></a>

## S3 — Support, propagation and numerical coordinates

**Observed.** The short teacher follows original atom IDs; native sequences refresh nearest-80 support. Native NPT and fixed-cell BAOAB are different kernels. Float64 physical propagation with float32 forces/features gives a mixed precision contract. These differences are not established explanations of current errors. [Native producer][ni-data], [physical producer][ni-simulator].

**Proposed assay.** On retained full native frames, compare fixed-ID and refreshed-neighbor features with coordinates, center, box, encoder and units held fixed. Report turnover and its feature effect relative to natural change and numerical uncertainty. A smooth radial taper already exists. Cropped teacher paths cannot reveal exterior entrants; exact teacher-side support comparison needs retained full-cell frames or a small matched acquisition.

**Conditional remedy.** If consequential, match the observation process and target for a specific task, or retain separate tasks. For kernel differences, compare paired short continuations from the same complete state. Q/V is not a full restart including thermostat/barostat/RNG state. Map derivative coordinates and units explicitly when using external implementations; JAX MD's custom-JVP conventions illustrate why shapes alone are insufficient. [Literature](./literature.md).

Existing Ni neighbor rebuilding and overflow failures are safeguards to preserve, not missing checks to reintroduce. [Force implementation][ni-potential].

<a id="s4"></a>

## S4 — Closure and numerical timestep

Comparing a recurrence with its own repeated application is an identity. Its augmented state includes history and refresh phase; autonomous correlated emissions also need residual state.

**Proposed assay.** Compare a separately estimated direct-lag predictor/operator with short-lag composition, using compatible anchors, training-only transforms and paired source uncertainty. These estimates may share data. VAMP does not require equilibrium sampling, but that does not justify powers of one pooled cooling-data operator: initial-anchor matching does not establish intermediate-state compatibility. Protocol/time inhomogeneity, finite-feature projection, truncation and estimation can all mimic a memory failure. Without justified homogeneity, retain a direct-lag predictive comparison without a Chapman–Kolmogorov verdict. [Implementation and theory](./literature.md).

If a continuous-time claim matters, evaluate the same fitted drift/diffusion at 1/5/10 fs over common physical intervals, preserving the 100-fs observation refresh. Sum fine Brownian increments into coarse increments rather than merely resetting seeds. This tests discretization sensitivity, not finite-step equality. [Fine model][ni-fine].

**Conditional remedy.** A controlled memory failure supports the simple adequate state found by H3. Material timestep sensitivity supports describing the model as a fixed-step forecaster until a specifically justified continuous-time treatment succeeds. Do not add forbidden time/temperature covariates.

<a id="s5"></a>

## S5 — Maturity, fitting randomness and adaptation

**Observed.** Fresh frozen readouts already exist. The onset evaluator has differing sampling modes and couples head/sampler seeds; selection intervals and calibration are part of the fitted recipe. [Readouts][onset-eval], [workflow][mechanism-workflow].

**Proposed assay.** For one treatment, cross two retained maturities, three existing encoder seeds and three matched readout seeds: 18 frozen fits. Fix rows, sampling, update/selection budgets and calibration; record encoder, head and order identities separately. Three encoder seeds do not guarantee power.

**Conditional remedy.** If adaptation remains scientifically unresolved, compare frozen/fine-tuned likelihood fitting at two maturities with three matched seeds: at most 12 fits, reusing compatible baselines. A flat frozen maturity comparison does not exclude an adaptation interaction. Prefer frozen features if mature fine-tuning has no reproducible practical benefit; retain target-specific adaptation if supported. Source and seed uncertainty remain separate. Reused historical test sets are development evidence; confirmation needs a genuinely unused source holdout.

<a id="s6"></a>

## S6 — Symmetry and observation changes

**Observed.** Native MACE contains equivariant tensor channels but exports even scalars: global rotation/reflection invariance can retain angular structure while excluding pseudoscalar handedness. Full-cell periodic construction precedes its local graph. The historical spatial recipe includes signed chirality, independent reflections of views, and neighboring-center support selected within a nearest-128 parent pool. All four views enter each alpha arm; alpha changes their alignment, not batch-normalization exposure. [Native encoder][native-mace], [graph][native-graph], [spatial recipe][spatial-recipe], [data][spatial-data], [training][spatial-code].

**Proposed assay.** Read existing consistency receipts first. Remaining checks use proper rotations; target-appropriate reflections; atom permutations preserving center identity; equivalent periodic images through the full-cell producer; and support-changing re-extraction as a separate intervention. Rotate vectors/tangents and history consistently. A full-system rotation must transform the cell compatibly; a local rotation against fixed surroundings changes a physical boundary condition.

**Conditional remedy.** Equivalent-image failure calls for producer repair and re-versioning. Support-changing effects need explicit interpretation and, if justified, overlap constraints. If handedness matters, test a separate augmentation/export treatment. If symmetry passes but readouts fail, investigate descriptor collisions or body order. Translation of one patch is not selection of a different center; jitter, strain, relaxation, occlusion and reversal are not automatically target-preserving augmentations.

## Adopting a change

Use the [experiment gates](./experiment_plan.md), then implement the surviving change in existing producers/evaluators and metric contracts. Register one primary contrast and practical tolerance, report source/fit/numerical uncertainty separately, and require controls with demonstrated sensitivity. Non-significance is not equivalence. Publish one consolidated analysis with machine-readable tables, receipts, stopped branches and remaining questions through the [existing results system][results-system]. Preserve historical definitions and results; new code must not silently revise their meaning.

[agents]: ../../AGENTS.md
[datasets]: ../../DATASETS.md
[ni-model]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/model.py
[ni-train]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/train.py
[ni-eval]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/evaluate.py
[old-forecast]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/experiments/embedding_forecast_20260911/README.md
[observables]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/response_geometry/features.py
[ni-data]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/data.py
[dense-protocol]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/experiments/ni_latent_sim_20261010/DENSE_PATH.md
[ni-simulator]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_response/simulator.py
[ni-dense]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/dense_response.py
[embryo-code]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/continuous_motion/embryo_cohort.py
[embryo-metrics]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/docs/metrics/embryo_establishment_cohort.md
[appearance-eval]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/structural_response/appearance.py
[quality-code]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/encoder_quality/run.py
[mechanism-protocol]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/experiments/encoder_mechanisms_20260926/README.md#L121-L129
[native-pretrain]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/encoder_context/stream_pretrain.py
[spatial-results]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/experiments/spatial_vicreg_bias_20260929/RESULTS.md
[spatial-data]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/spatial_vicreg_bias/data.py
[ni-targets]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_response/targets.py
[ni-potential]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/simulation/eam_gpu/potential.py
[ni-fine]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/ni_latent_sim/fine_model.py
[onset-eval]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/supervised_onset/evaluate.py
[mechanism-workflow]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/encoder_mechanisms/workflow.py
[native-mace]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/models/encoders/spatial_mace.py
[native-graph]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/encoder_context/geometry.py
[spatial-recipe]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/configs/spatial_vicreg_bias/geoframe_plain.yaml
[spatial-code]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/src/research/spatial_vicreg_bias/train.py
[results-system]: https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/docs/research_results_system.md
