# Can a kinetic latent state retain liquid order and generate its evolution?

Protocol and literature selection, October 15, 2026. This is a new comparison
on existing native Ni trajectories. The 18-pipeline queue launched at 14:17 CEST
with a fixed deadline of October 16, 02:17 CEST. All 18 pipelines and independent
evaluations have now completed. The frozen protocol follows; a liquid-order
results summary appears at the end.

The previous 100-fs phase-space study found that smooth measurement-response
supervision helped nucleus readouts, but present-measurement decoding generalized
poorly from 320 training anchors. The next question is broader than nucleus AP:
does a representation retain continuous organization within liquid, and can a
probabilistic model propagate that organization toward local crystallization?
We separate this local generation task from prediction of genuine full-cell
nucleus events. [Previous findings](../ni_phase_measurements_20261014/README.md).

## Literature and method selection

The selected design combines a structural representation control with kinetic
learning and a common stochastic propagator. It is an adaptation of the cited
ideas, rather than a reproduction of any complete published pipeline.

| Related work | Relevant finding or construction | Decision for this study |
| --- | --- | --- |
| [Díaz Leines and Rogal, Ni reaction coordinates](https://arxiv.org/html/1810.04782) | In their Ni system, information about a prestructured region helps characterize the reaction coordinate beyond crystalline cluster size. | Measure orientational order and persistence separately from crystalline labels. Their potential and state conditions differ from ours; no published threshold is transferred. |
| [LeaPP](https://arxiv.org/html/2405.09642v2) | Bond-order descriptors enter a VAE; fixed-particle latent paths are compared with dynamic time warping and hierarchical clustering to analyze nucleation pathways. Examples include Ni3Al. | Use a VAE measurement representation as a structural control and preserve fixed central identities. We do not implement its DTW/path-clustering pipeline or retrospectively select only atoms that join a nucleus. |
| [GDyNet](https://www.nature.com/articles/s41467-019-10663-6) | Local graph representations trained with a dynamical objective distinguish atomic environments and their transitions. | Adopt the local dynamical-learning question, using existing measured observations instead of introducing another graph encoder in the first comparison. |
| [Qiu et al., short-term fluctuations in supercooled liquids](https://pmc.ncbi.nlm.nih.gov/articles/PMC12012455/) | Lagged descriptor correlations recover information about much slower structural relaxation in a Kob–Andersen glass former. | Motivation for measuring liquid-order persistence even when nucleus prediction is weak. This concerns glassy relaxation, not demonstrated Ni nucleation or transferable lag scales. |
| [VAMP theory](https://arxiv.org/abs/1707.04659) and [VAMPnets](https://www.nature.com/articles/s41467-017-02388-1) | Singular components of lagged evolution motivate whitening and a variational dynamical score. The theory includes nonreversible and nonstationary settings. | Compare linear and nonlinear nonreversible VAMP-style representations. A high score does not establish a closed, homogeneous Markov model for an observed local neighborhood. |
| [Molecular latent space simulators](https://pmc.ncbi.nlm.nih.gov/articles/PMC8162036/) | The original construction combines an SRV encoder, a mixture-density latent propagator, and a conditional Wasserstein GAN configuration decoder. | Retain the separation of representation, propagation and decoding. Here the encoder is nonreversible VAMP or a matched control, the propagator uses four observations, and the decoder predicts measurements deterministically. We do not reproduce the SRV/cWGAN method or claim its atomistic generation guarantees. |
| [Time-lagged autoencoders](https://arxiv.org/abs/1710.11239) and [their limitations](https://arxiv.org/abs/1906.00325) | Lagged reconstruction can extract slow information, but nonlinear objectives can mix variance and slowness. | Use explicit static and kinetic objectives with a hybrid comparison, rather than treating future-reconstruction accuracy as proof of slow-mode recovery. A pure TAE is deferred. |
| [SRV](https://arxiv.org/abs/1902.03336) and [Deep-TICA](https://arxiv.org/abs/2107.03943) | Continuous slow coordinates provide an alternative to discrete states; Deep-TICA includes a crystallization application. | Defer reversible/equilibrium interpretations. Our collection combines different source conditions, growth and outcome-enriched windows. Linear nonreversible VAMP supplies a more appropriate inexpensive kinetic control. |
| [State predictive information bottleneck](https://arxiv.org/abs/2011.10127) | Predictive compression learns representations connected to delayed state assignments. | Defer until meaningful state assignments are established. Premature phase discretization could discard the liquid distinctions under investigation. |
| [Generator EDMD](https://arxiv.org/abs/1909.10638) | Generator approximation can use dynamical vector fields and derivatives of features. | A candidate derivative-based successor, deferred here because the existing short fixed-box NVE tangents do not provide the generator of our partially observed native NPT process. |
| [Implicit Transfer Operator learning](https://arxiv.org/abs/2305.18046) | Diffusion-based conditional generation models transitions at multiple time resolutions. | A possible later propagator replacement after establishing representation quality and a stable native-cadence baseline. The present mixture-density comparison does not implement diffusion or multiple-lag generation. |
| [Timewarp](https://arxiv.org/abs/2302.01170) | A conditional normalizing flow supplies time-coarsened proposals, with Metropolis-Hastings correction for Boltzmann sampling. | Defer: independent local measurement vectors lack a full-cell energy and proposal density suitable for that correction. Correct equilibrium sampling would also not establish correct nucleation kinetics. |

These choices leave open whether apparently ordered liquid is productive,
frustrating or transient. High q6 alone is not a phase or a validated precursor.
Existing reviews provide wider context and the history of earlier kinetic
experiments: [representation review](../../docs/encoder_research/representation_literature_20260926.md)
and [nucleation evidence](../../docs/encoder_research/nucleation_literature_and_open_questions.md).
Linear VAMP has already been tried elsewhere in this repository; this is a new
matched Ni comparison, not a claim that kinetic objectives have never been tested.

## Observations and time notation

Let i identify an input snapshot in a source trajectory, and let t_i be its saved
physical time. Let j identify the central atom. The local state H_ij(t_i+tau)
contains positions and velocities around that same atom, with the nearest 80
neighbors selected at each observed time. The elapsed time tau is measured from
the snapshot. Neither t_i nor tau is an explicit encoder or propagator input.

Write Y_ij(t_i+tau) for the 244 structural and velocity measurements of that local
state: BO40, TDA113, smooth density74 and velocity17. With native spacing
Delta = 0.1 ps and integer observation index k, define

\[
Y_{ij,k}=Y_{ij}(t_i+k\Delta),\qquad k=-3,-2,\ldots,60.
\]

The training-only transformation A first standardizes each measurement, applies
asinh, and standardizes again. Its mean and scale are fitted on all native
training observations only; both scale floors are 1e-4. Define

\[
u_{ij,k}=A(Y_{ij,k}),\qquad z_{ij,k}=E_\theta(u_{ij,k})\in\mathbb R^{32}.
\]

Here u is a normalized measurement vector, z is the learned latent state, and
E_theta is the encoder. The decoder D_psi maps z back to normalized measurements.
The inverse transformation A^{-1} returns the original measurement units.
The current encoder receives one 244-vector; the propagator receives the four
most recent latent states, spanning 0.3 ps. Actual velocities contribute through
the 17 measured velocity features, not as complete per-atom velocity arrays.

For generated continuation b, draw the next latent increment from an
eight-component Gaussian mixture conditioned on that four-frame history:

\[
\delta z_{ibj,k}\sim
\sum_{m=1}^{8}\pi_m(z_{ibj,k-3:k})
\mathcal N\!\left(\mu_m(z_{ibj,k-3:k}),
\operatorname{diag}(\sigma_m^2(z_{ibj,k-3:k}))\right),
\]
\[
z_{ibj,k+1}=z_{ibj,k}+\delta z_{ibj,k},\qquad
\widehat Y_{ibj,k+1}=A^{-1}(D_\psi(z_{ibj,k+1})).
\]

The mixture index m selects one component, pi_m are probabilities summing to
one, mu_m are increment means, and sigma_m are positive coordinate-wise standard
deviations. Diag forms the component covariance matrix. The implementation fits
and samples these increments in training-standardized latent units, then maps
them back. A component has diagonal covariance; the full mixture can represent
several correlated alternatives. The observed native continuation has only one
saved physical realization per window. The 32 generated values of b are neural
samples, not independently simulated MD continuations.

## Frozen six-method comparison

| Method | Representation objective | Purpose |
| --- | --- | --- |
| PCA | Leading 32 directions of training measurement covariance | Static linear compression control |
| Linear VAMP | Two-sided covariance whitening and singular vectors from pooled 0.5/1 ps pairs; the left map encodes every observation | Linear kinetic control |
| VAE | Present measurement reconstruction plus KL weight 0.001; posterior mean exported | Nonlinear structural control inspired by descriptor-based pathway studies |
| VAMP | Negative regularized VAMP-2 score, alternating 0.5 and 1 ps pairs | Continuous nonlinear kinetic representation |
| Hybrid | VAMP objective plus equally weighted present reconstruction objective | Test whether kinetic learning needs explicit retention of measured structure |
| Shuffle | Same VAMP objective, replacing future pairs by random windows/times within the same source | Control for learning source-specific distributions rather than local temporal correspondence |

The VAMP score uses centered latent covariances C00, C11 and cross-covariance
C01. If L0 and L1 are Cholesky factors of C00+epsilon I and C11+epsilon I,
the optimized score is ||L0^{-1} C01 L1^{-T}||_F^2 / 32, with epsilon=0.001.
Covariance arithmetic is float64. No reversibility constraint or soft phase
partition is imposed. The linear control uses the existing estimator's own
recorded covariance-scaled ridge convention; it is not an identical nonlinear
objective restricted to a linear network.

Seeds are 20261025, 20261026 and 20261027, giving 18 complete pipelines. Each
nonlinear representation trains for 12 full epochs. Linear representations use
their deterministic estimators. Next, freeze every encoder, initialize a fresh
matched measurement decoder, and train that decoder for 12 epochs. Finally,
freeze representation and decoder and train the common mixture-density
propagator for 12 epochs by conditional negative log likelihood. Thus downstream
decoder differences are not caused by unequal warm starts. Each MLP has two
256-wide SiLU hidden layers; all latent states have 32 coordinates.

The full batch and microbatch are both 1,024, an explicit deviation from the
spatial default for these small dense networks. AdamW uses maximum LR 0.001,
one warmup epoch and cosine decay per stage, weight decay 1e-5 and gradient clip
10. Representation/decoder epochs cover 195,840 training observations at window
offsets 0…5 ps: 2,304 updates per stage. Propagator epochs cover all 230,400 native
future transitions: 2,700 updates. A nonlinear pipeline therefore has 7,308
updates; a linear pipeline has 5,004 trained downstream updates. Select only
the final fixed-budget checkpoint. No test selection, AP loss or early stopping.

## Data and separate tasks

Use all 14,400 existing windows from 60 independent Ni source trajectories.
The preserved training ancestry is 16 sources / 3,840 windows; all former
nontraining roles become 44 test sources / 10,560 windows. Each window follows
one central atom for the complete 0.3 ps history and 6 ps future. The collection
contains 4,800 original, 4,800 liquid-persistent and 4,800 growing-front windows.
The last two are outcome-enriched; their frequencies are not natural incidence.
Source conditions and crystallization progress differ, and the local observation
omits the cell, thermostat and external environment. Consequently a learned
transition law is a conditional surrogate of this collection, not an established
equilibrium or homogeneous physical operator.

**Genuine nucleus prediction.** Reuse the unchanged 90-source full-cell Ni birth
catalogue and its 19,800 assay rows at exact 1/3/6 ps leads. Targets are 32-atom
and default 64-atom nuclei with 1.5 ps persistence. Declared ancestry is 28 train
and 62 test sources; actual assay rows have events in 21 train and 42 test
ancestors. No neighborhood filtering is added. The encoder sees present Y only,
whose support agrees across cohorts. Frozen likelihood-trained CatBoost probes
compare latent32, decoded244, actual present measurements plus latent, and
actual four-frame measurement history plus latent. The cached birth history
holds present neighbor identities fixed backward; it appears only as an explicit
observed-history readout control, not as a mismatched propagator input.

**Liquid information retention.** Evaluate all test-window present observations
with source-balanced reconstruction errors, within-source rank correlation and
R2 for qbar6, orientational coherence, qbar4 and wbar6. Report the complete cohort
and a diagnostic stratum whose present center is not PTM crystalline and whose
local crystalline fraction is at most 0.05. This stratum never changes the birth
assay or representation training. Include latent variance/effective rank and a
frozen training-only logistic phase readout to separate elementary phase
recognition from the harder liquid-order question.

**Generated local ordering.** Generate 32 autonomous paths through 6 ps from
four true past observations; future measurements never enter the rollout.
Use a hash-fixed panel of up to four test windows per ancestor/regime. Compare
path energy, increments, increment correlations, 1/3/6 ps mean error, interval
coverage, ensemble variation and temporal variation. Baselines are persistence,
recursive conditional-mean propagation, and an explicitly labeled oracle that
encodes and decodes actual future MD frames. The oracle diagnoses representation
and decoder limitations; it is not a forecast.

Local crystalline entry is the completion of three consecutive positive future
classifications at 0.1 ps spacing: a 0.2 ps first-to-last observation span.
Ground truth uses exact full-cell PTM labels of the fixed central atom. Generated
paths use the frozen latent phase probe, so its measured error limits this
diagnostic. Report entry probabilities and Brier scores, with initial
noncrystalline and present-liquid strata. Separately track continuous order
lag products and a high-order diagnostic exceeding the training-liquid 80th
percentiles of both qbar6 and coherence while noncrystalline. Report occupancy
and persistence rather than declaring that threshold a new physical phase.

Full-cell birth labels and independently generated local paths are different
objects. A local trajectory cannot establish spatial connectivity among 32 or
64 jointly evolving atoms. This study can show local ordering and useful
nucleus prediction; it cannot claim generated whole-cell nucleation, a critical
nucleus, a committor, a natural nucleation rate or a free-energy landscape.

## Assessment and limits

All transforms and diagnostic heads are fitted on train only. Report proper
scores, AP/AUROC as diagnostics, source-balanced results, paired source bootstrap
intervals and seed variability. The frozen primary contrasts are hybrid versus
VAE, hybrid versus VAMP, VAMP versus linear VAMP, VAMP versus shuffle, and hybrid
versus PCA. Source-paired contrasts average matched seeds before bootstrapping
independent ancestors; seed variability is reported separately. Missing or
undefined sources and incomplete seed sets remain visible rather than creating
a favorable complete-case comparison. These reused test sources provide development evidence, not a new
untouched confirmation cohort.

Useful evidence requires both accessible liquid distinctions and realistic
temporal behavior. A visually separated liquid/crystal projection, low training
loss or large generated crystal fraction alone is insufficient. Decoder error,
phase-probe error, future-selected sampling, approximate finite memory and
stochastic rollout excursions remain separately visible. The deterministic
measurement decoder can still average unresolved observations; mixture-density
propagation is not a guarantee against this. No atomic configuration or joint
physical consistency is supplied by the decoded measurements.

The queue has a 12-hour absolute cap, four GPU workers, independent CPU
annotation, checkpoint recovery and independent evaluations. Scientific runs
remain online in the existing W&B project. No new MD or relaxation is scheduled,
and no short NVE tangent is treated as long-time NPT ground truth.

[Frozen recipe](../../configs/ni_latent_sim/kinetic_latent_20261015.json) ·
[Workflow and data provenance](../../docs/ni_kinetic_latent.md) ·
[Metric definitions](../../docs/metrics/ni_kinetic_latent.md) ·
[Result folder](../../output/ni_kinetic_latent/native-20261015-v1/)

## Liquid-order results, October 15

The original precursor question has partial support for information retention,
but no demonstrated faithful generation of productive liquid precursors.
On 5,937 present-liquid test observations from 44 ancestors, the fresh decoder
recovers these continuous measurements from the 32-dimensional representation:

| Representation | qbar6 R² | Coherence R² | wbar6 R² |
| --- | ---: | ---: | ---: |
| PCA | 0.617 | 0.696 | 0.032 |
| Linear VAMP | 0.827 | 0.929 | 0.885 |
| VAE | 0.701 | 0.783 | 0.938 |
| Nonlinear VAMP | 0.381 | 0.330 | 0.017 |
| Hybrid | 0.746 | 0.791 | 0.823 |

These are equal-source means of within-source R², averaged over three seeds.
The shuffled control has severe raw-decoder failures and is not a useful
positive control. Hybrid improves qbar6 relative to VAE by 0.0453, paired
whole-source 95% interval [0.0330, 0.0566], but worsens wbar6 by 0.1155.
Linear VAMP is a strong control for retaining orientational organization;
the nonlinear kinetic objective alone does not improve that retention.

These measurements are also part of the encoder inputs. Reconstruction tests
whether compression preserves continuous distinctions within liquid; it does
not discover a new precursor or establish its relation to subsequent birth.
The liquid diagnostic constrains local80, not a whole surrounding region.

Autonomous generation still distorts liquid ordering. In the 303 initially
liquid forecast windows, observed high-order-bin occupancy over 6 ps is 14.8%,
compared with 21.8% for linear VAMP, 28.8% for VAE and 40.7% for PCA. The bin
requires qbar6 and coherence above their training-liquid 80th percentiles while
noncrystalline; it is not a validated phase. These are three-seed results.
Hybrid's third seed has undefined raw metrics; no two-seed survivor mean is
reported as its full result. Nonlinear VAMP and shuffle also have invalid raw
temporal metrics.

Only one actual local crystalline entry occurs in this strict liquid forecast
panel, from one source. This prevents a convincing demonstration of productive
liquid-to-crystal generation. The separate birth assay has many more real events,
but its predictive scores do not isolate a wholly liquid precursor mechanism.
The remaining scientific question is whether latent distinctions before birth
persist and predict which locally liquid environments progress, after accounting
for present order and existing outer crystals. That requires a separate declared
birth-aligned diagnostic; it must not change the full birth catalogue or turn
the ordering threshold into a precursor label by assumption.

Evidence: existing [comparison tables](/store/PERSO/vmorozov/experiments/ni_kinetic_latent/native-20261015-v1/analyses/comparison/tables),
particularly `retention_individual.csv`, `retention_paired.csv` and
`trajectory_individual.csv`. This summary adds no fits or exclusions.
