# Ni simulator response experiments and findings for the next chat

Prepared 12 October 2026. Results cover the two methods-diagnostic queues completed on 11 October, at 22:21:36 and 23:10:39 Europe/Paris. This is a handoff of this chat's completed audits, not a status report on every parallel research branch.

**Main finding:** the audited transformers produce stable forecasts through 6 ps, but simulator-derivative supervision has no consistent predictive advantage across two training seeds. The response-readout experiment also exposed a limitation of the diagnostic itself: it struggles to reconstruct a present response already available in its inputs. Its weak future scores therefore cannot establish that the observations lack physical information or that compression is the fundamental failure.

The research objective remains to make motion learning useful for crystallization prediction. These experiments evaluate dynamics and physical sensitivity; they do not supply new nucleus-birth AP results.

## What was implemented and completed

| Experiment | Question | Completed work |
|---|---|---|
| Initial path audit | Do sampled futures have useful temporal dependence, and do long rollouts remain stable? | Three existing MLP models, each with three sampling keys, autonomous forecasts through 6 ps, endpoint and whole-path scores, temporal scrambling, retained trajectories |
| Initial physical-response audit | Is response prediction better than zero, and are noise or finite-particle loss artifacts dominant? | Existing simulator targets re-encoded for 60 anchors, shot diagnostics, response secants, mean-loss decomposition |
| History audit | Does a longer observed past help, and does the compact representation retain that information? | 28 frozen Gaussian ridge readouts using 1, 4, 8 or 14 observations, with order controls |
| Transformer follow-up | Does training on 6-ps futures improve stability and prediction? Does derivative supervision help? | Four 6-ps models crossing values/response with two seeds, plus one matched 1-ps values control; 15 model/sampling-key tables |
| Response-predictability follow-up | Can the full observation or compact code predict the existing physical derivatives? | Nine representations × two perturbation families × two readout types = 36 selected frozen readouts |

All stages completed on their first attempts. These queues evaluated existing neural checkpoints and fitted local diagnostic readouts. They did not retrain eSEN, generate new physical trajectories, or perform new relaxation. The proposed next experiments below have not been launched by these queues.

## Data and time conventions

The sealed Ni motion collection contains **14,400 local sequences**: 3,840 fitting, 960 selection and 9,600 held-out rows, from 16, 4 and 40 ancestors respectively. Each local observation contains 80 atoms around the same central atom identity. There are 4,800 sequences from each of three regimes: original, liquid-persistent and growing-front. Because some regimes were selected using future behavior, this mixture does not estimate natural crystallization prevalence.

The path audit uses the same **214 held-out sequences from all 40 held-out ancestors** in both queues. It takes two deterministic rows per available ancestor/regime stratum. Thirteen ancestors lack growing-front rows; they remain represented in the other regimes. This is not the earlier four-source test-only evaluation.

At forecast origin \(t_0\), the ordinary model observes

\[
t_0-0.3\,\mathrm{ps},\quad t_0-0.2\,\mathrm{ps},\quad
t_0-0.1\,\mathrm{ps},\quad t_0.
\]

Native future targets are at \(t_0+\tau\), with \(\tau=0.1,0.2,\ldots,6\) ps. The learned simulator advances internally every **5 fs**, without future observations or physical-force corrections during forecasting. The separate physical teacher integrates every **1 fs**, saving responses at \(\tau=5,10,\ldots,100\) fs.

**Six-ps training refers to native future-value supervision. Physical derivative supervision still ends at 100 fs, which is 0.1 ps.** A fine neural integration step does not create additional measured ground truth between native snapshots.

The observation has 280 features: 128 frozen unrelaxed eSEN geometry features, 128 velocity-derived geometry secants, and 24 smooth radial/angular descriptors. The learned bottleneck exports 16 features. “Raw280” means this full encoded observation, not atomic coordinates, velocities or the full MD state. Train-only normalization is reused unchanged. Temperature, simulation age, absolute time, box and species covariates are absent from the predictor.

Native histories and response teachers also differ physically: native NPT observations refresh the nearest 80 atoms at each saved time; the fixed-box BAOAB teacher follows the initial 80 atom identities while evolving the full cell. These dynamics and observation definitions have not been made identical by this audit. Physical integration uses float64 states/tangents/noise with float32 GPU force arithmetic; the frozen eSEN targets are float32.

## How forecasts were scored

Each path is generated with 128 particles. Scores for 4, 8, 16, 32 and 64 particles reuse prefixes of that same simulation. Three independent sampling keys assess Monte Carlo variation; they are not additional training seeds or independent source histories.

Headline numbers below give equal weight to each ancestor and average the three sampling keys. Endpoint mean MSE measures error in the mean prediction of the 24 standardized descriptors. Whole-path energy scores the distribution of emitted descriptor trajectories over all 60 future times. Both are lower-is-better. Neither is atom-coordinate error, crystallinity classification accuracy or nucleus-birth AP.

The energy score uses all distinct particle pairs. For predicted vectors \(X_1,\ldots,X_K\), observed vector \(y\), and vector dimension \(D\), it is

\[
\frac{1}{K}\sum_i\frac{\lVert X_i-y\rVert}{\sqrt D}
-\frac{1}{2K(K-1)}\sum_{i\ne j}\frac{\lVert X_i-X_j\rVert}{\sqrt D}.
\]

The temporal control independently permutes particle identities at each future time. This preserves the exact endpoint sample sets while breaking which predictions belong to the same trajectory. Its reported gap is **scrambled score minus connected score**; positive favors the original connected paths. All primary scores retain every row and particle.

## Initial audit results

These three MLP models trained on native futures through 1 ps, with a 3,200-update budget, seed 20261011, and their original validation-NLL-selected checkpoints.

| Training treatment | Whole-path energy through 6 ps | Descriptor mean MSE at 1 ps | At 6 ps |
|---|---:|---:|---:|
| Endpoint response | 0.5937 | 0.7362 | 0.7479 |
| Path values | 0.6009 | 0.7292 | 0.8386 |
| Path response | 0.6279 | 0.7364 | 20.9851 |

Persistence has whole-path energy 1.1252. The path-response minus path-values difference is +0.02696, with a paired ancestor-bootstrap interval [0.01782, 0.03749], conditional on these checkpoints and this selected population.

The path-response model develops rare but severe excursions beyond its training horizon. Through 6 ps, **6.34%** of particles have a conditional-mean descriptor above 10 in absolute standardized units; **1.04%** exceed 100. None exceeds 10 through 3 ps. The favorable apparent temporal-scramble gap (+0.04862) is concentrated in those unstable rows. A post-hoc stable-row diagnostic gives −0.00152; it diagnoses the artifact and does not replace the primary score.

On 40 held-out physical anchors, the path-response model's derivative MSE is **1.069 times zero-response MSE for position perturbations and 1.011 for velocity perturbations**. Values-only gives 1.066 and 1.020. These are direct student response evaluations, distinct from the later frozen readouts.

The two-shot estimate of teacher-mean noise is about 3–5% of teacher power over the window, although two shots give only a coarse noise estimate. At the selected path-response checkpoint, the finite-particle variance penalty at K=4 is only **0.095% of value squared-mean loss and 0.030% of position-response loss**. Smaller secant amplitudes make small changes. This does not support prioritizing that estimator correction as the dominant remedy, while leaving its earlier training effects unresolved.

The history audit moves to a common origin 1 ps later so every compared history has the same forecast targets. Raw280 ordered-history MSE at a 1-ps future falls from **0.7113 with one observation to 0.6761 with fourteen**, a 4.9% improvement. Compact16 ordered readouts work best around four observations; an unordered summary of fourteen reaches about 0.6719. Reversing past slots has little effect in these fitted linear readouts. A useful past summary is supported; a general claim that temporal order is irrelevant is not.

## Transformer follow-up results

The five selected transformers share the same 214-row panel. Their original training budgets were 4,000 updates at batch 256. The architecture uses two transformer blocks over four ordered latent states, width 64 and four attention heads, with the same 16-dimensional compression and observation decoder.

The values-only 1-ps and 6-ps models at seed 20261014 are the matched horizon comparison; both selected update 1,000. The 6-ps response model at seed 20261013 selected update 800; the other selected checkpoints are at 1,000. Original validation likelihood determines selection, not these audit scores or AP.

| Native training horizon and treatment | Training seed | Whole-path energy | Descriptor mean MSE at 6 ps |
|---|---:|---:|---:|
| 1 ps, values | 20261014 | 0.591907 | 0.699656 |
| 6 ps, values | 20261013 | 0.582865 | 0.675267 |
| 6 ps, values + response | 20261013 | 0.587384 | 0.681563 |
| 6 ps, values | 20261014 | 0.586558 | 0.685096 |
| 6 ps, values + response | 20261014 | 0.585155 | 0.676326 |

All five produce finite forecasts without conditional-mean particles crossing the predeclared magnitude threshold of 10 through 6 ps. **The 1-ps transformer is also stable.** Consequently, stability relative to the earlier MLP response model cannot be credited solely to the longer training horizon or the transformer architecture; that cross-study comparison changes several factors.

In the matched values-only comparison, 6-ps training reduces 6-ps mean MSE by **2.08%** and whole-path energy by **0.90%**. This is one training seed. Adding physical-response supervision worsens both scores at seed 20261013 and improves both at seed 20261014. There is no consistent treatment advantage in these two seeds.

Whole-path scramble gaps for 6-ps values/response are **+0.000164 / −0.001264** at seed 20261013 and **−0.000218 / −0.000236** at seed 20261014. The response models do not benefit from their connected trajectories on this score. Future-increment scores do favor connected paths slightly, so the conclusion depends on the path statistic; this is not evidence of complete temporal independence.

The Gaussian decoder still emits negative values for **2.53–2.61%** of descriptor coordinates in the 6-ps models, although these descriptors are nonnegative by construction. Numerical stability therefore does not establish a physically valid output distribution.

## Response readouts and the failed present response control

The follow-up uses all **300 existing physical anchors**: 80 fitting, 20 selection and 200 test, from 16, 4 and 40 ancestors. Each anchor has two continuation shots; those shots do not increase the independent source count.

The nine representations are Raw280, five transformer compact codes and three earlier MLP compact codes. Each receives separate position (Q) and velocity (V) perturbation queries. One readout is directional linear ridge. The other is nonlinear in the observed history through a state kernel, while remaining exactly linear in the perturbation direction. Thus the comparison is not restricted to a linear readout of state. Regularization and kernel bandwidth are selected by held-out selection likelihood.

The target is the saved two-shot mean directional derivative of the 128 geometry features at each future offset. For this report, relative MSE is the mean squared prediction error divided by the mean squared response, averaging separately over held-out ancestors and the twenty 5–100-fs offsets. A value of 1 equals predicting zero everywhere; it is not the average of per-row error ratios.

| State-conditioned readout input | Q relative MSE | V relative MSE |
|---|---:|---:|
| Full 280-feature observation | 0.986273 | 0.971912 |
| 6-ps response code16, seed 20261013 | 0.999057 | 0.998935 |
| 6-ps response code16, seed 20261014 | 0.999573 | 0.997594 |

The full observation gives approximately **1.4% and 2.8% reductions** versus zero; these compact response codes give only **0.04–0.24%**. However, the present-time control makes a stronger causal interpretation unsafe:

| Position-response control | At the present | At 5 fs | At 100 fs |
|---|---:|---:|---:|
| Copy the known present geometry secant | 0.000006 | 0.0234 | 2.77 |
| Learned full-observation state-conditioned readout | about 0.963 | 0.9615 | 1.0029 |

These are also errors relative to zero response. At 5 fs, simply carrying forward the measured present secant removes about 97.7% of zero-baseline error, while the learned readout removes only about 3.8%. The observation demonstrably contains useful short-time response information. The readout, fitted on only 80 anchors, largely fails to recover even this elementary relation.

**Revised interpretation:** the readout scores describe this fitting procedure; they do not establish an information-theoretic limit, a fundamental compression obstruction, or impossibility of motion learning. Conversely, copying the present response becomes poor at 100 fs, so it is not a complete dynamics model.

## Next experiments and their decision criteria

These are proposals for continuation, not submitted jobs.

1. **Predict response change around a known present response.** Define \(R_\tau\) as the physical future geometry response to a specified perturbation. Use the measured present secant \(\widetilde R_0\) as a fixed term and fit
   \[
   \widehat R_\tau=\widetilde R_0+f_\theta(H,d)_\tau,
   \qquad f_\theta\text{ targets }R_\tau-\widetilde R_0.
   \]
   Here \(H\) is observed history and \(d\) is the perturbation query. Reuse the current archived teacher targets and source splits. Compare zero, present-response persistence, the original readouts and this residual readout at each physical horizon. The baseline term must come from the observation, not a future teacher value. If both Raw280 and compact16 receive the same present-response term, explicitly label the comparison as predicting additional response change given that term; it no longer tests compact16 alone.
2. **Measure fitting-data dependence before collecting more trajectories.** Use nested subsets of the existing 16 fitting ancestors, preserving all selection/test ancestors. For example, use 4, 8, 12 and 16 training ancestors, repeating ancestry subsets where feasible. Plot held-out predictive error and the present-response control, with uncertainty across ancestors. The current maximum remains 80 fitting anchors; do not silently use test data to extend the curve. Acquire more physical teachers only if these controls indicate a useful data-limited regime.
3. **Require a successful sensitivity assay before expanding derivative training.** If the revised readout beats both zero and present-response persistence at future offsets beyond the trivial near-present regime, compare values-only and derivative-supervised training with matched architecture, horizon, source roles and multiple seeds. Retain proper predictive scores, calibration, whole-path controls and stability diagnostics. Use nucleus-birth AP later as an evaluation diagnostic, never as a loss or checkpoint selector.

Kernel/support matching, response-direction coverage, population sampling and relaxed/unrelaxed domain comparisons remain separate unresolved questions. Only two query directions per anchor and two shots cannot characterize the full Jacobian or conditional future distribution. No failure in this audit proves the overall research direction impossible.

## Evidence and continuation references

Repository: `/home/infres/vmorozov/PointCloudMaterials`. Run paths below are relative to that repository; output symlinks resolve to `/store/PERSO/vmorozov/experiments/ni_latent_sim/`.

| Reference | Location |
|---|---|
| First audit and plots | [methods-diagnostics-20261011-v1](../../output/ni_latent_sim/methods-diagnostics-20261011-v1/README.md) |
| Follow-up and plots | [methods-followup-20261011-v1](../../output/ni_latent_sim/methods-followup-20261011-v1/README.md) |
| First recipe | [methods_diagnostics_20261011.json](../../configs/ni_latent_sim/methods_diagnostics_20261011.json) |
| Follow-up recipe | [methods_followup_20261011.json](../../configs/ni_latent_sim/methods_followup_20261011.json) |
| Exact metrics | [ni_motion_diagnostics.md](../../docs/metrics/ni_motion_diagnostics.md) and each exported table's frozen `METRICS.md` |
| Scientific protocol | [METHODS_DIAGNOSTICS.md](METHODS_DIAGNOSTICS.md) |
| Execution workflow | [Ni motion diagnostics](../../docs/ni_motion_diagnostics.md) |

Each run has `technical/plan.json` with row IDs, ancestry, checkpoints, normalization, time grids and kernel contracts; `technical/launch.json` and completion receipts; and named `analyses/` folders with scores, predictions and implementation hashes. The first audit's internal plan identity is `1e45d0a71b39ec1678e0701b14e464a1ff06cd50f8eb01531868be5ebeb7a901`; the follow-up's is `f00c4d796789cd82a3803a625c47936800d85ae81346c42986ea947a1736a7d0`. These identity fields are distinct from file SHA256 values.

The first jobs were 1034535–1034537. The follow-up used GPU array 1034576 and CPU job 1034577; all are complete. Any later jobs or results from other chats require their own receipts and must not be attributed to these queues. The shared workspace has uncommitted work from multiple chats; preserve the existing artifacts and use the exported implementation hashes when reproducing historical calculations.

For new work, retain `pointnet-torch214`, original ancestry roles, exact observation offsets, label-free/predictive-likelihood selection and the repository input restrictions. Frozen diagnostic readouts remain local; scientific training uses the existing online W&B project. Read `AGENTS.md` before implementation. This handoff does not authorize stopping or changing jobs belonging to another chat.
