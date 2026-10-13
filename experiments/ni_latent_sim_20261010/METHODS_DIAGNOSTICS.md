# October 11: what does the learned simulator actually predict?

[October 12 handoff for another chat](METHODS_HANDOFF_20261012.md) combines both
completed audit queues, their limitations and the proposed next experiments.

**Completed:** E0 manifest and bounded E2/E3/E4 analyses; all three job receipts
report success. Read the maintained [findings](../../docs/encoder_research/evidence.md#ni-methods-diagnostics)
and [stage coverage](../../docs/encoder_research/experiment_plan.md#execution-status).
This completion does not cover E1, physical finite differences, full support/kernel
matching, rank-sufficient response directions, or a newly trained remedy.

The first methods stage audits completed, matched 3,200-update MLP simulators:
endpoint response, dense-path values, dense-path response; seed 20261011 and
their original validation-NLL selected checkpoints. It changes neither the
trained models nor their historical metrics.

Questions:

1. Do connected sampled futures predict native cross-time dependence, beyond
   getting endpoint marginals right?
2. How much response signal survives physical-shot noise, observation encoding
   and student secant size? How much finite-particle mean loss penalizes spread?
3. Does extending the observed history help at a common origin and future window,
   and does compact16 retain the information present in raw280 observations?

E0 pins all row/checkpoint/normalization identities and kernel/time conventions.
E2 uses 214 held-out rows across all 40 ancestors, two rows per available
ancestor/regime stratum; absent front strata remain explicit. Three independent
outer keys generate K=128 once, retaining K=4/8/16/32/64/128 prefixes. Joint paths,
future-to-future increments, lag products, excursions and dwell descriptors are
scored alongside endpoints, spread allocation and support violations. A temporal
particle scramble preserves endpoint multisets. A synthetic positive control
must detect the scramble before interpreting a null result in the real panel.

E4 reuses the physical archives, retaining both per-shot values and analytic
responses from one anchor per each of 60 ancestors. It does not launch new MD.
Held-out secants compare epsilon=.5/.25/.125 at common process noise. Mean loss
is decomposed into its U estimate and the variance/K term; training denominator
variation is audited separately. Physical AD versus force finite differences and
a general response-rank obstruction remain unanswered by this two-direction bank.

E3 uses all 14,400 cached rows and original ancestor roles. Move the common
forecast origin forward 1 ps, making 1/4/8/14 exact observations available, with
.1/.5/1/3 ps targets for every history. Frozen Gaussian ridge readouts compare
raw280 with each compact16 code, ordered versus order-restricted histories, and
reversed past at fixed fitted weights. Ridge strength is chosen by selection
NLL, with no AP training or model choice.

Interpretation is conditional on this regime-selected population. Report all
three outer keys and row/family/ancestor/ancestor-family weightings. Path scores
use emitted samples; conditional-mean scores are separately labeled diagnostics.
A weak scramble effect can be inconclusive. The first response remedy, if its
penalty is material, is a matched finite-particle mean-loss comparison retaining
the original control. New physical acquisitions require a question not resolved
by the current bank. E1 and E5–E9 followups are recorded, not launched en masse.

Configuration: `configs/ni_latent_sim/methods_diagnostics_20261011.json`.
Reproduction: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m
src.research.ni_latent_sim.diagnostics submit --config
configs/ni_latent_sim/methods_diagnostics_20261011.json`.

[Execution](../../docs/ni_motion_diagnostics.md) ·
[Metric definitions](../../docs/metrics/ni_motion_diagnostics.md).

## Completed first-stage result

All stages finished on their first attempts on October 11, last completion
22:21:36 Europe/Paris. At K=128 and equal-ancestor weighting, averaging three
sampling keys, full-path energy is 0.5937 / 0.6009 / 0.6279 for endpoint response /
path values / path response. Path-response descriptor mean MSE rises to 20.985 at
6 ps, versus 0.839 for path values. These models trained on native futures only
through 1 ps; this is a long-extrapolation failure, not an audit of the later
6-ps-trained models. Apparent path-response scrambling benefits are concentrated
in rows with runaway particles.

Held-out path-response derivative MSE is 1.069 times the zero-response baseline
for Q directions and 1.011 for V. The finite-particle mean-loss penalty is small
at the saved checkpoints. Raw280 ordered-history prediction improves modestly
from 1 to 14 observations; compact16 benefits more from a small unordered past
summary than longer ordered concatenation in these linear readouts.

[Detailed completed report and plots](../../output/ni_latent_sim/methods-diagnostics-20261011-v1/README.md).
No particles or rows were excluded from primary scores. Three sampling keys do
not replace independent training seeds, and this regime-selected cohort does
not estimate population nucleation prevalence.

## Authorized follow-up: horizon stability and response observability

Recipe: `configs/ni_latent_sim/methods_followup_20261011.json`; the same diagnostic
CLI accepts this recipe. Five frozen transformer checkpoints retain the original
214-row panel, train-only normalization and three nested-particle sampling keys.
Four checkpoints cross 6-ps values/response training with two independent training
seeds. The 1-ps values-only transformer at seed20261014 is the clean horizon
control: both horizon arms used4,000 updates at batch256 and selected update1,000.
The other 1-ps batch revisions are excluded from this matched comparison.

The path audit records extreme-particle rates through1/3/6ps, in addition to the
unchanged endpoint, joint-future and scramble scores. It never removes unstable
particles. The question is whether the failure observed in the 1-ps MLP audit
persists within the training horizon of the 6-ps transformer. A cross-architecture
comparison alone cannot identify the causal effect of horizon length.

The second question is whether the physical derivative is more predictable from
the existing raw280 observation than from compact16. Use all300 physical anchors:
80fit/20selection/200test from16/4/40 ancestors. Nine representations (raw280,
five transformer codes, three original MLP codes) each receive Q and V queries.
Fit a global directional-linear ridge and a state-conditioned product-kernel
readout, totaling36 selected diagnostic fits. Both are exactly linear in the
query direction; the latter has a nonlinear dependence on observed history.
The exact future geometry-JVP targets remain unchanged. Select regularization
and the small kernel-bandwidth grid by selection Gaussian likelihood, not AP.

Zero, exact present geometry JVP and present geometry secant controls help
distinguish future-response difficulty from inability to fit present information.
The t0 target is reported separately and never selects a readout. Retain frozen
predictions and ancestry confidence intervals, without claiming a general
information bound from only80 fitting anchors and two directions per state.

This follow-up reuses observations and simulator targets. No new MD, relaxation
or scientific encoder fit is launched. A two-worker GPU array and an independent
CPU queue provide execution; each GPU model can complete after another fails.
Scientific results belong in
`output/ni_latent_sim/methods-followup-20261011-v1/`.

### Completed follow-up

All five transformer path audits and 36 frozen response readouts completed on
their first attempts (October 11, 23:10:39 Europe/Paris). All five transformers,
including the matched 1-ps control, avoid the predeclared extreme-particle
thresholds through 6 ps. The matched 6-ps values model reduces 6-ps descriptor
mean MSE from 0.699656 to 0.685096 (2.08%). Response supervision has no consistent
benefit across the two training seeds: 6-ps mean MSE changes from 0.675267 to
0.681563 at seed 20261013 and from 0.685096 to 0.676326 at seed 20261014.

Full280 state-conditioned readouts achieve future-response relative MSE
0.986273 (Q) and 0.971912 (V); compact response codes remain near the zero-response
baseline. However, the full-input readout also fails to reconstruct the present
Q response, which the observed secant directly supplies. This diagnostic cannot
establish information absence or attribute failure solely to compression.
A response-change readout with an exact present-response skip term and a
fit-cohort learning curve is the proposed next diagnostic, not yet launched.

[Completed follow-up report and plots](../../output/ni_latent_sim/methods-followup-20261011-v1/README.md).
