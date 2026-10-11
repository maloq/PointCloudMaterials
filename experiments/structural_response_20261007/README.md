# Can simulator derivatives improve useful motion prediction?

## Completed birth feature audit

All eight fast-v4 fits completed their 1,600 updates. On the unchanged unfiltered
birth cohort, a nonlinear readout of the single present-state neighbor-averaged
q6 scalar reaches AP 0.950/0.940/0.898 at 1/3/6 ps. At 3 ps, bond-order controls
reach 0.952, TDA 0.910, radial features 0.716, instantaneous velocity summaries
0.203 and relative-coordinate quantization indicators 0.198. These controls use
inner-training NLL selection, not AP selection. The case-control prevalence is
fixed at 0.20; scores do not describe a prospective scan of arbitrary liquid.

The eight structural+motion CatBoost readouts achieve 3 ps AP 0.918–0.931.
Zeroing their motion code changes AP by at most 0.0044. Permuting structural
codes within original source/frame-matched five-center sets reduces AP to
0.189–0.199. This supports structural reliance in these fitted readouts, not an
information-theoretic claim that velocities cannot aid nucleation prediction.
The eight physical logit surrogates have held-out R2 0.855–0.875 and probability
correlation about 0.95. Bond order and topology dominate their attributions;
these correlated feature families do not give a unique causal decomposition.

At a 3 ps lead, 84.0% of positive neighborhoods and 3.4% of controls contain a
PTM component of at least four atoms. Their mean qbar6 values are 0.371 and
0.175. No positive patch intersects a 64-atom component at this lead. Small
ordered components can precede the selected catalogue root, which is defined
by ancestry tracking and subsequent establishment, not by the first occurrence
of any local structural order. Every Q,V input was reconstructed exactly at its
declared lead; no exact input overlap or shared train/test ancestor was found.

These results explain why retrospective founder-versus-random-center AP is high.
The next discriminating question is establishment versus dissolution among
present-selected embryos with comparable order, using the existing separate
cohort and proper predictive scores. The original benchmark remains unchanged.
Future-law attribution has a separate unresolved numerical-replay limitation;
the completed embedding interventions do not depend on that branch.

See [audit definitions](../../docs/metrics/motion_birth_attribution.md) and
[artifact locations](../../docs/structural_response.md#completed-motion-fits-and-birth-feature-audit).

The [current mathematical formulation](current_method.tex) describes the v4
method in the baseline document's notation: physical observation time, elapsed
horizons, saved frames, local position--velocity input, latent dynamics, fixed
simulator values and derivatives, and frozen birth readouts.

The earlier continuous-motion derivative encoder had much smaller exported
channel variance than its matched value-only controls. A smaller derivative loss
therefore cannot establish improved representation quality. This experiment uses
an explicitly trained structural initialization and an immutable future teacher
so that reducing target scale cannot solve the motion objective.

Train geometry-only MACE128, correlation 3 and maximum angular order 3, on
1,056,768 broad unrelaxed Al/Mg/Ti/Ta neighborhoods for 12 full epochs. Global batch
256; one epoch linear warmup to learning rate 0.001, then 11 epochs cosine decay
to 0.00001. Export the final checkpoint. The two initializations compare direct
embedding VICReg plus bond-order/TDA prediction, and the same objective augmented
with partial-neighborhood prediction of the complete EMA teacher embedding.

For each initialization, compare value-only versus analytic response derivative
training and frozen versus jointly adapted geometry: eight motion fits total,
1,600 updates each. The motion state is (E(Q), 0.1 ps DE(Q)[V]). All arms predict
fixed descriptors and fixed pretrained teacher embeddings of actual futures at
0.1, 0.5, 3 and 12 ps. Short 100 fs MEAM banks supply matched endpoint values and,
in derivative arms, tangent targets. Historical v1/v2 joint arms specify broad
spatial replay; the v3 data revision disables it, keeping the large collection
exclusively in completed spatial pretraining.

New birth prediction uses a pinned `al-nuclei-birth-010` catalogue contract at
exact1/3/6ps pre-emergence leads. Original selection/calibration/test are pooled
for evaluation, with fitting/selection confined to original training ancestors.
The v4 clarification removes additional neighborhood filtering. Retain actual
input availability and physical-event deduplication; random controls exclude
catalogued birth founder IDs without a current-crystallinity screen. This
unfiltered birth assay remains distinct from the historical strict benchmark
and the present-selected embryo-establishment cohort.

The fast v4 comparison replaces nested full-MACE higher derivatives by centered
student differences while preserving analytic physical targets. Full-MACE and
cached-backbone residual-adapter regimes each have matched value/derivative arms.
The adapter changes the representation but cannot recover information discarded
by its fixed backbone; the full-MACE arm tests whether adapting earlier geometry
features provides additional future and birth information. Both structural starts
are reused. Numerical approximation error and complete-update timing are measured
before the scientific launch; motion fitting retains1600 updates per arm.

Primary evidence is paired held-out future likelihood and prediction error,
aggregated by original source ancestor. Compare derivative and value-only arms
within the same initialization and adaptation regime. Inspect embedding variance,
covariance participation rank and future-embedding error. A lower training
response loss or improved AP alone is insufficient. The common descriptor target
supports comparisons across initializations; latent errors use different teacher
coordinates and should be compared within initialization.

The experiment has one initialization seed, so it is an exploratory treatment
comparison. Short fixed-box BAOAB derivatives and long saved NPT futures remain
distinct targets. No 3/12 ps simulator derivative is claimed. Nucleus labels and
held-out original source roles do not train or select these representations.

See [definitions](../../docs/metrics/structural_response.md) and
[execution](../../docs/structural_response.md).

## October 7 morning findings

Both spatial initializations completed 49,536 updates (12 complete epochs).
On the fixed 4,096 held-out Al spatial observations, VICReg plus descriptors
has descriptor half-MSE 0.092502, mean coordinate standard deviation 1.0322
and covariance participation rank 83.92/128. Adding the masked teacher gives
0.093218, 1.0285 and 80.90/128, respectively. These measure descriptor prediction
and variance concentration; they do not establish motion or nucleation utility.
The single-seed comparison shows no clear masked-teacher advantage on these
diagnostics. The fixed final checkpoints remain reusable.

No motion fit started. The mandatory numerical preparation passed the short
response derivative backward but exhausted a 96 GB GPU during the long joint
objective at microbatch 16. All eight downstream jobs remained blocked by the
failed prerequisite. No new future scores, clustering scores or nucleus AP are
available for these encoders.

A separate local repair audit evaluated exact forward-mode AD for the velocity
JVP with the same weights and polynomial contractions. It agrees with the
previous JVP to maximum absolute error 6.44e-6 on the audited observation.
Long-objective backward passed at microbatches 1/2/4/8/16, peaking at
4.68/9.28/18.48/36.87/73.65 GiB. This is a numerical diagnostic, not scientific
training; full nested short-response gradient validation and relaunch remain.
The frozen original source and completed spatial results were not modified.
Receipts are in `technical/forward-ad-audit.json` and `technical/status.json`
under the campaign output recorded in the execution documentation.

## Validated motion restart

Expanded checks rejected the forward-AD candidate: it disagreed with the
reference derivative on a loaded checkpoint despite close forward values.
The motion-v2 protocol retains exact reverse-over-reverse differentiation and
uses smaller graph microbatches,8/4 accumulated to256. On the audited physical
input, the two pretrained encoders' directional derivatives agree with centered
finite differences to relative RMS0.001567 and0.001564. Both complete nested
response-loss backward passes and the densest-observation future-loss backward
passes succeeded. These checks establish numerical readiness, not motion utility.
The eight scientific comparisons are launched from the preserved epoch-12
checkpoints, with one shared latent-dynamics initialization and fixed teachers.
Future prediction and nucleation improvements remain unmeasured for these fits.

## Fast v4 interim findings, October 7 afternoon

Seven of eight fits have completed their 1,600 updates and birth evaluation.
The VICReg value-joint comparison is still incomplete. The requested unfiltered
birth cohort contains 79 held-out physical births across 33 original ancestors,
with identical cases at 1/3/6 ps and pooled original selection/calibration/test.
Its matched-control prevalence is 20%; these AP values are not a direct comparison
with the historical screened birth benchmark.

For the fully completed masked-teacher joint comparison, the same CatBoost
readout gives the following event-weighted held-out AP. All readout selection
uses inner-training NLL, not AP.

| Inputs and training | 1 ps | 3 ps | 6 ps |
|---|---:|---:|---:|
| Structural, derivative-trained | 0.9370 | 0.9333 | 0.8840 |
| Structural + motion, derivative-trained | 0.9394 | 0.9291 | 0.8818 |
| Structural + motion, value-trained | 0.9326 | 0.9268 | 0.8837 |
| Predicted future law, derivative-trained | 0.9141 | 0.9045 | 0.8612 |
| Predicted future law, value-trained | 0.9128 | 0.9144 | 0.8709 |

Motion features and derivative supervision show no consistent birth advantage
in these completed comparisons. At 3 ps, masked-teacher joint future-descriptor
NLL is 1.1983 with derivatives and 1.1987 without them. High absolute birth AP
therefore does not yet establish added information from response training.

The short held-out tangent loss improves from 1.2877 to 1.1288 in that paired
joint comparison, but remains worse than predicting zero tangent. The actual
sealed targets give a zero-tangent loss of 0.5 for both initializations: each
of the two terms contributes 0.25 under the recorded normalization. The VICReg
derivative-joint loss is 1.1762. Thus simulator derivative prediction itself is
still weak despite successful optimization and much faster execution.

The reported birth rows come from each fit's
`analyses/birth-prediction/tables/birth-scores.csv`; future and short-response
rows are in `tables/future-scores.csv` and `tables/heldout-responses.csv` under
the v4 run documented in the execution guide. This is a single-seed interim
comparison, not evidence of a reproducible gain.

## New question: first local crystalline appearance

The high-AP feature audit motivates a separate prospective target: from a local
observation with no present PTM-crystalline atom, predict whether any consumed
atom first becomes crystalline in the next 0.1/0.5/1/3/6 ps. Uniformly sampled
sites at fixed elapsed source times are selected before their futures are read.
Brief and failed appearances count; no eventual established nucleus is required.
The same-atom three-frame sensitivity distinguishes some flicker from persistent
ordering. Past-0.5-ps-clear rows form a reported subgroup, not a fitting filter.

All eight completed response/value models, both original spatial checkpoints,
qbar6, bond-order, TDA and instantaneous velocity controls share present inputs
and labels. Frozen CatBoost probes use training-only NLL selection and original
ancestor roles, with all three held-out roles pooled. The experiment tests whether
motion contributes useful predictive information before existing crystallinity
makes the task easy. It does not optimize AP or retrain the encoders.

This measures a future-horizon risk, unlike lead-before-a-preselected-nucleus
classification. Report natural row and equal-ancestor event prevalence with AP,
NLL, Brier and calibration; the previous forced 20% reference does not apply.
The new assay is exploratory and does not separate homogeneous local ordering
from an arriving crystalline front outside the observation. A first saved onset
does not establish absence of briefer events between 0.1 ps snapshots.

Recipe: `configs/structural_response/first_appearance_20261007.json`.
Definitions and execution: [workflow](../../docs/structural_response.md#first-crystalline-appearance-october-7).
The full comparison is running; no new-model AP result is available yet.
