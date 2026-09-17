# Local crystallization predictability and information retention

Status: **planned, not run**. The latest user constraints are one training seed
(20260919), one local H100 and one remote H200 working concurrently, and a maximum
16-hour window. The [supplied scientific plan](PLANNING_INPUT.md) motivates this
study; its suggestion to add seeds is superseded. The [active recipe](../../configs/local_predictability/two_gpu_16h.json)
is a planning specification, not an executable training configuration.
Execution and transfer instructions are in the [handoff](../../docs/local_predictability_16h.md).

## Question and interpretation

Which observations contain accessible information about future local onset, and
where is information lost between atomic inputs, the exported state and its
[readout](../../docs/research_glossary.md#readout), a predictor fitted to that state?
The supervised onset model is a diagnostic reference. The general encoder still
learns current and future physical quantities without crystallization labels.

The previous 450-window physical pilot did not test supervised onset prediction.
Its 270 training windows concentrated around 300 ps; increasing updates or width
cannot repair that sampling limitation. The older causal pilot's two positive test
windows represented one center event. Conversely, the [September 13 assay](../forecast_crystallization_20260913/RESULTS.md)
reported useful onset-within-9-ps warning (AP 0.447 versus persistence 0.106).
That result is a positive control under its original protocol, not an estimate
for the new sampling population or proof of exact 9-ps advance prediction.

Failure to improve a baseline means limited demonstrated skill for these inputs,
models and data. It does not establish a physical upper bound on predictability.
One seed gives an exploratory comparison; source intervals do not measure
variation between training seeds.

## Cohort and observations

Use the existing 150 independent 600-ps Al sources at 400, 450, 500, 510 and 520 K,
with 0.75-ps measurements and inherited 90/30/30 train/validation/test folds. Keep
all descendants of a source lineage in its fold. Shooting branches are not extra
independent sources. Freeze a 15-source checkpoint-selection / 15-source
calibration division of validation, three sources of each temperature per half,
by the hash rule in the portable manifest. Test sources are inherited and
exploratory, never a new sealed confirmation cohort.

Choose a pool of 64 center IDs per source independently of outcomes. Reuse legacy
64-center selections only after exact trajectory and lineage verification;
otherwise choose uniform atom IDs with the declared seed. Randomly select the
16 core centers from that pool before computing labels. Evaluate existing
verified 64-center labels where available; first compute missing full-timeline
labels for the 16 core centers so a larger census does not delay the experiment.
Report partial census coverage explicitly. Missing sources must be resolved or
block the declared cohort, never silently reduce the 150-source study.

Two fixed observation grids serve different costs:

- Descriptor population: 151 origins, every 3 ps from 48 through 498 ps, for
  16 centers per source: **362,400 candidate windows** before at-risk filtering.
- Native atomic training/comparison: 16 origins per center, every 30 ps from
  48 through 498 ps: **38,400 candidates**, including 23,040 train, 7,680 validation
  and 7,680 test candidates before filtering.

The common grid permits 48-ps history, 96-ps future and 6-ps confirmation padding
for the longest label sensitivity. It deliberately uses much more of each source
than the old three-anchor pilot. Paired model comparisons use identical rows,
physical targets, scales and weights. Additional horizon-specific eligibility
may use more origins but must appear in separate tables with its own denominator.
Do not rank a dense descriptor score against a sparse native score: evaluate the
fitted descriptors again on exactly the native comparison rows.

Histories include every recorded frame, tracked identities, boxes, center-relative
positions and velocities, with consistent periodic images. Descriptor histories
are 0, 3, 12 and 48 ps; native core history is 12 ps (17 observed frames).
Repeated-frame controls copy the current geometry and velocity at every history
slot while retaining the same time offsets and architecture.

## Assay before prediction

Use the existing local PTM producer: nearest 80 periodic neighbors, RMSD cutoff
0.1, crystal structure types 1/2/3. Verify patch/full-cell agreement on a fixed
training sample; selection-only PTM must not delete the neighbors needed to label
a selected center. Label the full timeline, including confirmation after the
forecast endpoint.

Onset is the first frame of the first run of three crystalline observations
(1.5 ps of subsequent confirmation). Five- and nine-frame runs are sensitivities
requiring 3 and 6 ps. An at-risk origin precedes first onset and has three observed
noncrystalline frames ending at the origin. Onset in `(t, t+tau]` is positive;
a confirmed absence is negative. Insufficient follow-up is censored, never
silently negative. The primary fully observed common grid excludes this censoring;
any expanded analysis must carry the censoring mask into the likelihood.

Count source lineages, center episodes, eligible origins and positives by fold,
temperature, horizon and history. Show how much coverage is lost to history,
confirmation and the prior-noncrystalline requirement. Coverage gates are
engineering rules, not power calculations: at least 20 event-bearing train
lineages, five in each validation half and 50 center episodes in each validation
half within the matched population. If they fail, publish coverage and fitting
diagnostics; do not interpret an expensive event comparison as adequately tested.

Reaggregate the original saved predictions under the original 74/24/27 folds,
weights and assay first. Its 125-source/8,000-center result stays separate from the
new protocol. Re-scoring old models on new folds requires a lineage crosswalk:
an old training source cannot become a new held-out test for that model.

Audit current-state observability on an all-state sample, including crystalline
frames (an at-risk-only sample would contain no positive current labels). Compare
current packet and raw-atom state classification; audit true future endpoint
packets for future state, then dense future packet sequences for onset. These
future-input models are explicitly diagnostic oracles, never usable forecasts.
The [packet observability queue](OBSERVABILITY.md) specifies the ten single-seed
fits and saved predictions for deferred analysis.
The 128-component packet is already center-relative, including correlations of
centered bond directions. Verify what it loses before assuming it is merely a
pooled group mean or changing the general target. If an audit fails, report the
mismatch and version a repair; do not quietly change the primary target mid-run.

## Cheap supervised benchmark on H100

Fit regularized linear and modest nonlinear six-bin hazard models for condition
only; current packet; 3-, 12- and 48-ps packet histories; and a repeated-current
12-ps control. Conditions are temperature indicators and elapsed quench time with
a quadratic time term, scaled on training data. History summaries concatenate
current, oldest, mean, standard deviation, time slope, minimum and maximum;
therefore test accessible summarized history, not full-history sufficiency.
Use train-only normalization, condition features in every model, fixed MLP widths
256/128, and one training seed. Select regularization using selection sources;
fit no further neural architecture sweep. A small linear/ridge grid is cheap
model selection, not a multi-seed replication.

Two additional descriptor comparisons audit center-sensitive order quantities
and wider spatial context. Trace the producer in `src/analysis/liquid_structure.py`
and freeze feature definitions before fitting. Wider context uses additional
smooth 7–17 and 17–25 Å shell summaries, computed from actual neighbors. Missing
validated producers block these two jobs; they do not trigger substitute features.
Physical references are train-fitted ridge predictors of the same six future
packets from conditions or each real descriptor history, plus persistence.

Report total skill and prespecified ordering/context strata. Use qbar6 < 0.3 for
a weak-order diagnostic; nearby crystal needs actual PTM labels for the neighbors
within 25 Å, not the 64 sampled centers. If those labels cannot be produced in
budget, report the no-nearby-crystal stratum unavailable. Do not call an unmeasured
neighborhood crystal-free or equate approaching-front skill with nucleation skill.

## Native comparisons on both devices

Both use width 16, two spatial/temporal blocks and one exported 128-dimensional
state. Keep the existing 17 Å observation radius, 5 Å spatial cutoff and smooth
pooling. No width, compression, mixture-head or slowness sweep.

| Worker | Training objective | Three final models |
| --- | --- | --- |
| H100 | Supervised local event-time likelihood | Current x/v; real 12-ps history; repeated current x/v |
| H200 | Current reconstruction + future physical conditional means | Current x/v; real 12-ps history; repeated current x/v |

The physical target is the existing 128-vector (32 radial, 32 pair, 16 angular,
16 speed, 16 signed radial velocity, 16 moments) at 0.75, 3, 9, 24, 48 and 96 ps.
Use a width-128 one-hidden-layer present decoder and a linear future-mean head
from state plus conditions. Average standardized squared error equally across
future horizons; present loss weight is 1. Report each physical block separately.
The six horizons here differ from the old five-horizon pilot and must receive a
new protocol identity. Do not compare raw aggregate scores across those studies.

Physical training uses all fixed origins, including post-onset states. Event
training uses only at-risk rows. Within an objective every model uses the same
rows and source-uniform sampling; no event oversampling. Across objectives,
compare frozen states only on identical external evaluation rows. Crystal labels
never select the physical encoder checkpoint, loss or hyperparameters.

Each objective trains one current-x/v parent for K updates, then initializes
three continuations from exactly that parent. **The continued snapshot also
receives K updates.** Thus all final models share K parent updates and have equal
additional training opportunity. This is two parent stages plus six final stages,
not six independent initializations. Set K once from 1024/2048/4096 using measured
costs and freeze it before core fits; validation every 256 updates. Reset optimizer
state consistently for all children. Use effective batches of eight source-drawn
windows; microbatching may differ by GPU but data order and update counts may not.

History must initially reproduce the useful parent exactly through trainable zero
residual gates. Gate the entire added computation, including support/normalization
changes, not just attention values. Test outputs and shared-parameter gradients
at zero, and nonzero finite gate gradients. Velocity extensions must likewise
have a genuine nested path if introduced; velocity is present in all core models.

Before native fits, require a varied 32-window, eight-train-source small-set fit
(current standardized MSE <= 0.1, every block <= 0.25, at most 2,000 updates).
Failure triggers diagnosis, not an inference about physics. Report encoder
gradient norms from present and future terms on fixed training batches. Compare
native heads to frozen-state ridge and matched nonlinear readouts. Intervene by
replacing state with its training mean and by shuffling within temperature and
30-ps time strata; preserve conditions and record singleton strata. Shuffling is
an inference diagnostic, not another training seed.

Only expand history after both useful validation event signal and physical-mean
competence are demonstrated. A working competence gate is native or frozen-state
ridge within 5% of current-packet ridge MSE on matched validation rows, without a
large concealed block failure. Otherwise focus the report on learning/retention
failures. A matched predictor of z versus z plus raw observations, with a
constant-observation control, remains a subsequent sufficiency test; descriptor
summaries alone cannot replace that test.

## Metrics, calibration and stopping conclusions

Primary horizons are 9 and 48 ps; report the other four without selecting a winner
from test data. Population scores use equal source weight and natural eligible
window prevalence within each source. List sources with no eligible at-risk rows
and normalize event scores over eligible sources; retain the full physical cohort.
Report event-time likelihood, per-horizon
binary log loss, Brier loss, precision–recall/AP and calibration. Include
condition-only and persistence references on the exact same rows. Choose alarm
thresholds only on calibration sources, after checkpoint selection is frozen.

For dense alarm evaluation use 0.75-ps origins, collapse consecutive positive
predictions into one episode, and enforce a 9-ps refractory interval. An alarm is
correct if first onset occurs in its next declared horizon; stop exposure at first
onset. Record unmatched alarms per observed at-risk center-ns. Target at most one
false episode per center-ns on calibration; separately report recall at <=5%
window false-positive rate. These are distinct metrics. Full-timeline dense native
inference can be expensive: first time a fixed training-source subset; if it
cannot fit, report dense descriptor alarms and sparse native population metrics,
and mark native dense alarm results unavailable rather than interpolate scores.

Report event-centered recall by exact lead separately. Timing tables retain
misses explicitly, with miss fraction and detection-lead distributions; if a
single error score is needed, predeclare the miss penalty as that horizon.
Event-centered samples cannot estimate population precision or prevalence.
Bootstrap complete test sources in 1,000 paired, temperature-stratified draws,
keeping all centers/windows together. These intervals condition on the one
trained seed. Save the source-level scores, censor masks and paired row IDs.

First publish assay and descriptor evidence, then complete native matched groups.
Do not wait for optional extensions to issue the core report. Optional H48,
raw-observation sufficiency, wider native context, shooting and precision work is
**disabled by default** under the faster-results request. Existing shooting
families use distinct conditional ensembles; never pool fixed-cell Langevin
position-conditioned shots with deterministic NPT continuations or call them a
velocity/history information ceiling. The completed paired-precision training
sources support a numerical audit, not held-out confirmation. New data generation
and restarts of stopped campaigns are outside this queue.
