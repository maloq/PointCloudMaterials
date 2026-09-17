# All available new results: predictive information, native encoders and topology

**17 September 2026; evidence captured at 19:03 UTC / 21:03 CEST.**

**H200 addendum received after capture:** training is reported complete;
validation scores added below. Final test comparisons remain pending.

**The strongest new finding is that useful onset information is accessible in
physical descriptors, while the tested native onset encoders make much weaker
use of it.** Twelve ps of descriptor history improves short-horizon prediction;
broader spatial context gives the strongest descriptor result. The current
GATr implementation does not improve physical prediction over MACE at the tested
budget, although their frozen topology readouts are broadly comparable.

This report covers all completed local studies identified in the configured
H100/RTX6000 outputs, including the previously deferred observability and frozen
readout analyses. It also carries forward the earlier H200 findings and the
subsequently supplied native physical validation scores. **All H200 training is
reported complete; its final native physical test comparisons are still pending.**
Raw H200 predictions have not been supplied locally. H200 values retain their
user-reported status; unavailable test results are not counted as negative results.

## What is complete, and what each experiment tests

Here, [native MACE](../../../docs/research_glossary.md#native-mace-in-the-current-experiments)
means our custom encoder that learns directly from atomic positions, velocities
and optional history using MACE blocks. Its parent starts from random weights.
Physical and onset objectives train separate copies of this architecture.

| Study | Completed evidence | Question |
| --- | --- | --- |
| Broad descriptor study | 16 onset classifiers and five physical ridge fits | Is useful information accessible directly from physical summaries, history or wider context? |
| Native MACE onset | One 4,096-update snapshot parent, then three 4,096-update continuations | Does snapshot/history/repeated-frame atom-level encoding support prospective onset prediction? |
| Frozen MACE onset readouts | Six fresh linear/nonlinear readouts | Can a newly fitted decoder recover more information from the selected frozen states? |
| Packet observability | Ten classifiers | Can the packet recognize current state or onset when the relevant future is explicitly revealed? |
| Raw-atom current state | One 8,192-update classifier | Can the native MACE architecture recognize current crystal state from atoms? |
| Physical MACE/GATr screen | Two 2,048-update fits on each of H100 and RTX6000 | Which configured encoder/head learns fixed present and future physical targets better? |
| GATr onset repetition | One 4,096-update parent and three 4,096-update continuations | Does changing the backbone improve the matched native onset experiment? |
| H200 native physical study | Training reported complete; three validation scores supplied; test comparisons pending | Does real history improve future physical prediction over snapshot and repeated-frame controls? |
| Frozen physical-state topology | Completed ridge/nonlinear readouts and paired analysis | Does a state trained without topology supervision retain instantaneous topology? |
| Legacy crystallization control | Reproduced saved predictions and assays | Does the older positive result reproduce under its own protocol? |

All new local comparisons use **seed 20260919**. The H100/RTX physical screens
repeat that seed and sample schedule; they are not independent seed replications.
Parent and child stages share ancestry, and diagnostic fitting gates are not
held-out research fits. No grand total of independent experiments is inferred
from these work counts. Final completion receipts supersede stale per-step
`running` logs. [Completion records](tables/completion.csv).

## Shared data and the comparison boundaries

The new cohort contains 150 independent Al trajectories at 400, 450, 500, 510 and
520 K. Source folds are 90 training, 15 checkpoint selection, 15 calibration and
30 test. Every source contributes 16 predetermined tracked centers. A
[tracked-center embedding](../../../docs/research_glossary.md#tracked-center-embedding)
describes the environment around that atom, rather than crystallization anywhere
in the simulation.

| Population | Training windows | Selection | Calibration | Test |
| --- | ---: | ---: | ---: | ---: |
| All-state native grid, 30 ps between origins | 23,040 | 3,840 | 3,840 | 7,680 |
| Eligible onset windows, native grid | 11,738 | 2,220 | 1,774 | 4,691 |
| Eligible onset windows, 3 ps descriptor grid | 109,838 | 20,881 | 16,584 | 44,385 |

No source is empty in the primary risk set. The onset target is first local
crystallization sustained for three recorded frames, with two subsequent frames
available for confirmation. Origins must precede first onset and have three
observed noncrystalline frames. Future horizons are 0.75, 3, 9, 24, 48 and 96 ps.
The source-weighted test prevalence on the native grid is **3.00% at 9 ps** and
**17.35% at 48 ps**. Recognition tasks also include already crystalline states.

For cross-model onset tables below, descriptor predictions were restricted to
the exact native source/center/origin rows and their thresholds recalibrated on
the native calibration subset. All native test labels and row identities were
verified. **This matches evaluation, not training exposure:** descriptor models
were fitted on roughly nine times as many eligible training windows. Wider
context also changes available input. These are information references, not
isolated architecture comparisons with equal fitting budgets.

The [physical packet](../../../docs/research_glossary.md#physical-packet-in-the-native-predictability-protocol)
contains 128 smooth geometry/motion coordinates within a region tapered from
5 to 7 Å. Native atomic input extends to 17 Å. Conditions contain temperature
and trajectory time. Present and future physical targets are standardized with
training data only. Topology targets and historical likelihoods have different
definitions and are kept separate.

## 1. Predictability from descriptors: history helps at short horizons; context helps more broadly

Nonlinear descriptor predictors, evaluated on the same **4,691 native test
windows** as the atomic onset models. Average precision (AP) measures ranking
quality at the natural event prevalence; higher is better. Joint NLL measures
the six-bin event-time distribution; lower is better.

| Input | Joint event NLL ↓ | 9 ps log loss ↓ | 9 ps AP ↑ | 48 ps AP ↑ |
| --- | ---: | ---: | ---: | ---: |
| Conditions only | 1.09958 | 0.13114 | 0.0624 | 0.2969 |
| Current physical packet | 1.07460 | 0.10081 | 0.3113 | 0.3447 |
| Packet, 3 ps history | 1.07383 | 0.09061 | 0.4285 | 0.3789 |
| Packet, 12 ps history | 1.08074 | 0.08992 | **0.4554** | 0.3781 |
| Packet, 48 ps history | 1.10714 | 0.09301 | 0.4235 | 0.3442 |
| Repeated current packet, 12 ps slots | 1.08570 | 0.09981 | 0.2968 | 0.3578 |
| Current packet + center-order descriptors | 1.07339 | 0.09968 | 0.3279 | 0.3560 |
| Current packet + context through 25 Å | **1.02485** | **0.07646** | **0.5060** | **0.4672** |

At 9 ps, history12 minus snapshot log loss is **−0.01089**, with a paired
source-bootstrap 95% interval **[−0.01768, −0.00308]**. Against repeated frames,
it is **−0.00989 [−0.01742, −0.00192]**. Both support accessible short-horizon
information in real history under this fitted comparison.

The advantage is not general across scores. At 48 ps the history12-minus-snapshot
log-loss interval spans zero, as does the joint-event NLL difference
**+0.00615 [−0.01393, +0.02621]**. Extending descriptor history from 12 to 48 ps
worsens joint NLL by **0.02640 [0.01169, 0.04168]**. These results do not identify
a physical memory cutoff; they concern the tested summary and fitting procedure.

Wider context lowers joint NLL by **0.04975 [0.04044, 0.05882]** relative to the
current packet, and improves log loss at both 9 and 48 ps. This supports useful
information outside the inner target region. The improvement includes extra
observations and features, so it is not a pure pooling or architecture effect.

The earlier dense-grid headline values remain correct: snapshot/history12/repeat
9 ps AP **0.2499 / 0.3508 / 0.2580**, and wider-context AP **0.4586**. Their change
in the table above comes from evaluation on different origins, not retraining.
Both grids and all six horizons are retained in [the full table](tables/onset_test.csv),
with [paired source differences](tables/paired_onset.csv).

## 2. Native onset states: little useful history gain, and fresh decoders only partially help

Each MACE or GATr architecture has its own snapshot parent, followed by matched
snapshot, real-history and repeated-frame continuations. Future frames never
enter their inputs. The six new frozen-MACE readouts use the same saved z128 plus
conditions, with no original history added. A
[readout](../../../docs/research_glossary.md#readout) is a predictor fitted to that
state; its success tests accessibility of information to that predictor.

| Model and head | Joint event NLL ↓ | 9 ps AP ↑ |
| --- | ---: | ---: |
| MACE snapshot, native head | 1.10252 | 0.0405 |
| MACE history12, native head | 1.10246 | 0.0405 |
| MACE repeat12, native head | 1.10253 | 0.0405 |
| MACE snapshot, fresh linear readout | 1.12560 | 0.0478 |
| MACE snapshot, fresh nonlinear readout | 1.09367 | 0.0772 |
| MACE history12, fresh nonlinear readout | 1.09411 | 0.0777 |
| MACE repeat12, fresh nonlinear readout | 1.09385 | 0.0772 |
| GATr snapshot, native head | 1.10336 | 0.0362 |
| GATr history12, native head | 1.09750 | 0.0369 |
| GATr repeat12, native head | 1.09556 | 0.0364 |

MACE history's joint-NLL improvement over its snapshot is only **0.000061**
(about **0.0055%**). Its paired source interval excludes zero, but the effect is
too small to support a practical improvement; the 9 ps ranking scores are almost
identical. The fresh nonlinear readouts also give almost identical results across
the three states. History's readout NLL is slightly worse than both controls.

A fresh nonlinear decoder increases snapshot AP from 0.0405 to 0.0772, but the
paired joint-NLL improvement **−0.00884 [−0.04215, +0.02455]** is unresolved.
We did not compute an AP-difference interval. The decoder also changes fitting
and selection procedures, so this point improvement does not isolate capacity.
It remains far behind the packet's 0.3113 AP. Its 9 ps log loss exceeds the
packet predictor by **0.03096 [0.02044, 0.04366]** on identical test rows.

GATr history improves joint NLL over its own snapshot by
**0.00586 [0.00014, 0.01171]**, but the repeated-frame control has a lower point
estimate. History minus repeat is **+0.00194 [−0.00267, +0.00705]**. Thus this
comparison does not establish a real-history benefit over both controls. Its
9 ps AP remains close to the 3.00% prevalence reference.

![Matched native-grid onset comparisons](plots/onset_comparison.png)

**Threshold transfer is also weak for several native models.** Thresholds were
chosen to satisfy calibration-window FPR ≤5%; they were not tuned on test.
At 9 ps, MACE snapshot has test recall **13.8% at FPR 10.9%**. Its fresh nonlinear
readout reaches **19.1% at FPR 6.9%**. Packet history12 reaches **68.0% at FPR
6.3%**, while wider-context packet reaches **73.0% at FPR 4.5%**. These are
operating points after calibration transfer, not recall values at an exactly
matched test false-alarm rate. Dense alarm episodes and onset timing have not
been evaluated for this new cohort.

## 3. Observability: the labels are accessible, including from raw atoms

These are newly analyzed saved RTX6000 outputs. Current-state recognition uses
7,680 all-state test windows, of which 33.84% are crystalline. The packet and raw
classifier labels were paired exactly.

| Available input and target | Test log loss ↓ | AP ↑ | Recall | Test FPR |
| --- | ---: | ---: | ---: | ---: |
| Current packet → current state, linear | 0.04269 | 0.9977 | 99.0% | 3.1% |
| Current packet → current state, nonlinear | **0.03606** | **0.9985** | 99.6% | 3.2% |
| Current raw atoms → current state, MACE | 0.06618 | 0.9940 | 98.5% | 3.6% |
| True 9 ps endpoint packet → endpoint state, nonlinear | 0.04182 | 0.9982 | 99.4% | 2.9% |
| True 48 ps endpoint packet → endpoint state, nonlinear | 0.04547 | 0.9982 | 99.3% | 3.5% |
| True future sequence → onset within 9 ps, linear | 0.02240 | 0.9370 | 97.8% | 3.4% |
| True future sequence → onset within 48 ps, linear | 0.09731 | 0.9714 | 94.4% | 4.0% |

**The last four rows are future-input diagnostic oracles, not forecasts.**
Endpoint state and first sustained onset are distinct targets; the sequence
includes the two frames needed to confirm endpoint onset. Sequence rows use the
4,691 at-risk windows. Linear sequence readouts have lower log loss than the
tested nonlinear versions at both horizons; the nonlinear 48 ps AP is slightly
higher. All ten packet diagnostics and the raw classifier are in
[the observability table](tables/observability.csv).

This establishes that the tested observations can express the labeling rule
well when the relevant state is observed. The successful raw-atom classifier
also shows that this native architecture can learn a useful structural
distinction. It does **not** establish that a causal local history determines
the future. Together with the descriptor forecasts, the result directs attention
to native representation learning, sampling and decoder optimization rather than
an assertion that the physical packet or raw inputs contain no useful information.

## 4. Fixed physical prediction: MACE beats this GATr screen; direct packets beat both

All models below use the same all-state native test rows and standardized future
packet. Ridge means a linear regression with regularization selected on selection
sources. Atomic models have the same exported dimension and physical-head design,
trained for 2,048 batch-eight updates; both select the final update.

| Predictor | Present MSE ↓ | Mean future MSE ↓ | 9 ps MSE ↓ | 96 ps MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Conditions, ridge | — | 0.86317 | 0.87119 | 0.85246 |
| Current packet, ridge | — | 0.72644 | 0.71822 | **0.78219** |
| Packet history3, ridge | — | 0.72194 | 0.71159 | 0.78578 |
| Packet history12, ridge | — | **0.71961** | **0.70817** | 0.78540 |
| Packet history48, ridge | — | 0.72032 | 0.70925 | 0.78598 |
| MACE snapshot, H100 | **0.53652** | 0.79993 | 0.79848 | 0.81619 |
| GATr snapshot, H100 | 0.71389 | 0.85640 | 0.86317 | 0.84973 |
| Persistence | — | — | 1.32295 | 1.47517 |

GATr's future MSE is **7.06% higher** than MACE's. The paired absolute difference
is **+0.05648 [0.04133, 0.07377]**. MACE exceeds the current-packet ridge error by
**0.07349 [0.06471, 0.08298]**. These comparisons establish a performance gap at
this budget, not the best achievable result for either architecture.

The 12 ps packet history reduces mean error by **0.00683 [0.00531, 0.00857]**
relative to the snapshot packet, about **0.94%**. At 96 ps, beyond the history
duration, it instead increases error by **0.00321 [0.00095, 0.00533]**. Thus the
small average history benefit is not a demonstrated longer-horizon advantage.

The atomic screen saw 16,384 sampled examples per model, drawn with replacement
from 23,040 available training windows. That is not one complete pass over all
training windows. GATr's present geometry errors are especially weak: radial,
pair and angular block MSE **0.802 / 0.881 / 0.994**, versus MACE
**0.499 / 0.644 / 0.718**. GATr is slightly better on speed and signed radial
velocity. The result concerns encoder plus native head; frozen physical readouts
are needed to separate inaccessible information from information already lost.

The RTX6000 repeats agree with the H100 aggregate present/future scores to within
**2×10⁻⁷** for each architecture. This is useful numerical reproducibility
evidence, not extra statistical replication. [Full physical table](tables/physical.csv),
[paired differences](tables/paired_physical.csv), and
[original block scores](../h100-backbone-v2-20260917/tables/axial_gatr_physical_means_snapshot_test.csv).

![Physical forecast errors across six horizons](plots/physical_forecasts.png)

## 5. Topology retention: a limited linear-readout advantage for GATr

These readouts use the frozen physical snapshot states above. **The encoders had
no topology supervision.** Targets are
[instantaneous topology](../../../docs/research_glossary.md#instantaneous-topology)
of current coordinates, with equal H0/H1/H2 block weighting. They are not relaxed
structures or future topology. Noncrystalline means PTM outside types 1/2/3,
including unknown labels: 5,081 test windows, not 5,081 unique centers.

| Test population / readout | MACE MSE ↓ | GATr MSE ↓ | GATr relative reduction, source 95% interval |
| --- | ---: | ---: | --- |
| All states / ridge | 0.22878 | 0.22265 | 2.68% [−3.07%, 8.53%] |
| All states / nonlinear | 0.19837 | 0.19719 | 0.59% [−7.74%, 8.95%] |
| Noncrystalline / ridge | 0.24042 | 0.22173 | **7.78% [0.48%, 14.94%]** |
| Noncrystalline / nonlinear | 0.21400 | 0.20734 | 3.11% [−7.54%, 13.52%] |

The noncrystalline linear result is a useful positive signal for GATr. The
stronger readout leaves the overall difference unresolved. Both outperform the
all-state training-mean and condition-only references (MSE 0.94457 and 0.68413).
However, **within-source/frame noncrystalline R² is negative for all three
topological blocks and both encoders**: good aggregate accuracy does not yet
demonstrate recovery of subtle differences between local environments in the
same frame. [Full topology analysis](../tda-backbone-v2-20260917/RESULTS.md),
[exported values](tables/topology.csv).

## 6. Latest H200 physical study: training complete, validation nearly tied

The receiving-server update reports all training complete, with these validation
future-MSE values:

| Input | Validation future MSE ↓ |
| --- | ---: |
| Snapshot | 0.75121 |
| Real 12 ps history | **0.75060** |
| Repeated current frames | 0.75114 |

History lowers the reported error by **0.00061 (about 0.08%)** against snapshot
and **0.00054 (about 0.07%)** against repeated frames. These are very small
validation point differences. No source intervals, final test comparisons or
exact validation-subset/reduction details were supplied. They cannot establish
a held-out history benefit or be ranked against the local test MSEs above.

Exports had stopped at an open-file limit. The remote update says the limit was
raised, opening all 150 sources was verified, and exports/evaluation resumed
detached without retraining. The quoted **2–3 hours** was the remote estimate at
that update, not a newly verified completion time. See the
[reported scientific evidence](technical/h200_native_reported_results.json) and
[execution addendum](../../../docs/local_predictability_16h.md#h200-reported-export-resumption-17-september).

The accompanying H100 descriptor AP values, **0.250 → 0.351 with history,
versus 0.258 for repeated frames**, reproduce the rounded dense-grid results
already included in section 1. They are not an additional experiment or the
native-grid scores in the main comparison.

## 7. Earlier H200 findings and longer-budget H100 follow-up

These are context under **different scientific protocols**, not additional rows
in the native onset comparison. H200 means and qualitative intervals come from
the supplied summaries; raw H200 per-source predictions and numerical interval
endpoints are unavailable locally.

**Earlier causal-state study:** 18 encoder fits, 48 probes, three seeds; 2.25 ps
history; frozen nonlinear prediction on the predeclared 17-source low-order
subset. Nine ps standardized physical MSE:

| Input | Width 16 | Width 32 |
| --- | ---: | ---: |
| Snapshot | 0.4009 | 0.4028 |
| Real history | 0.3990 | **0.3946** |
| Repeated frame | **0.3963** | 0.4056 |
| Persistence | 0.7209 | 0.7209 |

Width32 history reportedly improves over its snapshot by about 2.0% and repeated
frames by 2.7%, with paired source intervals excluding zero. The width improvement
for history itself is inconclusive; the history advantage is less clear over
all 30 test sources. Width16 favors the repeated-frame point estimate.

**Earlier partial-observation study:** eight H200 fits, two seeds, 3,000 updates,
five future lags through 96 ps; joint physical-path NLL per standardized
coordinate and lag:

| Input | Width16, H100 | Width32, H200 |
| --- | ---: | ---: |
| Snapshot positions/velocities | **0.9247** | 0.9354 |
| 12 ps history | **0.9254** | 0.9369 |
| 48 ps history | **0.9256** | 0.9302 |
| Repeated current frame | **0.9236** | 0.9405 |

Width32 was worse in all eight reported per-seed point estimates, with six
source intervals favoring width16. Longer history gave mixed results; increasing
width had no established benefit at this budget. These continuous-density NLLs
cannot be compared numerically with the new six-bin event-time NLLs.

The subsequent **local 12,000-update objective follow-up stopped at 10/16 fits**.
Under stronger present reconstruction (weight1), seed20260917 snapshot/H12/H48/repeat
NLL was **0.94213 / 0.91071 / 0.90483 / 0.92719**. H48 improved versus both controls
with source intervals excluding zero, but its second-seed replication was never
run. The last completed seed20260918 original-loss H12 model improved future MSE
by about 1.1% against snapshot, while joint NLL remained unresolved. It cannot
replicate the stronger-loss result because the objective differs.

The earlier short causal pilot also showed a history improvement over a weak
velocity baseline, but position-only models remained better; the longer local
causal comparison did not clearly retain that history advantage. None met the
0.10 normalized-jump target. Full seed, uncertainty, sufficiency, Gaussian-head
and state-use diagnostics remain in the
[previous comprehensive report](../../predictive_memory/research-summary-20260917/RESULTS.md)
and [final stop update](../../predictive_memory/research-summary-20260917-stopped/RESULTS.md).
The stopped precision campaign produced three completed training sources, but
no matched precision/cadence model result; its preservation record is in
[simulation documentation](../../../docs/simulations/predictive_memory_precision_20260917.md).

## 8. Legacy positive control: reproduced, but still not exact long-lead onset prediction

The earlier 125-source protocol reproduced its nine ps, three-frame-onset
result: autoregressive forecast **AP 0.44692**, direct forecast **0.43957**,
persistence **0.10585**. Autoregressive precision/recall were **49.0% / 51.5%**,
with F1 **0.5021**. This preserves the earlier positive result under its original
cohort, sampling, threshold selection and aggregation.

The exact-nine-ps-lead event subset is harder: autoregressive recall is **25.7%**
over 1,366 eligible positive origins, and recall with timing within 1.5 ps is
**4.0%**. Timing MAE is **4.85 ps among detected cases**; it excludes misses and
must be read with recall. AP on that positive-only fixed-lead subset is
uninformative. These metrics distinguish “onset sometime within nine ps” from
an exact nine-ps-ahead warning. Neither legacy AP nor recognition accuracy should
be used as a direct ranking against the new native-grid assay.

Sources: [reproduced onset table](../legacy-reproduction-20260917/tables/local-onset.csv),
[fixed-lead table](../legacy-reproduction-20260917/tables/fixed-lead.csv).

## 9. Runtime findings are separate from research accuracy

The recent H100 resident-batch profile compares cuEquivariance MACE with GATr on
the same eight windows, FP32 precision and GPU:

| Input | MACE update | GATr update | GATr/MACE throughput | MACE/GATr peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| Snapshot | 0.0517 s | 0.0345 s | **1.50×** | 1.91 / 0.88 GiB |
| 12 ps history | 0.4710 s | 0.5164 s | **0.91×** | 16.78 / 23.45 GiB |

GATr is faster for this snapshot workload, but its history workload is slower
and uses more memory. These are warmed resident-batch timings, not full-run
throughput or equal-wall-clock training outcomes. They provide no H100-versus-H200
hardware conclusion. The frozen v1 MACE onset scientific reference used e3nn;
new physical screens and raw-state/frozen-state execution use the accelerated
path. Historical v1 elapsed times were not used for acceleration claims.

The input-prefetch validation separately measured 20 matching CuEq training
batches at 12.56 s sequentially versus 8.87 s with lookahead (1.42×), with final
parameter discrepancy about 2.4×10⁻⁷. This is an engineering check on a short
workload. [Execution notes](../../../docs/local_predictability_16h.md),
[same-H100 measurements](tables/h100_speed.csv). No hardware benchmark or
training run was launched to prepare this report.

## What these results justify next

1. Use the current-packet and wider-context predictors as concrete information
   references. Match native training origins and fitting exposure before assigning
   the remaining gap specifically to architecture.
2. Diagnose the physical and onset decoders separately. The successful raw
   current-state classifier and packet forecasts argue against treating weak
   native onset performance as a physical predictability limit.
3. Keep MACE as the physical-screen reference. GATr merits a bounded readout or
   geometry-retention test, rather than assuming its snapshot speed or one linear
   topology gain implies better dynamics prediction.
4. Obtain the H200 test exports before closing the new physical-history
   comparison. Training is reported complete and validation is nearly tied;
   the final paired test evaluation is still needed.

These are recommendations, not new queued experiments. More seeds, the matched
native-grid descriptor retraining, physical-state readouts, broader native
context, full-history sufficiency tests, new dense alarm/timing assays and
precision/cadence comparisons remain unresolved. The new work has not established
a sufficient predictive state, a Markovian embedding, a physical memory cutoff,
or a model satisfying the earlier smoothness criterion while retaining utility.

All new source intervals condition on one training seed. The same inherited test
sources have been examined during development, so results are exploratory.
Source bootstraps preserve correlated windows within source; they do not measure
initialization uncertainty. New paired onset/physical intervals use 1,000
temperature-stratified draws; the copied topology analysis retains its own
4,000-draw source procedure. No multiple-comparison correction is claimed.

## Reproducibility and checks

The CPU collector verified exact native test identities/labels across all 28
native-grid onset model exports and exact physical identities with matching
standardized targets. It reproduced saved native-head NLLs, analyzed all eleven
observability and six frozen-readout outputs, and exported full-precision tables
plus PNG/PDF figures. Eight focused alignment/metric tests passed.

[Metric definitions](tables/METRICS.md), [implementation hashes](technical/metric-contract.json),
[input hashes](technical/inputs.json), and [reproduction command](README.md)
identify the computations. Original prediction/checkpoint files and historical
reports are preserved. The attached plan is treated as methodological context;
its proposals are not counted as completed evidence.
