# Does simulator-response learning improve unrelaxed Ni eSEN?

Start from the completed unrelaxed Small eSEN (12 epochs, 64 channels, 128 exported
features). Adapt it on Ni Q,V, using Ni GPU EAM/FS analytic short responses and
saved longer NPT futures. Compare joint response/value-only training and frozen
backbone response/value-only adapters, 1600 updates each, with the spatial start.
No crystallinity loss or structural replay is used in motion learning.

Evaluation distinguishes first crystalline appearance from persistent32/64-atom
liquid-origin nuclei. Default establishment is64 atoms for1.5ps. Readouts use
original training sources only, and pool original selection/test sources.
[Full metric/time/population definitions](../../docs/metrics/ni_esen_response.md).

Recipe: `configs/ni_response/esen_unrelaxed_20261009.json`.
Run `python -m src.research.ni_response.workflow submit --config` with that path,
inside conda pointnet-torch214. Existing launches are resumable through recorded
worker jobs; do not submit duplicate campaigns. Results and ownership receipts:
`${storage:training_storage}/ni_response/unrelaxed-esen-20261009-v1`.

## Completed unrelaxed results, October 9

All four arms completed 1600 updates and all five frozen-readout evaluations
completed. Original selection/test sources are pooled: 8027 initially clear
observations across 40 ancestors for appearance; 80 persistent32 events across
25 ancestors; 49 persistent64 events across 22 ancestors. Nucleus controls have
20% weighted positive prevalence and remain an event-matched localization assay.
Nucleus offsets are leads before the persistent interval starts, whereas
appearance offsets are future observation horizons.

CatBoost AP using the same structural+motion feature space for every model:

| Task / offset | Spatial start | Value joint | Response joint | Value adapter | Response adapter |
|---|---:|---:|---:|---:|---:|
| Appearance within 1 ps | .259 | .336 | .339 | .309 | .301 |
| Nucleus32, 1 ps lead | .724 | .734 | .748 | .740 | .738 |
| Nucleus32, 3 ps lead | .648 | .701 | .704 | .686 | .682 |
| Nucleus32, 6 ps lead | .561 | .625 | .624 | .599 | .615 |
| Nucleus64, 1 ps lead | .745 | .770 | .752 | .786 | .730 |
| Nucleus64, 3 ps lead | .629 | .685 | .678 | .691 | .713 |
| Nucleus64, 6 ps lead | .581 | .653 | .639 | .635 | .632 |

These are point estimates, not a selection of models by AP or a paired
significance result. Source-bootstrap intervals remain broad: nucleus64/6ps AP
is [.519,.652] for the spatial start, [.566,.747] for value joint and [.552,.742]
for response joint. Value/response joint nucleus64/6ps binary NLL improves from
.407 to .350/.343 and Brier from .128 to .108/.107, respectively.

The representation improvement does not yet establish a useful motion component.
For nucleus64/6ps, spatial-start/value-joint/response-joint structural-only AP is
.607/.687/.648. Their motion-only AP is .203/.244/.240. In this readout, adding
motion to the value-joint structure lowers AP from .687 to .653. Derivative
supervision has no consistent additional advantage across these tasks, though
the frozen response adapter has the highest point estimate at nucleus64/3ps.

Exact values, uncertainty, predictive scores, population counts and frozen metric
definitions are under `analyses/evaluation-v1/<model>/{tables,technical}` in the
run. No crystallization labels or AP values selected motion-training checkpoints.

## Completed relaxed comparison, checked October 11

All four relaxed-observation arms finished1600 updates, their five model
evaluations completed, and the paired relaxed-minus-unrelaxed analysis completed.
Original MD continues from original Q,V; only encoder observations/targets are
quenched. The labels, rows, ancestry and CatBoost structural+motion readout match
the unrelaxed comparison above.

| Relaxed observation task / offset | Spatial start | Value joint | Response joint |
|---|---:|---:|---:|
| Appearance within1ps | .319 | .379 | .354 |
| Nucleus32,1ps lead | .796 | .819 | .809 |
| Nucleus32,3ps lead | .795 | .784 | .762 |
| Nucleus32,6ps lead | .718 | .738 | .730 |
| Nucleus64,1ps lead | .750 | .749 | .756 |
| Nucleus64,3ps lead | .717 | .776 | .741 |
| Nucleus64,6ps lead | .657 | .673 | .682 |

For value-joint nucleus64/3ps, relaxing observations improves AP by .091,
with whole-source paired-bootstrap95% interval[.015,.161]. At6ps its improvement
is .020 with interval[-.054,.092], so that specific comparison is inconclusive.
Relaxed versus unrelaxed value-joint nucleus32/6ps improves by .113 with paired
interval[.058,.161]. These intervals compare entire input/checkpoint treatments;
they do not isolate relaxed pretraining from relaxed inference preprocessing.

Within the relaxed model, nucleus64/6ps response-joint AP is .690 structure-only,
.587 motion-only and .682 combined. Motion-only is now informative, but appending
it still does not improve the structural readout. Value-joint at3ps gives .764
structure-only and .776 combined, a small point-estimate gain requiring paired
uncertainty before declaring added motion information. Response supervision
remains mixed; future descriptor NLL is not compared across differently defined
relaxed/unrelaxed targets.

These training targets retain the original unsorted-force quench recipe. The
later repeatability study fixed identical-input force-order variation with
sorted neighbors, but did not replace these frozen targets. Minima also remain
sensitive to tiny input changes. Scores establish the behavior of the recorded
pipeline, not a smooth physical derivative through relaxation.
