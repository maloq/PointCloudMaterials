# Encoder mechanisms: completed results

28 September2026. **All planned unconditional stages completed:** nine label-free
encoder fits and nine likelihood adaptation fits, each24 epochs;54/54 milestone
quality evaluations,54/54 displacement assays and18/18 adaptation endpoint
evaluations. Readout controls and the60-source held-out birth audit also finished.
Birth-prediction and VAMP fits remain gated, as specified in the original plan.

## Alignment trades structural retention for prediction

Fixed epoch24, means over three seeds. Structural measurements first average
the three fixed Al frames within seed. Prediction uses the complete all64 test
population and the same128-unit frozen nonlinear readout.

| Recipe | Liquid-neighbor NMSE ↓ | Nonbulk spatial AUC ↑ | Nonbulk balanced accuracy ↑ | Raw6-ps log loss ↓ | Calibrated6-ps log loss ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial encoder, epoch0 | 1.48151 | 0.57481 | ≈0.642 | 0.07165 | 0.07121 |
| R0: Epi+variance, no alignment | **1.31270** | **0.58122** | 0.65801 | 0.07811 | 0.07496 |
| R1: Epi+variance, aligned | 1.39056 | 0.53880 | 0.65887 | 0.06967 | 0.06843 |
| R2: VICReg, same alignment | 1.54771 | 0.51184 | 0.65949 | **0.06874** | **0.06815** |

R1 versus R0 isolates the alignment coefficient while retaining the same
observed/relaxed pairs, geometric reference, variance penalty and per-seed
initial state/order. Alignment improves prediction in every seed; removing it
improves physical-neighbor and reference-boundary diagnostics in every seed.
Balanced nonbulk accuracy largely misses this difference. Spatial AUC measures
reference-label changes across atom pairs within matched physical-distance
bins, rather than global smoothness or phase discovery.

Aligned-minus-unaligned calibrated6-ps log-loss difference is **−0.006536**,
paired source-bootstrap95% interval **[−0.008522,−0.004728]**. The raw contrast
is−0.008445 [−0.011366,−0.005661], so calibration does not explain away the gain.

VICReg has lower raw log loss than aligned Epi in all three seeds:
Epi-minus-VICReg raw6-ps difference **+0.000923 [0.000405,0.001570]**. After
calibration it shrinks to **+0.000274 [−0.000266,0.000889]**. The calibrated
gap is small and uncertain; Epi retains substantially more physical structure.
Calibrated6-ps Brier point estimates slightly favor Epi:0.016100 versus
VICReg0.016121, further discouraging a universal ranking from one score.

All intervals resample30 whole test sources after averaging loss differences
across the three fitted seeds. They condition on those encoders/heads, do not
cover a population of training seeds, and do not correct for multiple comparisons.
Individual seed contrasts remain available. Predictions are not ensembled.

## What happens along training

| Epoch | R0 liquid NMSE | R1 liquid NMSE | R2 liquid NMSE | R0 calibrated6-ps loss | R1 calibrated6-ps loss | R2 calibrated6-ps loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.48151 | 1.48151 | 1.48151 | 0.07121 | 0.07121 | 0.07121 |
| 4 | 1.30215 | 1.40024 | 1.52280 | 0.07444 | 0.06942 | 0.07000 |
| 8 | 1.31248 | 1.39051 | 1.51163 | 0.07443 | 0.06979 | 0.06925 |
| 12 | 1.31720 | 1.39414 | 1.52928 | 0.07447 | 0.06928 | 0.06865 |
| 18 | 1.31417 | 1.38754 | 1.54706 | 0.07493 | 0.06879 | 0.06827 |
| 24 | 1.31270 | 1.39056 | 1.54771 | 0.07496 | 0.06843 | 0.06815 |

Epi's structural change happens mostly before epoch4, then liquid-neighbor
error is nearly flat while aligned prediction improves. VICReg's liquid-neighbor
error worsens overall while prediction improves; spatial boundary discrimination
approaches chance. This supports measuring structural information independently
of forecasting loss, without claiming every objective continuously erases nuance.

At epoch24, mean within-source Spearman association between embedding movement
and geometry-descriptor changes over0.75 ps is **0.714(R0),0.624(R1),0.186(R2)**.
Epi exceeds VICReg in each seed. This association of two changes is not a
forecast, causal sensitivity measure or demonstrated nucleation coordinate.

Displacement-matched real-MD/Gaussian response ratios are0.072/0.069/0.087 for
R0/R1/R2. All remain sensitive to synthetic displacements, which need not be
physically plausible thermal motion. R1 suppresses both real and synthetic
responses relative to R0. These ratios do not establish selective nuisance removal.

## Frozen Epi beats scratch; fine-tuning adds no demonstrated predictive benefit

Modes receive the same observed geometry and fresh head within seed. Frozen
and fine-tuned modes start from the predeclared R1 epoch24; scratch starts from
its exact retained epoch0. All fits run24 epochs. The primary adaptation
checkpoint minimizes validation-source hazard NLL after epoch12. A fresh frozen
readout is then fit identically to every export; fixed epoch24 is also reported.

| NLL-selected checkpoint | Joint-head raw6-ps loss ↓ | Joint-head calibrated6-ps loss ↓ | Fresh-readout raw6-ps loss ↓ | Fresh-readout calibrated6-ps loss ↓ | Liquid-neighbor NMSE ↓ | Spatial AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Epi + fitted head | **0.06832** | **0.06867** | 0.06967 | **0.06843** | 1.39057 | 0.53880 |
| Fine-tuned Epi + fitted head | 0.06881 | 0.06917 | 0.06967 | 0.06909 | 1.38848 | 0.55474 |
| Scratch encoder + fitted head | 0.07154 | 0.07137 | 0.07096 | 0.07041 | 1.48304 | 0.56761 |

Scratch-minus-frozen calibrated6-ps difference is **+0.002705
[0.001318,0.004288]** with the joint head and **+0.001988
[0.001003,0.003205]** with fresh readouts. Both favor frozen Epi in every seed.
Pretraining helps beyond co-adapting a particular predictor.

Fine-tuned-minus-frozen joint-head calibrated6-ps difference is+0.000501
[−0.000392,0.001374], which is uncertain. Fresh-readout raw6-ps loss is
essentially unchanged (difference−0.000001, interval spans zero), while its
calibrated difference is+0.000665 [0.000053,0.001285]. This fine-tuning recipe
shows no predictive benefit; finite readouts cannot prove intrinsic information
loss or rule out other fine-tuning schedules.

Fine-tuning also **improves spatial boundary AUC in every seed** and leaves
liquid-neighbor error approximately unchanged. It therefore does not support a
simple story that supervised updates necessarily discard all structural detail.

| Fixed adaptation epoch24 | Joint-head calibrated6-ps loss ↓ | Fresh-readout calibrated6-ps loss ↓ | Liquid-neighbor NMSE ↓ | Spatial AUC ↑ |
| --- | ---: | ---: | ---: | ---: |
| Frozen Epi | 0.06855 | 0.06843 | 1.39056 | 0.53879 |
| Fine-tuned Epi | 0.06949 | 0.06992 | 1.37909 | 0.55720 |
| Scratch | 0.07178 | 0.07009 | 1.47269 | 0.56739 |

Whole-event raw test NLL likewise favors frozen Epi: selected means
0.20372(frozen),0.20558(fine-tuned),0.20970(scratch); fixed-epoch means
0.20434/0.20707/0.21003. This conclusion is not based only on a6-ps metric.

## Remaining scientific limits

At the new epoch24 endpoints, same-patch32-descriptor add-back improves
calibrated6-ps loss in every seed. Means are0.07496→0.07144(R0),
0.06843→0.06765(R1),0.06815→0.06763(R2). These are descriptive endpoint
comparisons; the earlier dimension-matched redundant-feature controls supply
the separate capacity check documented in the [initial audit](RESULTS-20260928.md).

The birth audit still has only one established-crystal-clear test birth-positive
observation within6 ps, and none with a strictly PTM-crystal-clear8Å footprint.
Onset results therefore do not establish precursor detection in Al, Ta or Zr.
Static inputs are relaxed and their training ancestry/potential coverage remains
incompletely established; predictive inputs are observed and source-held-out.
These are complementary assays, not identical domains or targets.

The evidence supports treating invariance, physical neighborhoods, spatial
boundaries, temporal physical changes, predictive information and calibration
as separate requirements. Aligned Epi offers a tested structural/predictive
trade-off; keeping it frozen is a stronger matched baseline than scratch here.
No single smoothness, clustering or forecasting score establishes all desired
properties of an atomic-environment representation.

## Saved review

[Training trajectories](/work/PERSO/vmorozov/analysis/encoder_mechanisms/alignment-readout-recovery-20260928-v3/analyses/completed-review-20260928/plots/training-trajectories.png) ·
[Adaptation plots](/work/PERSO/vmorozov/analysis/encoder_mechanisms/alignment-readout-recovery-20260928-v3/analyses/completed-review-20260928/plots/adaptation.png) ·
[Full numeric table](/work/PERSO/vmorozov/analysis/encoder_mechanisms/alignment-readout-recovery-20260928-v3/analyses/completed-review-20260928/tables/completed-results.csv) ·
[Metric definitions](/work/PERSO/vmorozov/analysis/encoder_mechanisms/alignment-readout-recovery-20260928-v3/analyses/completed-review-20260928/tables/METRICS.md).

The review bundle also contains PDF figures, all per-seed values and paired
intervals in `technical/results.json`, and exact input paths/checksums in
`technical/provenance.json`. It summarizes saved results without overwriting
historical tables or fitting another encoder or predictor.
