# Fixed-snapshot spatial approach: analysis, 26 September 2026

Spatial context carries substantial information about distance to an existing
crystal. Vector messages give the best held-out distance likelihood in this
single-seed comparison. The initial alarm measures close proximity and is a poor
summary of longer warning distances: a broader alarm, evaluated from the same
saved probabilities, gives median warnings around 12–13 Å. These detections
mostly occur after crystal atoms are already present in the contextual inputs.
Reliable warning from entirely noncrystalline input neighborhoods is not yet
established.

## What was fitted and evaluated

Five spatial predictors were trained for sixteen epochs, batch/microbatch 256,
with one seed and validation distance-NLL selection. Four use one **frozen**
observed-geometry MACE encoder previously trained with temporal-onset likelihood;
the geometric-descriptor baseline has no encoder. This is supervised transfer,
not a comparison of five newly trained encoders or self-supervised methods.

The fixed Al64 release retains 126,545 rows, with 90/15/15/30 source roles for
training/selection/calibration/test. Pointwise test metrics cover 45,291 rows in
30 sources. Controlled scans cover 495 test approaches and 292 away controls in
28 test sources; 143 separate calibration away paths set alarm thresholds.
Routes approach known crystals through fixed snapshots. There is no motion
through evolving MD and no prediction of nucleation here.

Distance is to the nearest atom in a currently confirmed crystal component,
not to a fitted interface. All reported distances are Å. Local patches contain
up to 80 nearest atoms within 8 Å. Context models share the encoder over 25
patches with nominal offsets 0/10/20 Å and maximum input extent 32 Å. They receive
no temperature, simulation age/time, path direction, true distance, PTM labels,
velocity or future observation.

## Distance prediction: the main positive result

These proper scores use equal weight per test source. Lower NLL and Brier are
better. AP is a diagnostic for **current distance ≤8 Å**, not temporal 3/6 ps
crystallization AP, so those previous AP numbers are not comparable.

| Predictor | Distance NLL | Brier, within 8 Å | AP, within 8 Å |
| --- | ---: | ---: | ---: |
| Geometry MLP | 1.0677 | 0.07625 | 0.6564 |
| Local frozen MACE | 0.9676 | 0.05075 | 0.8102 |
| Symmetric context | 0.5413 | 0.00888 | 0.9943 |
| Vector messages | **0.5174** | **0.00833** | **0.9949** |
| Harmonic hierarchy | 0.5352 | 0.00913 | 0.9938 |

The training-prior reference NLL is 1.2505. Adding symmetric context reduces NLL
by 44.1% relative to local MACE. Vectors reduce it by another 4.4% relative to
symmetric context, and by 3.3% relative to harmonic hierarchy.

Paired source-bootstrap comparisons use 5,000 draws of the same 30 test sources.
Negative differences favor the first model. These intervals quantify source
uncertainty conditional on the fitted models, not uncertainty over training
seeds; they are not adjusted for multiple comparisons.

| Comparison | NLL difference | 95% source interval | Sources improved |
| --- | ---: | --- | ---: |
| Local MACE − geometry | −0.1001 | [−0.1267, −0.0767] | 29/30 |
| Symmetric context − local MACE | −0.4263 | [−0.4802, −0.3727] | 30/30 |
| Vectors − symmetric context | −0.0239 | [−0.0314, −0.0156] | 27/30 |
| Harmonic − symmetric context | −0.0060 | [−0.0143, +0.0027] | 21/30 |
| Vectors − harmonic | −0.0178 | [−0.0241, −0.0118] | 26/30 |

The main benefit is spatial context. The additional vector benefit is smaller
but consistent across sources in this run. A harmonic advantage over symmetric
context is not established. No model has been promoted or selected using AP or
these test comparisons.

## Original alarm: close-range detection

The original alarm uses P(distance ≤8 Å), with two consecutive observations
above a threshold calibrated to no more than 5% empirical false alarms on the
143 calibration away paths. Each model achieves 7/143 = 4.90% there. Its actual
held-out false-alarm rate must be measured separately.

| Predictor | Median warning, detected paths | Missed approaches | Recall at ≥12 Å | Test away false alarms |
| --- | ---: | ---: | ---: | ---: |
| Geometry MLP | 0.00 Å | 85.1% | 2.4% | 2.1% |
| Local MACE | 0.00 Å | 82.2% | 3.4% | 2.4% |
| Symmetric context | 4.66 Å | 6.9% | 2.0% | 1.7% |
| Vector messages | 4.76 Å | 4.2% | 2.4% | 1.7% |
| Harmonic hierarchy | 5.28 Å | 2.2% | 2.4% | 2.7% |

Medians exclude missed paths; recall uses every approach path. Zero median for
the local baselines means most of their successful alarms occur on crystal
atoms, while most approaches are missed altogether. The slight local-baseline
advantage in the rare ≥12 Å alarms is not convincing early-warning performance:
their far-control false-alarm rates are of a similar order, and the routes are
not matched exposure distributions for a formal significance test.

The context models recognize proximity reliably once close enough. However,
requiring two observations and using a probability of being **within 8 Å**
makes a late warning unsurprising. This metric alone cannot establish that the
representation lacks information about more distant crystals.

## Exploratory reanalysis: read out the broader distance probabilities

The model already predicts six distance bins. Without retraining, I evaluated
all five cumulative scores P(distance ≤R), R=4/8/12/20/32 Å. Each score gets its
own threshold using calibration away paths only. All 25 combinations are
exported; none was selected by test performance. This analysis was motivated by
the first results and is **post hoc**, not the original confirmatory protocol.

For the illustrative 20 Å score:

| Predictor | Median warning, detected paths | Missed approaches | Recall at ≥12 Å | Test away false alarms |
| --- | ---: | ---: | ---: | ---: |
| Geometry MLP | 0.00 Å | 85.3% | 2.6% | 3.1% |
| Local MACE | 0.00 Å | 81.8% | 3.6% | 2.4% |
| Symmetric context | 11.69 Å | 5.9% | 44.0% | 4.1% |
| Vector messages | 13.11 Å | 3.4% | 57.6% | 5.5% |
| Harmonic hierarchy | 12.32 Å | 4.6% | 50.5% | 3.4% |

These are not comparisons at identical test false-alarm rates. In particular,
vectors exceed the nominal 5% target on test controls: 16/292 false alarms. The
empirical calibration target is not a guaranteed population rate. Harmonic has
10/292 false alarms and symmetric has 12/292. We cannot claim vectors dominate
every warning/false-alarm tradeoff.

The 32 Å score barely extends median warning (symmetric 11.75, vectors 13.31,
harmonic 12.50 Å). Vector false alarms rise to 7.5%. For the 20 Å score only
2.0%, 3.4% and 2.8% of all approaches warn at ≥20 Å for symmetric, vector and
harmonic respectively. Reliable warning at 20–30 Å remains unestablished.

## Is the warning caused by information in liquid alone?

The input audit directly checks whether any consumed atom belongs to a
confirmed reference crystal, and separately whether any is PTM FCC/HCP/BCC.
These flags are evaluation labels, never predictor inputs.

Reference crystal first enters the contextual input at a median distance of
**21.70 Å**, compared with **5.60 Å** for a local patch. This instantaneous
visibility diagnostic is not an operational predictor or an upper bound on
possible liquid-structure information; learned alarms require two observations.
It does show that the 12–13 Å contextual warnings usually occur after the model
has already received geometry containing crystal atoms, and before the focal
local patch normally reaches the crystal.

For the 20 Å score:

| Model | Detected approaches | Reference crystal visible at alarm | Early alarms without reference crystal | Early alarms without any PTM crystal |
| --- | ---: | ---: | ---: | ---: |
| Symmetric | 466/495 | 458/466 (98.3%) | 8/495 | 0/495 |
| Vector | 478/495 | 465/478 (97.3%) | 13/495 | 0/495 |
| Harmonic | 472/495 | 460/472 (97.5%) | 12/495 | 0/495 |

Early here means warning distance >8 Å; both observations contributing to the
alarm must be clear. Absence of an established component is weaker than absence
of all PTM crystalline atoms. These counts do not establish a reliable
noncrystalline precursor signal.

There are also limits to this negative result. On test scans, 1,869 observations
at 8<distance≤32 Å contain no reference crystal in context, but only 316 contain
no PTM crystal at all. Repeated observations along paths are not independent.
On the original fixed test cohort, the 34,433 reference-clear context rows have
**zero positives at distance ≤8 Å**. Consequently the original high proximity
AP cannot test recall before reference crystal enters context.

Context models still have lower distance NLL on the common reference-clear
subset: local 0.6266, symmetric 0.4657, vector 0.4596, harmonic 0.4503. Thus it
would also be incorrect to conclude that all contextual information disappears
when reference crystal is absent. These are source-weighted scores recomputed
within the subset, largely for farther-distance outcomes; they do not by
themselves demonstrate useful early alarms or an intrinsic liquid precursor.

## Dataset and readout limitations

1. **Population shift.** Training observations are the historical liquid,
   pre-first-onset centers. Scans introduce new centers and end on crystal
   atoms. For local MACE, mean P(distance≤8 Å) in the 0–4 Å band falls from
   0.872 on fixed test observations to 0.517 on scan observations. For vectors
   it falls from 0.996 to 0.894. The populations differ, so these changes are
   descriptive evidence of transfer difficulty, not a matched causal diagnosis.
2. **The two-observation rule matters.** A local model may only become confident
   at the final position, leaving no second positive observation. Its 82% miss
   rate therefore describes this detector and scan protocol, not a proof that
   its embedding contains no spatial information.
3. **Routes and controls are limited.** Toward paths aim at known crystals; away
   paths remain beyond 32 Å. Tangential motion, near misses and autonomous search
   have not been evaluated. Very distant false alarms are not equivalent to
   false alarms near an interface.
4. **No temporal or nucleation claim.** These are snapshot proximity results,
   not evidence for a particular temporal lead, crystal-front velocity, nucleus
   emergence or a precursor to future crystallization.
5. **One frozen encoder and one seed.** This study isolates readout/context
   differences. It neither ranks encoder-training methods nor measures seed
   variability. Raw predictions and labels agree across model evaluation rows;
   frozen artifacts pass their recorded checksums. A manual cross-path check of
   1,183 shared starting positions for local/vector/harmonic predictors found
   near-probability differences below 6.1e-6 between the reused and newly
   extracted inputs (matching source, frame and unique recorded distance).

## What I would do next

First improve the spatial experiment rather than immediately enlarge the
encoder. Keep vector, harmonic and symmetric readouts as separate comparisons.

1. Add spatially sampled **training-source** centers at declared distance bands,
   keeping the Al64 source split and all original benchmark rows unchanged.
   Train with distance likelihood, and explicitly distinguish liquid-only
   approach evaluation from arrival/interior evaluation. This addresses the
   new-center distribution and endpoint mismatch without collecting simulations.
2. Declare the warning task before fitting: for example, predict proximity
   within 20 Å with the fixed causal alarm rule and calibration-only thresholds.
   Report proper scores, complete warning curves, misses and actual false alarms
   together. Confirm the exploratory improvement under that locked protocol.
3. Separate contextual crystal detection from warning with no crystalline atoms
   in any input. Build matched clear-input opportunities and near-miss/tangential
   controls; keep PTM as a label-side diagnostic. This directly addresses whether
   liquid structure contributes beyond direct observation of a crystal.

No new training or Slurm jobs were launched for this analysis. All diagnostic
score radii are retained; the original report and model artifacts are unchanged.

## Reproduction and artifacts

Use conda `pointnet-torch214`:

```bash
python -m src.research.spatial_approach.review \
  --run /work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926 \
  --output /work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926-review
```

- [Comparison figure](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926-review/plots/comparison.png)
- [Paired source comparisons](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926-review/tables/paired-nll.csv)
- [Every alarm radius and model](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926-review/tables/alarm-radius.csv)
- [Metric definitions](../../docs/metrics/spatial_approach_review.md)
- [Original protocol](README.md)
- [Original results](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/RESULTS.md)
