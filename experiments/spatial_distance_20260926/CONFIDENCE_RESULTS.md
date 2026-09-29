# Fixed-confidence detection of an existing crystal

These completed results reuse the original five fitted predictors. They do not
come from the new continuous-distance queue. Probability means **P(distance to
confirmed reference crystal ≤20 Å)** in the table below, with strict threshold
exceedance at **two consecutive observations**. All 4/8/12/20/32 Å events and
one/two-observation rules are available in the [complete CSV](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/analyses/confidence-v1/tables/alarms.csv).

| Predictor | Probability > | Median alarm distance, detected only (Å) | Misses / 495 | Recall ≥12 Å, all paths | False alarms / 292 |
|---|---:|---:|---:|---:|---:|
| geometry_mlp | 0.5 | 0.00 | 448 / 495 | 2.0% | 4 / 292 |
| geometry_mlp | 0.75 | 0.00 | 474 / 495 | 0.4% | 1 / 292 |
| geometry_mlp | 0.95 | 0.00 | 492 / 495 | 0.2% | 0 / 292 |
| mace_local | 0.5 | 0.00 | 359 / 495 | 6.1% | 17 / 292 |
| mace_local | 0.75 | 0.00 | 434 / 495 | 2.8% | 6 / 292 |
| mace_local | 0.95 | 0.00 | 479 / 495 | 1.4% | 1 / 292 |
| symmetric_invariant | 0.5 | 11.88 | 22 / 495 | 46.1% | 13 / 292 |
| symmetric_invariant | 0.75 | 11.12 | 48 / 495 | 36.8% | 8 / 292 |
| symmetric_invariant | 0.95 | 9.04 | 108 / 495 | 16.2% | 1 / 292 |
| vector_messages | 0.5 | 12.77 | 18 / 495 | 54.5% | 14 / 292 |
| vector_messages | 0.75 | 11.66 | 30 / 495 | 43.6% | 4 / 292 |
| vector_messages | 0.95 | 8.84 | 96 / 495 | 20.2% | 0 / 292 |
| harmonic_hierarchy | 0.5 | 12.00 | 25 / 495 | 47.5% | 8 / 292 |
| harmonic_hierarchy | 0.75 | 10.86 | 59 / 495 | 35.6% | 5 / 292 |
| harmonic_hierarchy | 0.95 | 8.82 | 104 / 495 | 18.8% | 1 / 292 |

Medians exclude missed paths and are not guaranteed detection ranges. These are
pooled path counts from 28 held-out sources, one fitted training seed, without
confidence intervals. Far-away controls stay beyond 32 Å; near misses and
tangential approaches are not represented. Changing the distance event changes
the meaning of the probability, so these numbers must always retain “within 20 Å.”

## Does crystal visibility explain the warning?

For vector messages at >0.5, reference crystal is already in the context at
466/477 alarms (97.7%); at >0.75, 457/465 (98.3%); at >0.95, 398/399 (99.7%).
Harmonic context has reference crystal visible at 461/470, 433/436 and 391/391
alarms. Every detected approach for these two models contains some PTM-labelled
crystal at its alarm. This strongly supports **recognition of an existing crystal
in the surrounding patches** as the main interpretation of the warning distances.
It does not demonstrate detection of a completely unseen crystal or nucleus birth.

A label-side alarm that simply checks confirmed crystal presence in the context
at two consecutive points detects 495/495 approaches, median distance **19.12 Å**,
with 0/292 far-path false alarms. The local-only version has median **3.01 Å**.
The oracle uses full-cell confirmed reference membership and is not a deployable
geometry-only predictor or a fair supervised-performance upper bound. By contrast,
any PTM crystal in context gives 285/292 false alarms (97.6%): isolated crystalline
motifs are insufficient to establish proximity to a confirmed crystal component.

These comparisons do not prove that visibility explains all improvements in
distance likelihood. The new queue includes a learned two-flag reference-visibility
control, scored on the same samples, to quantify what distance information that
simplification retains. Clear-context performance also needs positive opportunities;
the historical fixed test subset without context reference crystal has no examples
within 8 Å.

## Reliability of probabilities

On the controlled scan positions, the vector model’s source-weighted precision
for distance ≤20 Å is 95.6%, 97.7% and 99.7% among scores above .5/.75/.95,
respectively. Corresponding harmonic values are 96.9%, 98.2%, 99.8%. These are
pointwise conditional precision estimates in an artificially sampled population,
not path-level guarantees. Their denominators and mean predicted confidences
are in [confidence-reliability.csv](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/analyses/confidence-v1/tables/confidence-reliability.csv).

[Definitions and frozen implementation hashes](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/analyses/confidence-v1/tables/METRICS.md).
