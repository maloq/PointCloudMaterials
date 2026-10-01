# Birth prediction leakage and important structural features

The September 30 audit found no direct target or future-coordinate leak in the
consumed observations. It did identify a substantial limitation in the research
question: examples are retrospectively localized and aligned to known future
births. The existing scores measure separation of selected future birth sites
from matched persistent-liquid controls. They do not establish prospective
nucleation risk or the ability to find an unknown birth site anywhere in liquid.

**Reassessment of the one-frame claim:** independent fixed-test results do not
establish useful one-frame prediction. Flat-looking AP curves must not be read
as evidence that history is unnecessary. One-frame proper-score improvements
over the prior have intervals spanning zero for all four fitted model families.
The observed effects and limitations are detailed below.

The most consistent useful descriptor groups are **local void topology and bond
orientational order**. Their predictive contribution survives comparisons inside
the same source and observation time. This evidence supports retaining those
descriptors as diagnostic controls, without claiming a causal nucleation
mechanism or selecting new models on these inspected test results.

## Literature and the audit question

Leakage includes dependence on information unavailable under the intended
prediction setting, including dependence introduced by data construction. A
chronological feature array can therefore be clean while the evaluation population
still relies on an oracle that supplies the location or timing of an event.
This is the distinction applied here, following the leakage taxonomy of
[Kapoor and Narayanan, 2023](https://arxiv.org/abs/2207.07048).

Case/control studies also alter the probability population. Evaluation requires
sampling-design weights if the desired quantity is performance in the original
cohort. Our inverse-event weights give equal mass to births; they are **not**
inverse inclusion probabilities and do not recover natural nucleation incidence.
The distinction is supported by
[Rentroia-Pacheco et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38760688/).

Correlated descriptors can substitute for each other, so an individual feature's
tree attribution is not its unique information content. Conditional-permutation
work explains the importance of controlling the donor population, and model
reliance work distinguishes a model's dependence from the importance of a
physical variable across all useful predictors.
[Strobl et al., 2008](https://link.springer.com/article/10.1186/1471-2105-9-307),
[Fisher, Rudin and Dominici, 2019](https://www.jmlr.org/papers/v20/18-760.html).
Our shuffling conditions on the matched set, not every other descriptor; it is
not an implementation of fully conditional variable importance.

Bond orientational precursors have been observed in hard-sphere simulations,
where orientational ordering can precede translational crystalline order. This
provides a plausible interpretation of useful sub-PTM order information; it
does not validate the same mechanism for Lee-MEAM Al.
[Russo and Tanaka, 2012](https://www.nature.com/articles/srep00505).

## What was checked

The audit follows the repository producers from source trajectories through
row sampling, patch construction, descriptor export, temporal packet assembly,
readout fitting and calibration. It reuses the original 150-source release and
all 1,475 eligible histories, including 200 original test rows from 15 births
in 11 sources. The 77 sources with eligible rows contribute 11,793 unique input
patches; zero-yield sources retain their original source roles.

| Route | Evidence and interpretation |
| --- | --- |
| Future frames in inputs | Every observation is reconstructed from its actual tracked atom and past frame. Histories end before local appearance and dropping newest frames preserves the common start. |
| Visible PTM crystal | Full-cell PTM is inspected over every observed 8 Å sphere, including atoms outside the nearest-80 patch. No crystalline observation was found among the 11,793 patches. |
| Labels, frame IDs or outcome metadata | Reversing metadata arrays while preserving observation indices leaves the predictor packet exactly unchanged. No labels, timestamps, temperature, atom/source IDs or event metadata enter its numerical columns. |
| Source or ancestry crossing roles | All 150 recorded melt lineages have one role. Exact coordinate/descriptor duplicates crossing roles are absent. This does not certify missing ancestry or near duplicates. |
| Descriptor calculation | Sixteen input patches, four per role, are independently recomputed using the geometry-only producer and actual descriptor names. Numerical replay allows float32 reduction differences at relative/absolute tolerance 1e-6. |
| Linear scaling | Fitting mean and scale are estimated on readout-training rows; validation chooses regularization by NLL. |
| Boosting selection | Validation-source Logloss chooses tree count. Test rows do not select iterations. |
| Calibration | A separate calibration-source population fits the monotone logit map and false-alarm threshold. |
| Frozen model replay | All 36 explained models reproduce their saved raw and calibrated predictions; maximum observed probability error is approximately 2e-16. |
| Neural pretraining | Rich and VICReg pretraining metadata overlaps all 90 original training lineages and none of the original selection/calibration/test lineages. Readout CV is conditional on previously exposed encoder sources; fixed test remains the unseen-source encoder comparison. |

The audit is limited to this retained producer and release. It does not certify
every historical crystallization experiment or every potential's labels.

## Future information in sampling

There are four deliberate uses of future observations in example construction:

1. Positive center candidates are chosen within 8 Å of the established cluster's
   **future centroid**. Input coordinates are centered on the tracked atom's
   actual past position, so the future centroid is not a numerical feature.
   Nevertheless, the site-selection oracle would be unavailable in a blind scan.
2. The original endpoint is one frame before **future first local PTM appearance**.
   Its time must be discovered retrospectively. A deployed predictor would receive
   arbitrary currently liquid observations instead.
3. Negatives must remain crystal-free through the positive's confirmation,
   with at least 6 ps follow-up. This excludes ordinary at-risk sites that change
   later and can select unusually persistent liquid environments.
4. Positives must eventually meet the isolated-establishment criterion:
   64 connected atoms with three-frame confirmation. PTM flickers that never
   establish and other crystallization origins are not represented by the same
   positive label.

These choices are valid for the stated retrospective discrimination assay. They
become evaluation leakage or population mismatch if its scores are interpreted
as blind prospective prediction. The future-dependent inclusion rule is identical
at every truncated endpoint, so a persistent site-associated signal is a plausible
explanation for some flat lead curves. That explanation remains a hypothesis.

## Which structural information the models use

We explain retained rich-descriptor boosting and linear readouts at all eight
fixed-test endpoints, plus full-history and one-frame readouts in all five outer
CV folds. No encoder or predictor is retrained. TreeSHAP provides descriptive
logit contributions, using CatBoost's
[documented additive attribution](https://catboost.ai/docs/en/concepts/shap-values).
Linear contributions use the fitting-standardization reference. The principal
check jointly shuffles a descriptor group across each set of one case and four
controls, including every frame and its derived summaries. Source, endpoint and
matching metadata stay fixed. It uses 32 permutations and 2,000 whole-source
bootstrap draws, conditional on the fitted readouts.

With all eight frames, the calibrated NLL increase under matched shuffling is:

| Removed information | Boosting test increase and 95% interval | Boosting CV increase and 95% interval | Linear test increase | Linear CV increase |
| --- | ---: | ---: | ---: | ---: |
| TDA topology | 0.00428 [0.00102, 0.00869] | 0.00598 [0.00183, 0.01039] | 0.01253 | 0.01635 |
| Bond orientational order | 0.00320 [0.00030, 0.00687] | 0.00389 [0.00158, 0.00630] | 0.00889 | 0.00661 |
| Geometry | 0.00373 [−0.00016, 0.00784] | 0.00085 [−0.00156, 0.00315] | 0.00138 | −0.00041 |
| CNA fingerprints | 0.00128 [−0.00011, 0.00248] | −0.00005 [−0.00083, 0.00074] | 0.00071 | −0.00307 |

The topology and bond-order intervals exclude zero for both models and both
populations. This is exploratory evidence without multiplicity correction.
Negative importance means the intervention improved that particular frozen
readout; it does not prove that the physical descriptor has no useful information.
Effects are not additive, and correlations with unshuffled groups are broken.

Within topology, the **80-point H2 cavity descriptors** are the most consistently
useful subgroup. Boosting loses 0.00335 test NLL and 0.00434 CV NLL after their
matched shuffling. The 80-point H1 loop descriptors also help CV. Within bond
order, the **l=6 invariants and coherence statistics** consistently help both
linear and boosted readouts.

TDA birth, death and lifetime refer to **geometric filtration scales**, not
physical trajectory time. A persistence-lifetime feature is not itself a
measurement of how long an atomic arrangement survives in ps.

Individual descriptors recurring near the top of the fitted-model attribution
rankings include:

| Descriptor | Interpretation and limits |
| --- | --- |
| `tda/n80_h2_image12`, `image13`, `loglife_sum` | Smoothed persistence features of voids in the 80-point local cloud. These depend on observed geometry and finite patch boundaries. |
| `tda/n80_h1_loglife_sum`, `entropy` | Loop lifetime strength and dispersion. |
| `bond_order/l4_qbar`, `l6_qbar` | Norms of locally averaged spherical-harmonic bond order. The producer averages over the center and twelve local neighbors. |
| `bond_order/l6_coherence_std`, `l6_neighbor_q_std` | Variation of neighboring l=6 orientations or local order strength. |
| `geometry/angle_n12_1`, `angle_n12_2` | First-shell angle bins. Their large individual SHAP values do not imply strong unique geometry-family information; related descriptors can substitute. |

No single descriptor dominates: the leading boosted descriptor contributes about
2% of total absolute calibrated-logit attribution in either population. Ranking
importance is diffuse and model-dependent. We have not shown that any descriptor
causes nucleation, that larger values always favor birth, or that this ordering
transfers to other materials.

## How much does observation timing matter

The independent fixed-test boosting results are:

| Input | AP | AUROC | Calibrated NLL (lower is better) |
| --- | ---: | ---: | ---: |
| Constant 20% prior | 0.200 | 0.500 | 0.50040 |
| Eight frames | 0.292 | 0.661 | 0.49126 |
| Oldest frame only | 0.256 | 0.525 | 0.49822 |

One-frame AUROC has a source-bootstrap 95% interval of [0.417, 0.662], and
AP has [0.180, 0.469]. Its NLL difference from the prior is
−0.00219 [−0.01076, 0.00298]. These results do not establish single-frame skill.
AP above the sampled prevalence alone is insufficient; finite-sample random
ranking also requires a design-aware null, which this audit has not fitted.
The fixed test contains only 15 births in 11 sources, with multiple correlated
center windows per birth. More windows do not create more independent births.

Flat aggregate AP does not establish timing independence. In a matched-set
diagnostic, boosted scores rank a case above a randomly chosen one of its four
controls with probability **0.654** using eight frames ending 0.75 ps before
appearance, versus **0.499** using only the oldest frame, 6 ps before appearance.
The paired difference is 0.156 with a source-bootstrap interval
**[−0.003, 0.333]**, so this small test population does not establish the magnitude
of the decline precisely. CV yields 0.594 versus 0.542, with difference interval
[−0.001, 0.109]. Random ranking is 0.5. The apparent decline is not evidence for
no timing effect, but neither interval provides a decisive separation.

The separate VICReg-linear result is stronger on proper predictive likelihood:
eight-to-one truncation worsens test NLL by 0.01720 [0.00224, 0.03157] and CV NLL
by 0.01109 [0.00509, 0.01703]. These are the existing paired comparisons, not new
fits. Endpoint lead and history span change together in this experiment.

Full-history boosting assigns about **31%** of its absolute temporal attribution
to the last two chronological frames on fixed test and 27% in CV. This is model
attribution, not a causal estimate of the value of adding those frames. Calibration
also flattens likelihood differences: its full-history logit slope is 0.257,
compressing test probability standard deviation from 0.070 to 0.020. This does
not explain flat AP, because the monotone probability map preserves ranking.

### What the truncation experiment can and cannot identify

The eight-frame observation covers −6 to −0.75 ps relative to first local PTM
appearance; the one-frame observation contains only −6 ps. Truncation changes
history length, endpoint lead, feature dimension and the separately fitted
readout. It is not a controlled comparison of snapshot versus history at the
same prediction time.

There are 442 rich descriptors per frame. The actual producer concatenates
frames, mean, standard deviation and last-minus-first change. Eight frames
therefore produce 4,862 columns. One frame produces 1,768 nominal columns but
only 442 distinct informative values: the mean duplicates the snapshot and the
standard deviation and change are zero. With only 59 training births, additional
inputs can increase estimation difficulty. An ideal predictor can ignore
unhelpful history; a finite fitted predictor need not achieve that ideal.
Both neural representations are computed independently for each frame before
this temporal packet is assembled. Explicit velocities and tracked-atom
displacements are absent. Failure to exploit motion is therefore a possible
model limitation, not an established absence of dynamical information.

The event clock also needs precise interpretation. `appearance_frame` denotes
the first local PTM-crystalline observation found in the bounded pre-establishment
search, associated with an atom of the eventual birth core. It does not establish
the first persistent growing cluster or crossing of a critical nucleus size.
Across the 40 positive fixed-test rows, the full-history endpoint precedes the
64-atom establishment frame by **11.25–24 ps**, with an unweighted row median of
**19.125 ps**. No positive test row establishes within 3 or 6 ps of that endpoint.
These values are obtained directly from
`0.75 * (rows['birth_frame'] - rows['end_frame'])`, restricted to
`role == 'test'` and `label == 1` in the retained dataset's `rows.npz`.
Thus the reported appearance lead is not a 3/6-ps established-nucleus forecast
horizon. Early isolated PTM detections need not mark committed growth.

A single configuration can in principle carry information about an event's
probability without determining its realized outcome. Configuration-dependent
dynamical propensity provides one established example of this distinction;
it does not demonstrate local nucleation predictability in our Al data.
[Widmer-Cooper and Harrowell, 2005](https://arxiv.org/abs/0901.3759).
Nucleation-precursor evidence is also system- and definition-dependent: a study
of hard and charged colloids found no precursor using its structural and
machine-learning analyses.
[de Jager, Smallenburg and Filion, 2023](https://arxiv.org/abs/2306.05886).
Our earlier discussion of persistent structural propensity remains a hypothesis,
not an explanation demonstrated by these fitted models. Future-based case/control
sampling alone is not proof of a leaking numerical feature.

## Experiments needed for a prospective claim

Keep this assay as an annotated-site precursor diagnostic. Build a separate
prospective release from existing trajectories, with centers and observation
times selected **without looking at future birth centroids or appearance times**.
Eligibility should use current/past crystal clearance only. Sample negatives
from the same currently at-risk population, then label every selected row by
subsequent local birth, front arrival, no event or insufficient follow-up. Future
observations may define labels and sustained confirmation; they must not choose
the observed coordinate origin or candidate pool.

Compare fixed-length histories at different leads, and snapshot versus history
at the same endpoint. This separates information loss caused by an earlier
endpoint from information loss caused by a shorter history. Include separately
trained repeated-current-frame and shuffled-order controls with matched input
capacity and fitting rules. Recompute temporal summaries after shuffling;
inference-only shuffling is a diagnostic, not a replacement for these controls.
Keep whole-source
and melt-ancestry roles fixed, train encoders only on their fitting ancestors,
and reserve independent held-out sources for confirmation after this exploratory
feature audit. Use likelihood, calibration and conditional information as primary
checks; keep AP diagnostic. No time or temperature covariates are proposed.

For interpretation, compare declared physical groups with matched drop-group
refits on validation data, and check TDA sensitivity to patch size, neighbor
selection and input noise. Preserve coordinate precision and identical label
definitions. These controls would distinguish robust structural information from
a dependence on finite patch boundaries. None of these new fits or simulations
were started in this audit.

## Saved evidence

The [feature audit bundle](/store/PERSO/vmorozov/experiments/birth_prediction/drop-to-one-cv-20260930/analyses/leakage-feature-audit-v1/)
contains attribution/permutation tables with frozen definitions, PNG/PDF figures,
model replay receipts, raw-coordinate/PTM checks and exact source snapshots.
[Feature groups](/store/PERSO/vmorozov/experiments/birth_prediction/drop-to-one-cv-20260930/analyses/leakage-feature-audit-v1/plots/feature-family-reliance.png),
[individual attribution](/store/PERSO/vmorozov/experiments/birth_prediction/drop-to-one-cv-20260930/analyses/leakage-feature-audit-v1/plots/top-descriptor-attribution.png),
[temporal attribution](/store/PERSO/vmorozov/experiments/birth_prediction/drop-to-one-cv-20260930/analyses/leakage-feature-audit-v1/plots/temporal-attribution.png).
The command and operating details are in [birth prediction](../birth_prediction.md).
