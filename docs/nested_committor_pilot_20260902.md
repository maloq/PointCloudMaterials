# Nested-shooting committor pilot

This experiment tests whether the frozen GeoFrameTransformerV2 representation
adds predictive information about crystallization first passage beyond the
collective variable used to construct the shooting campaign.

## Data contract

The authoritative input is
`/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902`.
The fit starts only after the strict campaign `summary.json` reports all 144
branches complete. Only direct `branches/<branch_id>/outcome.json` files with
`state: complete` are read; archived interrupted attempts are not searched.

There are 36 immutable position parents from independent source trajectories.
Each parent has two independently sampled momenta and two independent Langevin
noise futures per momentum. The target is the probability that a trajectory
first persistently reaches the crystal basin before the liquid basin within the
temperature-specific maximum duration. Censored branches are reported but are
not silently converted to either class.

The manifest's source-run split is preserved:

- `optimization`: grouped source-run cross-validation and coefficient fitting;
- `model_selection`: an optimization-only model is reported here, then these
  parents are added for the final refit;
- `final_validation`: used only for the final report.

## Structure representation

The encoder is the frozen checkpoint
`GF_V2_FACTOR_SN_GROUPED_SCRATCH_G0.1_N0.1_B16384-epoch=34.ckpt`, using its
checkpoint-selected VICReg-projector representation. Point clouds contain 160
neighbors and use the same 9.192189 A normalization radius as the earlier
shooting experiments.

Uniform center sampling is inappropriate here: a 20-atom nucleus occupies only
0.028% of a 70,304-atom system. Instead, PTM and connected-cluster analysis are
recomputed at each parent, without future labels, and deterministic centers are
drawn from three regions:

1. the largest crystalline cluster;
2. its nearest non-cluster interface atoms;
3. a reproducible bulk-background sample.

GeoFrame token PCA is fit using only the permitted training parents. Per-region
means and standard deviations are the fixed-size parent representation.
Rotation-invariant local velocity descriptors are constructed separately for
the two momentum samples and summarized by the same regions.

## Models

All heads are regularized logistic first-passage classifiers. Regularization and
PCA dimensions are chosen by grouped cross-validation over source-run IDs in the
optimization split.

- temperature only;
- temperature, initial largest-cluster size, and crystalline fraction;
- collective variables plus per-region initial-force statistics;
- GeoFrame regional structure;
- GeoFrame plus collective variables;
- GeoFrame plus collective variables and initial-force statistics;
- all preceding position features plus momentum-specific invariant velocity
  descriptors.

For the collective-variable, GeoFrame-plus-collective-variable, and phase-space
models, a second calibrated variant adds a fixed Jeffreys Beta(1/2, 1/2)
pseudocount to each independently conditioned state. This prevents four
same-basin shots from being treated as proof of an exact zero or one committor.

The primary comparison is branch Brier score and negative log likelihood on
transition-candidate parents in `final_validation`. Controls are also reported,
but are not allowed to make the primary result look artificially easy.

## Reproduce

```bash
cd /home/infres/vmorozov/PointCloudMaterials
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
PYTHONPATH=. python scripts/run_nested_committor.py \
  --config configs/nested_committor_geoframe_v2_20260902.yaml
```

Outputs are written to
`/home/ids/vmorozov/experiments/nested_committor_geoframe_v2_20260902` and
include the immutable feature cache, resolved configuration, metrics, plots,
predictions, and explicit PCA/scaler/logistic parameters.

## Results

The strict campaign completed all 144 branches from 36 independent position
parents. There are 131 resolved first passages (29 liquid and 102 crystal) and
13 censored branches. Only 2 of the 28 transition parents with at least one
resolved child have empirical resolved committor strictly between 0.2 and 0.8.
The effective sample size for a structure model is 36 parents, not 144 branches;
the final transition evaluation contains only 4 parents and 14 resolved futures.

Primary final-validation results on transition candidates are:

| Model | Branch Brier | NLL | ROC AUC | Parent pB MAE |
| --- | ---: | ---: | ---: | ---: |
| Temperature / training prevalence | **0.1271** | **0.4253** | 0.792 | 0.2305 |
| Collective variables | 0.1424 | 0.7122 | 0.917 | 0.1498 |
| Collective variables, Jeffreys | 0.1323 | 0.4430 | 0.917 | 0.2410 |
| Collective variables + force | 0.1456 | 0.4712 | 0.333 | 0.2675 |
| GeoFrame regional structure | 0.1652 | 0.6660 | 0.917 | 0.2076 |
| GeoFrame + collective variables | 0.1620 | 0.6584 | 0.917 | 0.2024 |
| GeoFrame + collective variables + force | **0.1340** | 0.5008 | **0.917** | **0.1219** |
| Phase space | 0.2419 | 0.6706 | 0.708 | 0.2570 |
| Phase space, Jeffreys | 0.1967 | 0.5842 | 0.542 | 0.3238 |

The temperature model selected extremely strong regularization and predicts
approximately 0.789 for every final state: it is effectively the resolved
training prevalence. It wins branch Brier because 12 of 14 final transition
outcomes are crystal. It is not a useful reaction coordinate.

The best structured model, GeoFrame + collective variables + forces, does show
useful ordering: its AUC is 0.917 and its parent-level committor MAE is almost
half that of the prevalence predictor. It predicts the three crystal-only final
transition parents accurately, but predicts 0.032 for the sole mixed parent
whose empirical pB is 0.5. Consequently it does not beat the prevalence model
on calibrated Brier or likelihood. With four final transition parents, the
parent bootstrap interval on its Brier difference from the prevalence model
includes both meaningful improvement and degradation; this pilot cannot
resolve a small gain.

Momentum is not the missing ingredient in this dataset. Only 3 of 63 resolved
same-momentum thermostat pairs disagree, and only 1 of 34 parents has different
empirical pB between its two momentum samples. The phase-space models therefore
mostly fit finite-sample noise and are worse on held-out data.

The most important failure is boundary coverage. At 400 K, the optimization
parents with initial clusters of 38 and 46 atoms are fully censored, so the
binary likelihood receives no labels between the 20-atom liquid outcome and
the 64-atom crystal outcome. The sole 29-atom mixed parent is in final
validation. The model-selection split is also entirely crystal among its 14
resolved transition outcomes, so it cannot test calibration.

This data is valuable as a campaign-design pilot and as evidence that regional
GeoFrame features contain some ordering information. It is not yet sufficient
to train a serious calibrated committor model. The next campaign should:

1. concentrate structurally diverse parents near the observed boundary bands
   (approximately 25--55 atoms at 400 K, 25--40 at 450 K, and 18--30 at 500 K);
2. continue or lengthen 400 K futures so metastable parents are not discarded
   from a binary analysis, while retaining censoring metadata;
3. acquire at least tens, preferably 100 or more, independent position parents
   per temperature before increasing model capacity;
4. use 8--16 futures only for boundary parents, since additional shots far from
   the boundary add little information;
5. construct source-run-held-out splits that each contain liquid, mixed, and
   crystal outcomes at every temperature.

A censor-aware competing-risks/survival head is the next appropriate model for
the existing 13 censored branches. A nonlinear regional token aggregator should
wait until the campaign supplies substantially more independent parents.
