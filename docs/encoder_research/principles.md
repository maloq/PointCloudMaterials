# Research principles

[Handbook](README.md) · [Encoder families](encoders.md) · [Inputs](prediction_context.md) · [Evaluation](evaluation.md)

Reference reviewed against repository commit `05e3e8c` on 11 October 2026. The standing instructions in [AGENTS.md](../../AGENTS.md), dataset contracts and frozen metric definitions govern individual experiments. This page explains the scientific distinctions shared across studies.

## What improvement means

The objective is information useful for understanding and predicting crystallization. Evaluate present structural information, organization of latent distances, response to perturbations and dynamics, and future prediction separately. Clear clusters, high rank, smooth trajectories and low reconstruction error can disagree. There is no universal encoder ranking. A continuous ordering process need not produce separate clusters. [Evidence framework](https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/docs/encoder_research/quality_criteria.md).

## Define the observation before comparing models

Record geometry, spatial support, atom cap, edge cutoff, message-passing halo, normalization, precision, history offsets, velocity and relaxation. Full-cell relaxation can introduce information beyond the final local crop. A training-only teacher differs from preprocessing required at inference. Temporal pairing during training does not make snapshot inference history-dependent.

For new Ni relaxations, use the [default recipe](../../configs/simulation/ni_relaxation_default.json): maximum absolute force component 0.05 eV/Å, float64 minimizer states and float32 force arithmetic. Reuse indexed relaxations and retain historical tolerance/precision identities; the default does not relabel existing minima or targets.

Keep encoder, projector, jointly trained head, frozen probe and forecaster identities separate. Invariant scalar features and typed directional exports expose different information. Document encoder and predictor inputs independently using the [prediction-context contract](prediction_context.md). Model names do not specify these contracts. [Architecture distinctions](https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/docs/encoder_research/encoders.md).

## Training and selection

New experiments use **train and test only** (user instruction, 13 October 2026).
Keep declared training ancestry and pool every other role into test. Do not
create an inner validation/selection/calibration partition. Fit transforms on
train; predeclare hyperparameters, training duration and checkpoint rules
without consulting test outcomes. Original roles remain provenance, and frozen
historical or running protocols keep their actual splits. This default takes
precedence over older examples of multi-split protocols.

Keep two branches distinct:

| Branch | Encoder objective and selection |
| --- | --- |
| Task-supervised | Declared predictive likelihood objectives may use future-outcome labels. |
| Self-supervised | Label-free objectives and selectors; outcome labels cannot determine encoder sampling, selection or promotion. |

Frozen predictive readouts may learn outcome labels in either branch. Report their scores separately from jointly trained heads. AP remains an evaluation diagnostic; do not use AP-specific losses, ranking replay, AP-driven selection/search or AP-fitted ensemble weights. Physical-reconstruction encoder pretraining is retired; physical-information readouts remain diagnostics. Preserve historical artifacts with their actual objectives. [Branch policy](https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/docs/encoder_research/training_branches.md).

New encoders, predictors, probes and baselines exclude explicit temperature, simulation age and time covariates unless the user requests them. Such metadata can organize observations, labels and audits. Current native spatial MACE defaults are geometry-only with one constant atom channel, width/export 128 and batch/microbatch 256. Historical settings remain historical. [Standing instructions](../../AGENTS.md).

## Evidence and interpretation

Use matched initial, constant-risk, persistence and descriptor controls where appropriate. Compare frozen linear and nonlinear readouts, proper predictive scores, calibration and source uncertainty. Measure raw feature spread, physical neighbors, noise response and exact-lag dynamics alongside prediction. An add-back benefit shows that the supplied features help the tested readout. Matched controls are needed to distinguish omitted information, readout limitations and dimension or regularization effects; no benefit does not prove sufficiency. Low loss or smoothness does not establish a Markov state or causal mechanism. [Evaluation guide](evaluation.md).

For descriptor-group clustering accuracy, allow multiple neural clusters to map to the same descriptor group, as requested. Retain the one-to-one assignment only as a separately labeled historical diagnostic. State the mapping/evaluation population and report occupancy and continuous readouts alongside assignment accuracy.

Separate recognition, proximity to an existing crystal, crystal arrival and new establishment. An operational size/persistence threshold does not establish criticality. Retrospectively selected birth sites can have clean past inputs while still depending on future location or timing in sample construction. Such results do not establish prospective discovery of unknown birth sites. Read the [birth and fate summary](evidence.md#birth-and-fate) before making a precursor claim. [Sampling audit](https://github.com/maloq/PointCloudMaterials/blob/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896/docs/encoder_research/birth_leakage_features.md).

## Preserve comparability

Split by independent preparation/source ancestry, including shooting branches, overlapping windows and relaxed descendants. Fit transforms and selection only on their declared roles. Reused test sources remain development evidence. Consult [DATASETS.md](../../DATASETS.md) and the [birth dataset contract](../datasets/nuclei_birth.md): historical source IDs do not imply current raw-data availability.

Compare scores only when populations, rows, targets, horizons, feature stages, readouts and aggregation agree. Frozen `tables/METRICS.md` definitions take precedence over cross-study prose. Preserve negative findings, retired approaches and historical input errata in the evidence index; use the [representation summary](evidence.md#representations) and [forecasting summary](evidence.md#memory-and-context) to avoid repeating already-tested proposals.

## Classify a new finding before changing the method

A code-verified loss or producer definition can motivate a hypothesis without establishing its effect on a trained result. Keep analytical deductions, observed performance and proposed treatments distinct. The [methods page](methods.md) names the assumptions and the [experiment plan](experiment_plan.md) defines the discriminating comparisons.

Record the forecast object (endpoint, joint future or entire emitted path), the target ascertainment rule, and the population average before fitting. Source/ancestor uncertainty, training variation and Monte Carlo resolution are different. Repeated shots, neighboring centers and overlapping windows preserve their dependence and identity; more rows or particles do not automatically provide more independent physical evidence.

Use current data availability, original ancestry roles and exact saved offsets. Keep the consolidated 0.10-ps and native 0.30-ps birth protocols separate; the broad spatial collection does not become motion replay. A diagnostic liquid stratum must not introduce extra filters into the canonical birth catalogue. Missing tracking outcomes remain unknown unless a different operational target is explicitly defined.

Preserve source-linked negative results and exact historical metric exports. A proposed calculation enters the execution system with its own version only when implemented. The documentation update itself does not change source data, metric calculations, trained models or the completion status of an experiment.
