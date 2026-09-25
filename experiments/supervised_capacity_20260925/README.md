# Does larger native MACE improve supervised 3/6 ps onset prediction?

**Historical AP-specific workflow — retired 25 September 2026.** Do not launch or resume AP tuning. Use the likelihood-based `supervised_onset_information_v4` workflow and `configs/supervised_onset/information_20260925/campaign.json`. Preserve historical results unchanged.

The preceding study retained a 32-channel pilot architecture. Its size was not
established as optimal for the latest onset task. Compare three requested
capacities with a newly trained small control, without temperature or simulation
age. This is the crystallization-supervised branch.

| Capacity | Channels | Encoder parameters | Encoder + hazard head |
| --- | ---: | ---: | ---: |
| Small control |32|79,232|96,389|
| Approximately 500k |112|503,552|520,709|
| Approximately 1M |168|1,029,056|1,046,213|
| Approximately 2M |240|1,981,184|1,998,341|

Counts exclude buffers and optimizer states. All models retain two MACE layers,
correlation order two, angular degrees through two, the same radial network and
a128-D export. The old export concatenated pooled channels with a learned block,
requiring twice the channel width to be smaller than the export. The new export
uses an orthogonally initialized trainable linear projection plus an MLP residual
of the normalized pooled features. All four sizes use this rule. Thus the new
small control has79,232 encoder parameters rather than the historical62,784.

Each size trains **O-AP36** on observed geometry and **R-AP36** on relaxed geometry:
eight fits, one seed (20260924). Both optimize natural-distribution hazard NLL plus
full-population differentiable AP weighted2:1 for3/6 ps. Event oversampling is
importance-corrected. All sizes use batch256, microbatch64, encoder/head learning
rates1e-4/5e-4, warmup128, weight decay1e-5 and the same decay after update2048.
All encoder parameters receive gradients.

The target is8192 updates per fit, with selection every256 updates, also scoring
update0. AP3 is primary; AP6 breaks selection ties. AP6-selected checkpoints and
fresh embedding-only linear/MLP probes are separate secondary rows. Only selection
sources choose checkpoints, continuation order and the two-model ensemble.

Reuse31,609 prospective observations from150 independent sources:90 training,
15 selection,15 calibration and30 historical test sources. Existing patches have
at most80 atoms within8 A including the center,5 A edges and no extra halo.
Relaxation uses the full periodic current cell before cropping. Inputs are a
single geometry observation: no velocity, history, teacher, temperature or age.
Timestamps/temperature remain audit and label metadata. Every run logs prediction
context and verifies its actual parameter count before fitting/inference.

Each capacity receives an independent8.5-hour budget: up to45 minutes per arm for
the initial2048-update screen, then both continue toward8192 with remaining
training time shared;75 minutes are reserved for evaluation. A time-limited fit
must report its actual updates and cannot be described as an equal-update
comparison. Completed update budgets end early.

Report joint-head/probe AP3/AP6, calibration/Brier/alarm metrics, source-bootstrap
intervals, covariance spectra and normalized input-noise response. This cache has
12 ps anchor cadence and cannot measure0.75 ps movement; no substitute lag is
used. Source intervals do not measure seed uncertainty. The reused test sources
and one seed limit generalization claims. Compare sizes within the same input
arm and against the **new metadata-free small control**. Historical conditioned
AP values are contextual references, not matched baselines.

[Recipes](../../configs/supervised_onset/capacity_20260925/)
· [Execution](../../docs/supervised_capacity.md)
· [Prediction-input policy](../../docs/encoder_research/prediction_context.md)
