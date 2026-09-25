# Supervised AP3/AP6 execution

**Historical AP-specific workflow — retired 25 September 2026.** Do not launch or resume AP tuning. Use the likelihood-based `supervised_onset_information_v4` workflow and `configs/supervised_onset/information_20260925/campaign.json`. Preserve historical results unchanged.

The active workflow is the [MACE capacity campaign](supervised_capacity.md),
using protocol `supervised_onset_capacity_v3`: small/500k/1M/2M encoders, observed
and relaxed inputs, and no temperature or explicit time covariates.

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m src.research.supervised_onset.campaign check --config configs/supervised_onset/capacity_20260925/campaign.json
python -m src.research.supervised_onset.campaign submit --config configs/supervised_onset/capacity_20260925/campaign.json
python -m src.research.supervised_onset.campaign collect --config configs/supervised_onset/capacity_20260925/campaign.json
```

Use conda `pointnet-torch214`. See the campaign guide for resource requests,
preflight scope, immutable source, job IDs, outputs and resumption. Each result
links to `technical/prediction-context.json`. The unlaunched geometry-v2 recipe
was superseded by the capacity study.

The [completed six-arm24 September protocol](../experiments/supervised_onset_20260924/README.md)
used temperature and simulation age in prediction heads. Its files live under
`output/encoder_supervised/ap36-large-20260924/`. Inspect its `RESULTS.md`,
`tables/comparison.csv` and dated context erratum. Reproduce/resume that historical
study only with its frozen `technical/code`; the live runner rejects its protocol.
Historical metric snapshots and trained checkpoints remain unchanged.
