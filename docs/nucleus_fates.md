# Nucleus fate audit

Use conda `pointnet-torch214`. Reuses the completed original Al64 PTM cache and
cluster graphs, with original trajectories used only for selected membership
reconstruction and spatial checks. No GPU, simulation or W&B run is needed.

```bash
python -m src.research.crystallization_origin.fates prepare \
  --config configs/analysis/nucleus_fates_20261002.json
python -m src.research.crystallization_origin.fates submit \
  --config configs/analysis/nucleus_fates_20261002.json
```

The queue freezes code/configuration and runs eight independent one-CPU Slurm
lanes over 150 sources, followed by a dependent collector. Each source has a
verified completion receipt and resumes without recomputation. Output is stored
outside the repository under `${storage:training_storage}/nucleus_fates/al64-20261002`.
Jobs and stage receipts are in `technical/launch.json` and `technical/worker-*.json`.
The four-hour worker limit is a limit, not a runtime estimate.

Submitted 2 October 2026: CPU array `1018485` (eight lanes), dependent collector
`1018486`. Real-source checks on sources 860 and 890 completed with the same
frozen contract and are reused by the array. Job state belongs to the receipts,
not this static execution record.

Both jobs completed successfully. All 150 source receipts and the final collector
receipt are complete; the 1,475-row additional-label export preserves every original
sample identity. See the scientific protocol for the completed counts.

`source --source-id ID` runs one actual source using the same prepared contract;
its completed result is reused by the queue. Resume workers/collector using the
frozen code directory and its `config.json`.

The analysis exports `analyses/fates-v1/tables/birth-row-fate-labels.csv` with
stable existing row IDs and unchanged binary labels. It applies to both original
and relaxed inputs. Negative controls are explicitly not assigned their matched
positive's fate. Active predictor jobs are unaffected. New transient episodes
form an audit inventory; crystal-free classifier inputs have not been built.

[Scientific protocol](../experiments/nucleus_fates_20261002/README.md) ·
[Metric definitions](metrics/nucleus_fates.md).
