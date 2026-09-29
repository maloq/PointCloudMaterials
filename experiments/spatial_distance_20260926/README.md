# Distance to an existing crystal in fixed snapshots

[Completed direct-distance results](RESULTS.md): five 16-epoch fits and all
evaluations finished in about 23 minutes. Context improves over local geometry,
with strong visibility-only control performance and larger errors on scans.

Question: how confidently and how far away can a local observation locate an
existing crystal, and how much of this information is explained by crystal
already appearing in the wider spatial input?

The historical six-bin distance models are reviewed at fixed probability
thresholds >0.5/>0.75/>0.95 for every distance event at 4/8/12/20/32 Å. Report
misses, conditional warning distance, all-path early recall, false alarms,
probability reliability and label-side visibility. The two-observation rule is
primary; instantaneous alarms are a declared sensitivity analysis.

The new controlled experiment fits direct continuous-distance readouts using a
zero-inflated lognormal distribution right-censored at 64 Å. Four predictors
(local frozen MACE, vector messages, harmonic hierarchy, symmetric invariant)
share the previously trained observed MACE checkpoint. A fifth, label-assisted
control sees only local/context reference-crystal presence. This is supervised
distance-head training; the encoder is frozen, not retrained.

The fixed Al64 source roles and all original evaluation samples are unchanged.
Training and selection additionally receive 16 uniform atom centers per original
frame; each population contributes half the likelihood mass, with equal source
weights inside each half. This exposes the readouts to crystal interiors that
were absent from the historical liquid at-risk population. It does not make the
controlled scan distribution identical to the fitting distribution; probability
reliability is reported separately. No new simulations are generated.

All five models use one seed, 16 epochs, minimum selection epoch 12, source/mix
weighted censored likelihood, batch/microbatch 256 and mandatory online W&B.
Geometry-only patch encoding, shared patch predictors, no privileged central
encoder and no explicit time/temperature input. Frozen input records distinguish
the deployable geometric treatments from the label-side visibility control.

Recipe: [spatial_distance_20260926.json](../../configs/analysis/spatial_distance_20260926.json).
Metrics: [distance](../../docs/metrics/spatial_distance.md) and
[confidence](../../docs/metrics/spatial_confidence.md).

```bash
python -m src.research.spatial_distance.confidence --run /work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926 --output /work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/analyses/confidence-v1
python -m src.research.spatial_distance.queue prepare --config configs/analysis/spatial_distance_20260926.json
python -m src.research.spatial_distance.queue worker --config configs/analysis/spatial_distance_20260926.json
```

Run artifacts live at `${storage:analysis}/spatial_distance/al64-uniform-20260926`.
Code/configuration are frozen before detached submission. Generated geometry
resides in the external training-cache storage; encoder feature caches use the
existing six-entry policy with active leases. Checkpoints/predictions are durable.
