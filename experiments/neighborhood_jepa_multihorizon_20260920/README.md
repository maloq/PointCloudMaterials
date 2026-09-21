# Does longer-horizon JEPA embedding prediction improve the local state?

Add prediction of the same tracked center's invariant128 and equivariant120
embeddings at3,6,9ps to the existing.75ps objective. The future teacher states
come from the same jointly trained MACE encoder, with gradients on both sides.
The predictor sees current features, temperature and lag; future graphs never
enter its inputs. Snapshot deployment remains unchanged.

Run three matched MLP-projector/order-anchored arms: SIGReg, VICReg variance/
covariance, and EpiJEPA-inspired regularization. Compare with the corresponding
single-horizon arms in `neighborhood_jepa_regularization_20260920`.

- Same development-selected width64 warm checkpoint, seed20260920, B512,
 768 updates, encoder/head peak LR.0001/.001 and cosine/warmup schedule.
- Future-center latent family total weight stays.1 and fixed physical/TDA future
 weight stays.25, each divided equally over.75/3/6/9ps. Present and short-lag
 spatial-neighbor family weights are unchanged. No extra predictor parameters.
- Seventeen encoded views per anchor: the original14 plus three future centers.
 Missing late-trajectory targets are masked; current/short-lag sampling is unchanged.
- Physical85 and instantaneous TDA144 remain fixed anchors decoded from predicted
 invariant states. They supplement JEPA prediction, rather than replacing it.
- Use all existing32768/480 training/development anchors,90/15 independent native
 Al Lee-MEAM lineages. At3/6/9ps train coverage is32768/32448/32112; all480
 development anchors have every horizon. No new MD.

Report per-horizon invariant/equivariant prediction errors relative to persistence,
decoded physical/TDA errors, present information retention and frozen crystallization
NLL/AP/calibration/timing/spatial scores. Latent raw MSE alone does not compare
representations fairly. Checkpoint selection remains present physical+.25TDA,
matching the single-horizon comparison. One seed and the historical reused assay
limit the strength of conclusions. Crystallization test metrics never select fits.

[Metric definitions](../../docs/metrics/neighborhood_jepa_multihorizon.md)
· [Queue operation](../../docs/neighborhood_jepa_regularization.md)
· Configuration: `configs/neighborhood_jepa/multihorizon_20260920/study.json`.
