# Shooting representation and prediction ablations — September 2026

Research question: which geometry, temporal history and training objectives predict
branch outcomes and structural changes on the recorded Al shooting ensembles?

This record groups the historical standalone configurations formerly at the root of
`configs/`. Each YAML in `technical/` retains its original dataset, checkpoint, split,
seed, horizon and output directory. The original names carry the exact run dates.
Scientific protocols remain distinct, including fixed-horizon and exact-continuation
methods. This is a configuration relocation, not a new experiment or a recomputation.

Reproduce the geometry ablation, for example, from the repository root:

```bash
conda run -n pointnet python scripts/run_shooting_ablation.py geometry --config experiments/shooting_ablation_20260901/technical/shooting_geometry_ablation4_fixed_geoframe_v2_20260901.yaml
```

Use the matching family/method from [maintained commands](../../scripts/README.md)
for the other configs. Implementations are in `src/temporal_vamp/`. Output locations
and reference checkpoints are declared explicitly in each YAML; most shooting
artifacts are under `/home/ids/vmorozov/experiments/`. Findings and prior run provenance
are linked from the [output registry](../../docs/output_registry.md). No new scientific
finding is claimed by this storage cleanup.
