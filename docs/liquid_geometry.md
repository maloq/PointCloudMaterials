# Running the liquid geometry study

Use conda `pointnet-torch214`. The implementation is
`src/research/liquid_geometry/`; the active recipe is
`configs/analysis/liquid_geometry_20260922.json`. Existing frozen populations and
features are reused. No simulations, encoder training, or forecast-head training
are required for this diagnostic study.

From the repository root:

```bash
python -m src.research.liquid_geometry.study prepare --config configs/analysis/liquid_geometry_20260922.json
python -m src.research.liquid_geometry.study model --config configs/analysis/liquid_geometry_20260922.json --model hot-control
python -m src.research.liquid_geometry.study latest --config configs/analysis/liquid_geometry_20260922.json
python -m src.research.liquid_geometry.study checkpoints --config configs/analysis/liquid_geometry_20260922.json
python -m src.research.liquid_geometry.study report --config configs/analysis/liquid_geometry_20260922.json
```

Run `model` once for each named recipe entry. Preparation validates the complete
population and all feature/forecast inputs. Model workers verify the frozen
population/configuration and feature hashes. The latest stage verifies its own
producer-specific caches and saved predictions. Checkpoints performs native
CPU preflight then sequential GPU inference; native source and sampled-array
checksums must agree. Failures retain a stage-specific traceback JSON.

Slurm launch scripts are generated in the run's `technical/slurm/`, and the
submission receipt records their job IDs and dependencies. They execute copied
source/configuration in `technical/code/`, with frozen machine paths, rather than
live edited research code. CPU model workers use an array; latest paired analysis
is a separate CPU job; the early/late audit uses one RTX6000PRO GPU. A dependent
CPU report runs after all jobs terminate and marks missing primary encoders
explicitly. Component failures remain visible in their logs and failure files.

Completed outputs live under the configured WORK analysis root, exposed through
`output/representation_audit/liquid-geometry-20260922/`. Human figures and metric
CSVs are under `plots/` and `tables/`; logs, frozen checkpoint copies and arrays
are under `technical/`. `tables/METRICS.md` and `technical/metric-contract.json`
preserve the exported metric definitions and implementation hashes.

The [scientific protocol](../experiments/liquid_geometry_20260922/README.md) keeps
the three cohort protocols distinct. Historical exports must retain their
definitions if the maintained implementation changes later.
