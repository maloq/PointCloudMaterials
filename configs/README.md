# Configuration index

- Top-level YAML: reusable Hydra training composition. Defaults and config names are
  part of the training API; keep them resolvable together.
- `analysis/`: standard checkpoint-analysis templates.
- `data/`: dataset/loader composition; see its README.
- `experiments/`: reusable experiment-plan templates (model, objective and data sweeps).
- `simulation/`: maintained simulation protocols and source definitions; see its README.

Historical standalone shooting and GeoFrame configs are in the STORE experiment
records; start at
[shooting ablations](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/shooting_ablation_20260901/README.md) or
[GeoFrame continuity](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/geoframe_continuity_20260905/README.md).
Hydra defaults and simulation configs retain their existing paths. Recent submitted
queue specifications also retain their paths until scheduler quiescence is verified.

For a new run, override dataset, checkpoint, seed, temperature, horizon and output
path using the existing method's configuration/arguments. Do not copy a runner.
Use `output/<question>/<run-name>/` for new explicit result directories; see the
[record and result layout](../docs/research_layout.md).

Two invalid historical templates were retired on the cleanup branch. Their exact
contents and failure reasons are recorded in [the cleanup report](../docs/repository_cleanup_20260912.md);
training requires `--config-name NAME` instead of guessing a replacement default.

Machine-only locations and execution settings live in ignored `machine.local.yaml`; examples are in `machines/`. `datasets.json` supplies stable dataset IDs. See [portability](../docs/portability.md). `vicreg_mace_full_cpu` selects the CPU-compatible MACE execution path.
