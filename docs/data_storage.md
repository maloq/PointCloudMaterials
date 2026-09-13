# Data and result storage

| Contents | Location on this cluster |
| --- | --- |
| Existing datasets and simulation inputs | `/work/PERSO/vmorozov/{datasets,simulations}` |
| Analysis products | `/work/PERSO/vmorozov/analysis` |
| Training and dataset caches | `/home/ids/vmorozov/training-cache` |
| New simulation runs | `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations` |
| Published simulations and archives | `/store/PERSO/vmorozov` |
| Current research reports and run links | Repository `experiments/` and `output/` |

Machine-specific paths belong in ignored `machine.local.yaml`. Stable dataset IDs
and dependencies are in [configs/datasets.json](../configs/datasets.json).
`python scripts/project.py paths` shows effective roots; `datasets` resolves IDs.
See [portability](portability.md) for setup, verified copies, bundles and publication.

SCRATCH has a 30-day inactivity purge. Publish completed simulations and preserve
stopped failures/restarts on STORE before that deadline. Conversion must verify
checksums and float16 position quantization before removing larger exports;
[trajectory conversion](trajectory_conversion.md) describes the maintained commands.

[Old research is archived on STORE](archived_research.md). The
[September 11–13 retention review](research_retention.md) identifies the live research,
checkpoints and older inputs that must remain available. A large file can be a
necessary paired test array or exact-resume checkpoint; do not classify by size alone.

For disposable inference caches:

```bash
python scripts/experiment_registry.py storage
python scripts/experiment_registry.py clean --root output/QUESTION/RUN
python scripts/experiment_registry.py clean --root output/QUESTION/RUN --apply --inactive
```

Review the preview and establish that readers/writers have stopped before applying.
Keep reconstruction receipts. [The result layout](research_layout.md) separates
plots/tables from technical artifacts and documents metric exports.

Operational history: [initial migration](storage/migration/README.md),
[portable environment and storage validation](storage/portability_validation/README.md),
and [dataset inventory method](storage/dataset_inventory/README.md).
The initial migration briefly placed caches on WORK; the later validation record
moved them back to IDS. The table above is the current placement.
