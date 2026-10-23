# PointCloudMaterials

Research tools for structural representations, atomistic dynamics and crystallization prediction from point-cloud observations.

| Start with | Purpose |
| --- | --- |
| [Research guide](docs/encoder_research/README.md) | Evidence, methods, open hypotheses and the next experiments |
| [Documentation](docs/README.md) | Setup, execution, analysis and reference contracts |
| [Development workflow](docs/development.md) | CPU integration checks, complete example and metric provenance |
| [Dataset registry](DATASETS.md) | Available collections, exact protocols, provenance and source roles |
| [Scientific studies](experiments/README.md) | Questions and detailed source protocols, grouped by topic |
| [Results system](docs/research_results_system.md) | Recorded experiments, simulations, analysis artifacts and the generated local dashboard |

The [evidence summary](docs/encoder_research/evidence.md) is the main reading path through past work. The [methods update](docs/encoder_research/methods.md), [experiment plan](docs/encoder_research/experiment_plan.md) and [literature review](docs/encoder_research/literature.md) separate measured findings from proposed remedies. Older reports remain accessible in the [archive/source index](docs/archive/README.md).

Maintained commands are indexed in [scripts/README.md](scripts/README.md). Research recipes live in [experiments](experiments/README.md); simulation campaigns, storage and data conversion have their [operating guides](docs/README.md).

Shared implementation ownership and recent simplifications are recorded in the
[source maintenance notes](docs/src_refactor.md#repository-readability-pass--2026-10-23).

## Installation

Use the maintained conda environment `pointnet-torch214`. Fresh-machine setup,
pinned dependencies and CPU/GPU profiles are documented in
[portable setup](docs/portability.md).

---

## Analysis and results

Use the existing conda environment:

```bash
conda run -n pointnet-torch214 python -m src.analysis.pipeline configs/analysis/static.yaml --checkpoint CHECKPOINT --output-dir output/static-al/review
```

New results have a readable `README.md`/gallery, `plots/`, metric CSVs plus definitions
in `tables/`, and machine artifacts in `technical/`. Existing run paths retain their
resume semantics. See [the folder and retention conventions](docs/research_layout.md)
and [visualization options](docs/analysis_visualization.md).

```bash
conda run -n pointnet-torch214 python scripts/experiment_registry.py build
conda run -n pointnet-torch214 python scripts/experiment_registry.py storage
conda run -n pointnet-torch214 python scripts/experiment_registry.py clean --root output/QUESTION/RUN
```

The cleanup command previews reclaimable inference caches. Applying it requires
`--apply --inactive` after checking that the selected runs are inactive. Dataset and
trajectory conversions still use the maintained conversion command.

Portable machines and storage: see [setup, CPU/GPU environments, dataset bundles and SCRATCH-first simulations](docs/portability.md).
