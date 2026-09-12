# Pytorch Implementation of PointNet

[Experiment & simulation dashboard](output/registry/index.html) ·
[Ideas backlog](experiments/ideas.json) · [Run organization guide](docs/output_registry.md)

Maintained commands are indexed in [scripts/README.md](scripts/README.md).
Research-specific recipes live in [experiments/](experiments/README.md).
Use the [trajectory conversion tool](docs/trajectory_conversion.md) for format changes.
Test commands and coverage guidelines are in [tests/README.md](tests/README.md).

The [temporal MACE encoder](docs/mace_temporal_encoder.md) combines physical
snapshot histories with learned temporal attention and exports one anchor embedding.
The [encoder, TDA and relaxation review](docs/encoder_tda_relaxation_problems_20260910.md)
collects confirmed problems, implemented corrections, unresolved questions and
the evidence needed for the next experiments.

The [embedding-trajectory forecast experiment](experiments/embedding_forecast_20260911/README.md)
predicts three future time-bin means or every future embedding through 9 ps from
causal histories using direct or autoregressive decoding, with frozen MACE
targets, source-separated evaluation and
optional joint path uncertainty. Methods live in `src/training_methods/embedding_forecast/`.

The frozen-encoder temporal predictive-representation prototype is documented in
[docs/temporal_vamp.md](docs/temporal_vamp.md). Its reference configuration uses the
current pretrained `GeoFrameTransformer` and fits a linear VAMP/kinetic map on
tracked-atom temporal pairs.

## Installation

### Create a new uv environment

```bash
uv pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
```
### Install all other requirements

```bash
pip install -r requirements.txt
```

---

## Analysis and results

Use the existing conda environment:

```bash
conda run -n pointnet python -m src.analysis.pipeline configs/analysis/static.yaml --checkpoint CHECKPOINT --output-dir output/static-al/review
```

New results have a readable `README.md`/gallery, `plots/`, metric CSVs plus definitions
in `tables/`, and machine artifacts in `technical/`. Existing run paths retain their
resume semantics. See [the folder and retention conventions](docs/research_layout.md)
and [visualization options](docs/analysis_visualization.md).

```bash
conda run -n pointnet python scripts/experiment_registry.py build
conda run -n pointnet python scripts/experiment_registry.py storage
conda run -n pointnet python scripts/experiment_registry.py clean --root output/QUESTION/RUN
```

The cleanup command previews reclaimable inference caches. Applying it requires
`--apply --inactive` after checking that the selected runs are inactive. Dataset and
trajectory conversions still use the maintained conversion command.
