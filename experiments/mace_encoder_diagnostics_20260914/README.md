# Does the forecast MACE encoder encode stable, generalizable local structure?

This study tests the exact single-frame, seed-20260910 MACE encoder used as the
256-dimensional embedding-forecast target. The selected checkpoint has TDA
supervision disabled; trained readouts assess whether its representation contains
TDA information. Observed and FIRE-relaxed TDA remain separate targets.

The [extraction recipe](../../configs/analysis/mace_encoder_diagnostics.json)
selects the checkpoint and measured inputs. The
[readout recipe](../../configs/analysis/mace_encoder_readout.json) identifies the
completed forecast checkpoint supplying its training-only normalization.
Implementation is in
[`src/research/mace_encoder_diagnostics`](../../src/research/mace_encoder_diagnostics/).
Detailed formulas and limitations are in the
[metric contract](../../docs/metrics/mace_encoder_diagnostics.md).

The protocol covers:

1. Repeated inference, batch order/size, independent rotations and point
   permutations, translation, radial FP32 arithmetic and controlled jitter.
2. Atom-ID-matched motion, instantaneous neighbor retention, exact separation
   of motion and reselection terms, and 1/2/4 fixed-frame boundary substitutions.
3. Coordinate and embedding storage round trips, including retained original
   float32 shooting coordinates and independently recomputed TDA/PTM labels.
4. Temporal embedding increments, observed and relaxed TDA readouts, actual
   observed TDA increments, PTM margins, q4/q6 and nearest-shell density.
5. Temporal correlations at 0.75/1.5/3/6/9/12 ps and a retained 16-sibling
   ensemble at 0/0.3/0.6/1.5/3/6/9/12/15 ps.
6. Training/validation/test readout scores, local centered scores, source-level
   bootstrap intervals, geometry and mean controls, shuffled-label controls,
   and smaller readout training samples.

Whole melt lineages are disjoint. Readout fitting uses 18 training sources;
six validation sources select ridge regularization and six test sources assess
generalization. There are 64 sampled centers per source/frame context and three
contexts per source. The temporal assay uses eight independently selected centers
per held-out source. The shooting ensemble supplies one parent lineage only;
branch pairs cannot support population uncertainty estimates.

The test cohort has been used in earlier research. The study is exploratory;
there is no claim of a new blind test, unseen material transfer, phase identification
from TDA alone, or fully predictable microscopic dynamics.

Run from the repository root:

```bash
conda run --no-capture-output -n pointnet python -m src.research.mace_encoder_diagnostics.extract \
  --config configs/analysis/mace_encoder_diagnostics.json --stage all
conda run --no-capture-output -n pointnet python -m src.research.mace_encoder_diagnostics.analyze \
  --config configs/analysis/mace_encoder_readout.json
conda run --no-capture-output -n pointnet python -m src.research.mace_encoder_diagnostics.verify \
  --config configs/analysis/mace_encoder_diagnostics.json
```

Extraction refuses to overwrite completed arrays. A stage may be run separately
with the same recipe; its RNG is independent of which other stages ran in that
process. No encoder weights or simulation inputs are modified.
The verification command recomputes original TDA labels, compares the PTM labels
to the established assay, replays three stored forecast source shards and checks
the sibling momentum/timeline identities. Source files are frozen separately for
each extraction stage so a corrected diagnostic does not relabel earlier arrays.

Results: `${storage:analysis}/mace_encoder_diagnostics/forecast-seed20260910-20260914/`,
with a readable report and plot, compact tables, and exact measured arrays and
provenance under `technical/`. See [completed findings](RESULTS.md).
