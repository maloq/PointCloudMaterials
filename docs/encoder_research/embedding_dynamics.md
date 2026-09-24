# Embedding stability and dimension

The [input-noise supplement](input_noise.md) adds controlled coordinate-noise
response and its ratio to natural 0.75 ps movement to the matched table.

[Handbook](README.md) · [Definitions](../metrics/embedding_dynamics.md) ·
[AP experiment proposal](ap_experiments.md)

The **current table uses only 0.75 ps**, including freshly exported Geoformer
VICReg/VISReg and current MACE on the same observed trajectories as the older
baselines. [Matched results](../../output/encoder_research/dynamics-lag075-20260924/RESULTS.md),
[recipe](../../configs/analysis/encoder_dynamics_lag075_20260924.json).
Use this table for the requested short-lag comparison. The earlier coarse-time
table remains available as a historical result.

```bash
conda activate pointnet-torch214
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m src.research.trajectory_stability.native_dense \
  --config configs/analysis/encoder_dynamics_lag075_20260924.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m src.research.trajectory_stability.audit \
  --config configs/analysis/encoder_dynamics_lag075_20260924.json
```

The native export is resumable with checked inputs/checkpoints; the report
requires a fresh output. Native checkpoint loading uses its pinned producer
checkout. Both encoder/projector exports are retained for Geoformer. Recent
MACE checkpoints are evaluated on observed coordinates here, although their
training used relaxed inputs. The report labels this distinction.

Movement d95 = 5 means that five principal directions retain at least 95% of
the total squared 0.75 ps embedding changes. Those directions can mix all
embedding channels. It differs from participation rank, which is a continuous
measure of how evenly movement energy is distributed across directions.

The earlier analysis-only recipe below supplements completed exports without new training, encoder inference or
simulation. It verifies saved feature checksums and source/atom/time identities.
Existing reports and historical metric definitions are preserved.

See the [completed results](../../output/encoder_research/dynamics-20260924/RESULTS.md)
for the September 24 audit of 19 representations.

```bash
conda activate pointnet-torch214
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.research.trajectory_stability.audit \
  --config configs/analysis/encoder_dynamics_20260924.json
```

The [recipe](../../configs/analysis/encoder_dynamics_20260924.json) selects eight
completed current checkpoints, retaining encoder/projector exports separately,
and seven historical dense-trajectory representations. Choose a fresh output
directory for a rerun. The output refuses to overwrite completed evidence.

Outputs under `output/encoder_research/dynamics-20260924/`:

| File | Use |
| --- | --- |
| `tables/stability.csv` | Physical-lag RMS and tail jumps, movement and fluctuation dimensions |
| `tables/ranks.csv` | Full available dataset, fitting reference, evaluation and within-track state spectra, by population |
| `tables/per-track.csv` | Each source/atom's dimensions, duration, sample ceiling, direction and roughness |
| `tables/eigenvalues.csv` | Complete spectra for scree/cumulative-energy plots |
| `plots/coarse.png`, `plots/dense.png` | Separate current coarse screen and historical dense trajectory comparisons |
| `technical/<model>.json` | Full metrics, input/checkpoint checksums, sampling identity |
| `tables/METRICS.md`, `technical/metric-contract.json` | Frozen definitions and implementation hashes |

New evaluations through `geoframe_evolution.prediction.evaluate` (also used by
the native screen/parameter search) include an `embedding_dynamics` JSON block
automatically. Queues running immutable old code keep their old evaluation;
apply this analysis-only supplement to those completed exports afterward.

The current native dataset has four observations per tracked atom, 108–120 ps
apart. Its individual trajectory state rank cannot exceed three. This is **not**
evidence of a three-dimensional physical manifold. Its aggregate state/movement
spectra remain measurable, but fine-scale jitter and event response need dense
trajectories. The historical dense cohort supplies 0.75 ps measurements for its
own older models; it does not validate the latest encoder's temporal smoothness.

Read `dataset` as the complete available exported assay rows, not all atoms or
every training example. Use the whole-dataset rank together with noncrystalline
and within-temperature ranks: phase/material/source separation can dominate a
pooled spectrum. Effective ranks are linear variance/energy dimensions, not a
nonlinear intrinsic-manifold estimator.
