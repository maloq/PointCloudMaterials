# Temporal kinetic coordinates from frozen local-environment embeddings

This experiment asks one deliberately narrow question: does temporal pairing reveal
a small coordinate system in which nearby local atomic environments have similar
futures? It freezes the repository's pretrained point-cloud encoder and fits only a
linear two-sided VAMP model. There is no decoder, nonlinear VAMPnet, or use of state
labels in the fit.

## Repository choices

The reference configuration uses
`VICREG_GEOFRAME_MULTISCALE_8_12_PAPER_l128_N160_M80_GeoFrameTransformer-epoch=95.ckpt`.
This is the `GeoFrameTransformer` checkpoint currently selected by
`configs/analysis/static.yaml` and
`configs/analysis/temporal_crystallization_step187800.yaml`. The experiment consumes
the encoder's 128-dimensional invariant output directly, before VICReg or other SSL
heads. A different `GeoFrameTransformer` checkpoint can be substituted through
`encoder.checkpoint`; its Hydra training config must remain beside it, following the
repository checkpoint contract.

Embedding inference is deterministic for the selected checkpoint:

- the model is put in evaluation mode, which disables all training augmentations;
- eval-time grouping uses FPS, and this checkpoint has `deterministic_fps: true`;
- the checkpoint's deterministic crop from 160 neighborhood points to 80 model
  points is reused;
- `embedding.repeats` can average repeated embeddings for a future checkpoint whose
  eval-time grouping is stochastic. A stochastic checkpoint with `repeats: 1` is
  rejected explicitly.

Temporal neighborhoods are not reimplemented. `TemporalPairDataset` is a two-frame
view over `TemporalLAMMPSDumpDataset`, so it inherits stable atom-ID tracking, wrapped
LAMMPS coordinates, minimum-image periodic displacements, fixed-radius
normalization, and the existing KD-tree/precomputed-neighbor cache.

## Run

The checked-in configuration points to the current crystallization trajectory and
checkpoint:

```bash
conda run -n pointnet python scripts/run_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame.yaml --stage all
```

The expensive encoder pass is cached as memory-mapped `.npy` arrays. Development can
therefore be staged without keeping raw point clouds on the GPU or re-encoding them:

```bash
conda run -n pointnet python scripts/run_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame.yaml --stage extract
conda run -n pointnet python scripts/run_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame.yaml --stage fit
conda run -n pointnet python scripts/evaluate_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame.yaml
```

The cache manifest includes checkpoint and trajectory identity, lag, and resolved
data settings. A stale cache causes an explicit error; set
`cache.force_recompute: true` to intentionally rebuild it.

`lags.frames` accepts one or more frame offsets. Alternatively, set it to `null` and
use `lags.timesteps`; timestep lags must align exactly with a uniformly sampled
LAMMPS trajectory. Each lag is independently fitted and appears in the combined
singular-spectrum plot.

## Fit and saved representation

For centered present/future embeddings, the estimator accumulates `C00`, `C11`, and
`C01` in float64 and in bounded-size batches. It diagonalizes both instantaneous
covariances, removes relative near-null eigenmodes, applies a scale-relative ridge,
whitens both sides, and computes the SVD of the whitened cross-covariance. The saved
present-state map is

```text
xi(z) = ((z - mean0) @ whitening0 @ left_singular_vectors) * singular_values
```

so ordinary Euclidean distance in `xi` is the linear kinetic-map distance. The model
archive also contains the right/future singular functions, spectra, covariance
eigenvalues, retained whitening bases, means, and resolved ridge values.

Each lag directory contains:

- `embeddings/{train,validation}/`: reusable memory-mapped embeddings and atom/time
  metadata;
- `vamp_model.npz` and `pca_model.npz`: reusable linear transforms;
- `coordinates_{train,validation}.npz`: encoder, PCA, kinetic coordinates, future
  embeddings, atom IDs, run indices, frames, timesteps, and positions;
- `metrics.json`: held-out predictive-neighbor, optional future-state probe, split,
  spectrum, determinism, rotation, and regularization diagnostics;
- `plots/`: spectrum, kinetic maps colored by time/run/future state, selected-atom
  paths, and a future-neighbor comparison.

The output root additionally contains `singular_spectra_all_lags.png` and a combined
`run_summary.json`.

## Evaluation semantics

The primary metric finds nearest neighbors using the held-out present state in the
raw encoder, matched-dimensional PCA, or VAMP kinetic space. It then measures mean
Euclidean distance between the corresponding future encoder embeddings and reports
that value relative to random retrieval. Lower is better. By default, all states of
the same tracked atom are excluded from its candidates, preventing adjacent samples
of one atom from making retrieval trivial.

For visualization and the optional logistic probe, future states are K-means labels
fit only on training future embeddings. These labels are evaluation annotations and
are never supplied to VAMP. Time and trajectory/run colors come directly from the
dataset cache. Existing externally computed structural labels are not present in the
current temporal dataset sample contract; `coordinates_*.npz` contains the exact
`(run_index, atom_id, frame)` keys needed to join PTM/CNA or previous cluster labels
without changing the training path.

The default data contains one trajectory, so `split.mode: auto` uses large contiguous
time blocks and drops every pair that would cross their boundary. This prevents the
worst overlap leakage but is not equivalent to validation on independent dynamics.
When multiple independent dumps are available, list them under `data.trajectories`;
`auto` then splits whole runs. This limitation should be stated with any result from
the checked-in single-run configuration.

## Tests

```bash
conda run -n pointnet python -m pytest -q tests/test_temporal_vamp.py
```

The tests cover exact tracked-atom lag construction, leakage-resistant contiguous
splitting, physical-lag alignment, serialization, redundant embedding dimensions,
and recovery of a known slow process embedded in a higher-dimensional noisy linear
representation.
