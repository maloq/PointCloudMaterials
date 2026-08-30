# Temporal kinetic coordinates from frozen local-environment embeddings

This experiment asks one deliberately narrow question: does temporal pairing reveal
a small coordinate system in which nearby local atomic environments have similar
futures? It freezes the repository's pretrained point-cloud encoder and fits only a
linear two-sided VAMP model. There is no decoder, nonlinear VAMPnet, or use of state
labels in the fit.

## Repository choices

The 70,304-atom campaign configuration uses the latest corrected epoch-159
`GeoFrameTransformer` checkpoint in
`output/detached/vicreg_geoframe_corrected_h100_20260829_220250`. It consumes the
trained 128-dimensional VICReg projector representation selected by that checkpoint's
current static analysis. `encoder.representation_source: checkpoint` resolves this
choice from the checkpoint Hydra config; `encoder` and `vicreg_projector` are explicit
overrides. A different `GeoFrameTransformer` checkpoint can be substituted through
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

### Optional mesoscopic spatial context

`data.spatial_context.enabled` adds spatial reach without modifying or retraining the
point-cloud network. For every central environment and frame, deterministic
farthest-point selection chooses well-separated satellite atoms from the full
160-neighbor cloud. A normal PBC-aware 160-point environment is built around each
satellite and passed through the same frozen GeoFrameTransformer. The VAMP feature is

```text
[central embedding, mean(satellite embeddings), std(satellite embeddings)]
```

which is permutation invariant in satellite order. With the 128-dimensional encoder
this produces 384 VAMP input features. Satellite centers occupy the outer part of the
central neighborhood and their own environments extend the effective spatial support
to approximately two local-environment radii. The local 128-dimensional embeddings
are retained separately so evaluation compares `encoder`, `context_encoder`, PCA,
and context-VAMP on exactly the same samples. Future-neighbor distances and future
state labels continue to use the central atom's local future embedding, preventing
the larger context vector from redefining the evaluation target.

The first context configuration uses eight satellites, 64 tracked atoms per run,
and 6/12/24 ps lags:

```bash
conda run -n pointnet python scripts/run_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame_70304_meam_transition_context.yaml \
  --stage all
```

## Validated simulation catalog

`configs/temporal_vamp_geo_frame_70304_meam.yaml` discovers 32 compatible unseeded
Al trajectories: the requested 24-replica 400/450/550/600 K campaign, seven 500 K
600 ps replicas, and one 500 K 999 ps replica. Every discovered run is required to
have 70,304 atoms, the same Lee-Shim-Baskes 2NN-MEAM parameter hash, no crystal seed,
and fully periodic boundaries. The other 70k EAM simulations under
`output/synthetic_data` are deliberately excluded because changing the potential or
adding a crystal seed changes the dynamical operator being estimated.

The catalog cross-checks `manifest.json`, `analysis.json`, `in.lammps`,
`crystallization_progress.npz`, and the LAMMPS dump. It preserves temperature,
pressure, physical time, velocity seed, potential hashes, nucleation metadata,
per-frame crystalline fraction, structure fractions, cluster counts, and run
identity. It fails explicitly if the inputs disagree. The saved
`dataset_catalog.json` is the resolved provenance record for a run.

The campaign split holds velocity seeds 35869 and 35879 out at every temperature.
This yields 22 training and 10 validation trajectories and prevents paired replicas
with the same velocity seed from crossing the split.

## Run

The checked-in configuration points to the current crystallization trajectory and
checkpoint:

```bash
conda run -n pointnet python scripts/run_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame.yaml --stage all
```

For the multi-temperature 70,304-atom run:

```bash
conda run -n pointnet python scripts/run_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame_70304_meam.yaml --stage all
```

For a transition-focused comparison, the event-aligned configuration restricts
every run to a physical window around its detected nucleation time:

```bash
conda run -n pointnet python scripts/run_temporal_vamp.py \
  --config configs/temporal_vamp_geo_frame_70304_meam_transition.yaml --stage all
```

`data.event_window` is resolved independently for each catalogued trajectory. Both
members of every temporal pair must lie inside the resolved interval, and the exact
frame bounds and physical times are saved in each lag's split metadata. Event labels
control sampling only; they are not features or targets in the VAMP fit. Results from
an event-aligned ensemble are conditional transition-path results rather than an
unbiased equilibrium estimate. Mixing temperatures also mixes transfer operators,
so pooled fits should be checked against temperature-conditioned fits.

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

`lags.frames` accepts one or more frame offsets. Alternatively, use
`lags.timesteps` or, for catalog-backed data, `lags.picoseconds`; physical lags must
align exactly with each run's recorded cadence. Each lag is independently fitted
and appears in the combined singular-spectrum plot.

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
  embeddings, atom IDs, run indices, frames, timesteps, physical times, temperature,
  velocity seed, global structural progress, and positions;
- `metrics.json`: held-out predictive-neighbor, optional future-state probe, split,
  spectrum, determinism, rotation, and regularization diagnostics;
- `plots/`: spectrum, kinetic maps colored by time, temperature, crystallinity, run,
  or future state, selected-atom paths, and a future-neighbor comparison.

The output root additionally contains `singular_spectra_all_lags.png` and a combined
`run_summary.json`.

## Evaluation semantics

The primary metric finds nearest neighbors using the held-out present state in the
raw encoder, matched-dimensional PCA, or VAMP kinetic space. It then measures mean
Euclidean distance between the corresponding future encoder embeddings and reports
that value relative to random retrieval. Lower is better. By default, all states of
the same tracked atom are excluded from its candidates, preventing adjacent samples
of one atom from making retrieval trivial.

For the multi-run context experiment, `evaluation.future_neighbors.matched_profiles`
adds increasingly strict post-hoc controls. Candidate neighbors can be required to
come from a different MD run, have the same temperature, lie within a configured
time window relative to that run's nucleation event, and have a similar instantaneous
global crystalline fraction. Queries without at least `k` eligible candidates are
excluded and their coverage is reported. The random reference draws `k` states from
the exact same eligible pool for every query, so improvement cannot be attributed to
the metadata restrictions themselves. These metadata are used only to define the
held-out comparison set; they are never inputs to the encoder, PCA, or VAMP fit.

For visualization and the optional logistic probe, future states are K-means labels
fit only on training future embeddings. These labels are evaluation annotations and
are never supplied to VAMP. Time, temperature, run, and global PTM-derived
crystallinity come directly from the dataset cache. `coordinates_*.npz` also contains
the exact `(run_index, atom_id, frame)` keys needed to join future per-atom PTM/CNA or
previous cluster labels without changing the training path.

The legacy `temporal_vamp_geo_frame.yaml` data contains one trajectory, so
`split.mode: auto` uses large contiguous
time blocks and drops every pair that would cross their boundary. This prevents the
worst overlap leakage but is not equivalent to validation on independent dynamics.
When multiple independent dumps are available, list them under `data.trajectories`;
`auto` then splits whole runs. This limitation should be stated with any result from
the checked-in single-run configuration.

## Tests

```bash
PYTHONPATH=. conda run -n pointnet pytest -q \
  tests/test_temporal_vamp.py tests/test_temporal_simulation_catalog.py
```

The tests cover exact tracked-atom lag construction, leakage-resistant contiguous
splitting, physical-lag alignment, serialization, redundant embedding dimensions,
strict multi-file metadata construction, and recovery of a known slow process
embedded in a higher-dimensional noisy linear representation.
