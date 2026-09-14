# Measured structures and alternative embedding futures

[Open the gallery](../../output/embedding_forecast/structure-embedding-paths-umap-20260913/index.html)
or the [figure report](../../output/embedding_forecast/structure-embedding-paths-umap-20260913/README.md).
The gallery contains seven PNG/PDF figures, a synchronized GIF, and four offline
HTML explorers with rotating 3D point clouds, a shared two-dimensional UMAP, and a time slider.

The question is what local geometry and embedding evolution look like around a
forecast origin, and how the mixture's possible futures differ from the actual
future. This is a visualization of measured data and selected-model predictions,
not a new accuracy evaluation. All 12 fits of the original study are complete;
these examples use seed 20260913 of the 12 ps deterministic, spatial deterministic,
and spatial K=4 checkpoints. The ongoing short-history extension is separate.

## Diagnostic cases

Selection uses the existing spatial-mixture probability readout and its frozen
validation-selected 9 ps / three-frame-persistence threshold, 0.4922295808792114.
Choose the first eligible center/origin for the first remaining diagnostic category
in test-source manifest order, with a different source for each category. Positive
events must start 3–6 ps ahead to leave visible post-onset frames. This rule selects
four 400 K simulations; it does not represent the study's temperature distribution.

| Case | Source | Center atom | Origin, ps | True first sustained onset after origin, ps | Forecast onset after origin, ps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Well timed | 24 | 469 | 403.50 | 6.00 | 6.00 |
| Early | 25 | 10983 | 427.50 | 5.25 | 0.75 |
| Missed | 26 | 27041 | 579.75 | 6.00 | No crossing in 9 ps |
| False alarm within 9 ps | 27 | 1392 | 461.25 | 25.50 | 8.25 |

The observed paths are noisy and the conditional component means separate in
embedding space; a gate-weighted mean can lie between these different paths.
The well-timed case concerns the frozen readout's threshold crossing, not perfect
prediction of every embedding channel. The early example contains transient crystal
frames before its sustained onset; a single crystal snapshot does not define this event.
The false alarm is specific to the 9 ps prediction horizon, although that center
crystallizes later. These chosen examples do not establish aggregate performance.

## Representation and rendering

Track the exact same center atom used by the embedding cache and physical assay.
At each of 29 frames from −12 to +9 ps, reconstruct its instantaneous 80 nearest
periodic atoms, including the center. Show minimum-image coordinates in angstrom,
with fixed orientation and scale within each example. Neighbor identities are
reselected at each time, just as in the encoder producer. Blue neighbors occur at
the origin; amber neighbors do not. The larger center is green for measured PTM
FCC/HCP/BCC; smaller neighbor colors do not indicate their phases. The static
point-cloud figures emphasize the nearest 12 neighbors while retaining all 80 atoms.
These are simulation coordinates; this forecaster has no coordinate decoder.

Use one UMAP fitted only to the retained 4,736 standardized training embeddings:
four centers and sixteen frames per center from each of the 74 training sources.
Use 30 neighbors, minimum distance 0.15, Euclidean distance in the original 256
standardized features, and fit/transform seed 20260913. The installed umap-learn
version is 0.5.12. Transform 4,692 unique observed/predicted query embeddings in one
batch and reuse the exact resulting coordinates for overlapping windows. Fit no
projection on test futures. The saved reducer and sample allow reproduction.
UMAP axes have no physical units or explained-variance percentages. Transform the
256-dimensional mean embedding itself; averaging UMAP coordinates would give a
different result. This follows the [train/transform workflow](https://umap-learn.readthedocs.io/en/latest/transform.html).

The complete-trajectory figure separates three views in each row: occupancy in the
UMAP map, UMAP 1 through time, and UMAP 2 through time. It retains all 801 frames as
points without connecting them in the map. Time panels show faint raw points and
nonoverlapping eight-frame medians, with the final partial bin retained. Physical
PTM state has its own strip. These six-ps summaries are display coordinates, not
new physical trajectories or modified model inputs.

The feature heatmap asks which learned features push the crystal score toward or
away from crystal. Show 20 channels ranked on training data by |classifier weight|
times channel standard deviation, retaining that row order in every example. A
cell is weight × (feature value − mean observed-history feature value). Red means
a positive contribution to score change and blue means a negative one. Positive
and negative contributions can cancel strongly; the bottom score curves use all
256 features. This is a linear-score decomposition, not a claim of physical causes.
The full-channel future figure uses changes from the same history baseline, grouping
features by training Pearson-correlation distance with average-linkage clustering
and optimal leaf ordering. Its color scale is common and unclipped.

### What are crystal margin and readout probability?

**Crystal-classifier score**, previously called crystal margin, is the frozen
class-balanced ridge classifier's crystal score minus its noncrystal score:
`s(z) = w·z + b`. Positive favors crystal, negative favors noncrystal. It is a
learned diagnostic score, not a physical order parameter or a probability. The
classifier was fitted to training embeddings and local PTM labels, with ridge
strength selected on validation data.

**Model P(positive score)**, previously called readout probability, is the probability
that an embedding drawn from the predicted future mixture has `s(z) >= 0`. For
component k it is the Gaussian probability implied by its projected mean and variance;
combine these using the component weights. This is per future frame. It has not
been calibrated as a probability of actual PTM crystallinity or sustained onset.
The displayed examples stay close to 0.5, and warnings depend on crossing the frozen
validation threshold 0.49223. A “well-timed” crossing does not imply high confidence.
Measured PTM state, classifier score and this probability now have separate panels.

Draw 24 future paths per case with the maintained mixture sampler. A component
identity is selected once for the complete 12-frame future; conditional Gaussian
residuals are independent across time/channels. No smoothing, interpolation or
selection for agreement with truth is applied to samples. Colored component curves
are conditional means. Active component weights do not by themselves establish four
distinct physical pathways. Frame readout probability is not a calibrated physical
sustained-onset probability.

Source, atom, frame, cache and checkpoint identities are checked. Selected-window
CPU predictions agree with the retained GPU readout scores within 2e-4 absolute
and relative tolerances. Spatial pooling preserves float32 reduction followed by
the assay's cache-dtype storage. Ground-truth future geometry is used for display
only; all forecast inputs stop at the origin.

## Reproduction

Use conda `pointnet` from the repository root:

```bash
python -m src.research.forecast_spatial_mixture.trajectories --config configs/embedding_forecast/trajectory-visuals.json
```

The current recipe reuses the retained raw extraction and preserves the previous PCA
figures and definitions in their original output. Choose a fresh output to repeat
projection, or set `reuse_extraction` to null to repeat raw extraction as well. To refresh only the figures
from retained arrays, without inference or reading trajectories:

```bash
python -m src.research.forecast_spatial_mixture.trajectories --config configs/embedding_forecast/trajectory-visuals.json --stage plots
```

Implementation is in `src/research/forecast_spatial_mixture/{trajectories,trajectory_projection,trajectory_plots}.py`.
The output retains selected cases, raw standardized paths, complete mixture parameters,
sampled paths, exact measured clouds and neighbor identities, training projection,
input hashes and frozen metric definitions. The figure CSV contains observed UMAP
coordinates, relative times and physical labels. Ten focused checks passed for periodic reconstruction, diagnostic event semantics,
exact score decomposition, time-bin boundaries, and metric-document contracts.

![Measured local structure evolution](../../output/embedding_forecast/structure-embedding-paths-umap-20260913/plots/point-cloud-evolution.png)

![Observed and alternative future embedding paths](../../output/embedding_forecast/structure-embedding-paths-umap-20260913/plots/embedding-futures.png)

The previous PCA renderer, configuration and scientific notes are retained under
`output/embedding_forecast/structure-embedding-paths-20260913/technical/renderer-source/`.
