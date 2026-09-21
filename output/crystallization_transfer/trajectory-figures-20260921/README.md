# Trajectory prediction and atomic context — figure gallery

Seven PNG figures at 300 dpi, with separate captions. No PDF or SVG files are generated. Completed, development-selected context-night forecasts; existing fixed MACE backbone; one training seed. The historical test sources have already been inspected. No predictors were refitted.

Example selection is deliberately transparent: median physical/order/crystallinity error per source, then the median eligible source in each predefined outcome stratum, excluding already chosen sources. The fourth example is explicitly a failure case, not an estimate of its frequency.

| Panel | Source | Center slot (0–15) | Origin (ps) | Onset lead (ps) |
|---|---:|---:|---:|---:|
| Early onset | 911 | 13 | 438 | 10.5 |
| Later onset | 998 | 3 | 309 | 23.25 |
| No onset within 96 ps | 1009 | 7 | 66 | >96 |
| Missed early onset | 970 | 1 | 126 | 10.5 |

## Structural trajectories

![Structural trajectories](plots/01_structural_trajectories.png)

Observed local bond order $q_6$ (black) and open-loop direct (blue) and autoregressive (orange) predictions. Grey shading marks the observed past; zero is the forecast origin. Dashed vertical lines mark the actual first sustained local crystalline onset. Future structural targets are sampled every 3 ps through 96 ps. The physical/order heads predict these quantities alongside the future embedding; these are not reconstructed atomic trajectories or quantities decoded from predicted embeddings. No future values are fed back. Direct/AR models see snapshots at −48, −12, −3 and 0 ps; the dense black past is shown for orientation.

## Crystallization onset probabilities

![Crystallization onset probabilities](plots/02_onset_probabilities.png)

Archived cumulative onset probabilities for the same four windows. Vertical dashed lines denote observed onset, and the light dotted line marks 12 ps. “No onset” means no event observed within 96 ps, not permanent stability. “Missed early onset” is an intentional failure example under the direct predictor’s calibration-only 5% false-positive-rate operating point. Onset uses the tracked center becoming crystalline for three consecutive original 0.75 ps frames. The CDF is a prediction of event timing, distinct from the instantaneous crystallinity state head.

## Probabilistic trajectory spread

![Probabilistic trajectory spread](plots/03_predictive_spread.png)

Mixture and diffusion predictions for the later-onset and surviving examples. Colored lines are the sample means; shading shows pointwise 5–95% intervals from 64 fixed-seed samples. Five predetermined sample paths are shown faintly, without choosing the closest to truth. Black is observed $q_6$. Bands describe model-generated spread, not uncertainty in the mean, not simultaneous coverage, and not a claim of calibration. These two selected models use −12, −3 and 0 ps input frames. Fresh CPU samples are used for illustrations; the aggregate scores retain the original archived samples.

## Held-out forecast quality

![Held-out forecast quality](plots/04_forecast_quality.png)

All 44,385 at-risk test windows across 30 sources. Left: standardized physical-packet MSE of the predictive mean and a constant-current-state persistence baseline. Right: onset Brier score at each lead time. Lower is better. Every source has equal total weight. Shading is a 95% paired source-bootstrap interval (1,000 resamples); it does not represent seed uncertainty. Models are the four family promotions chosen on development data, with a 36-epoch maximum and early stopping. Their observed context differs as documented; this is not a capacity- and input-matched architecture comparison.

## Frozen MACE embedding space

![Frozen MACE embedding space](plots/05_embedding_umap.png)

The same 1,920 held-out atomic states in three colorings: instantaneous PTM crystalline status (FCC/HCP/BCC), local $q_6$, and temperature. The 128-channel encoder is the original fixed trajectory backbone, SHA-256 `ccca9087cd6974023dc0c5c9631337c542efd96fa28245bea883e771530a4722`. Per-channel standardization and UMAP are fitted only on 5,760 states from 90 training sources. Sampling is uniform without replacement over each source’s center/time grid, independently of outcomes, spanning 0–594 ps. UMAP uses 30 neighbors, min_dist 0.15, Euclidean distance and seed 20260921. Held-out points are transformed afterward. Colors never enter the fit. Apparent clusters, overlaps and distances are projection-dependent and do not establish physical phases or predictive sufficiency. These are the frozen MACE features used by the trajectory study; they are not the newly training BCR codes.

## Observed and predicted paths in embedding space

![Observed and predicted paths in embedding space](plots/06_forecast_umap_paths.png)

The same training-fitted UMAP, with held-out states in light grey. Each panel corresponds to the examples in Figures 1–2. Grey dotted lines show past observed embeddings, black shows future observed embeddings, and blue/orange show the direct/AR future predictions. The white circle marks the shared forecast origin; squares mark 96 ps endpoints. Each 128D prediction is transformed into the fixed map. A short or visually close 2D path is not proof of small error: UMAP can compress, distort or fold forecast deviations, including off-manifold predictions. The numeric quality comparison remains in the original standardized target space.

## Spatial context from real atomic neighborhoods

![Spatial context from real atomic neighborhoods](plots/07_spatial_context.png)

Original periodic Al coordinates: source 911, atom 47082, time 438 ps. Panel a is a fixed orthographic projection of a real 3D neighborhood, with minimum-image displacements. Dashed/dotted circle silhouettes mark 25 Å and 12 Å spherical radii; projected overlap does not mean the atoms coincide in 3D. Token 0 is the tracked center. Within each of (0,12] Å and (12,25] Å, the nearest atom seeds a farthest-point selection of three representatives. Each representative is independently encoded from its own local point cloud, shown in panel b with the same camera. The recorded local cutoff is 7.938 Å after converting the normalized cutoff back to physical units; the seven actual atom counts are [122, 119, 128, 120, 117, 122, 127]. The neighboring encoder crops can extend beyond the 25 Å center-selection radius. Panel c shows the direct/AR 48 ps input schedule: seven features at each of −48, −12, −3 and 0 ps. Representatives are selected afresh in each frame, not tracked as fixed neighbor identities. Learned spatial attention uses feature interactions, pairwise geometry and a smooth radial key weight; weighted pooling produces one summary per frame, then causal temporal attention combines those summaries. The current-center residual and known conditions also enter the predictor. Extra observed shell/descriptor features enter a separate auxiliary head (omitted from the diagram). Arrows represent computation, not measured attention or causal influence.

Reproduce: `python -m src.research.crystallization_paths.figures --config configs/analysis/crystallization_figures.json`. CPU replay checks, exact example identities, source/checkpoint hashes, UMAP fit and point-cloud arrays are retained in `technical/`.
