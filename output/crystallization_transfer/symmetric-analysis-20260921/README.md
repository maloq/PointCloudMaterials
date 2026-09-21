# Symmetric-context prediction analysis

All eight completed MACE/GATr predictors; no new training. The event-aligned study follows **287 local events from 27 sources** at every offset, with fixed same-source controls. **Matched AP has 50% case prevalence** and is not comparable numerically with natural-population AP. **Timing uses every event**, retaining survival probability through a 96 ps restricted mean. PNG only; captions below.

[Readable metric table](tables/event_offset_summary.csv) · [Definitions](tables/METRICS.md) · [HTML gallery](index.html)

| Model | Nominal lead (ps) | Matched AP | All-event timing MAE (ps) |
|---|---:|---:|---:|
| mace-direct | 3 | 0.851 | 19.25 |
| mace-direct | 12 | 0.764 | 18.29 |
| mace-direct | 24 | 0.654 | 25.69 |
| mace-direct | 48 | 0.538 | 27.65 |
| mace-ar_mse | 3 | 0.840 | 20.75 |
| mace-ar_mse | 12 | 0.758 | 20.55 |
| mace-ar_mse | 24 | 0.654 | 27.85 |
| mace-ar_mse | 48 | 0.528 | 28.98 |
| gatr-mixture | 3 | 0.848 | 20.09 |
| gatr-mixture | 12 | 0.731 | 19.23 |
| gatr-mixture | 24 | 0.635 | 25.86 |
| gatr-mixture | 48 | 0.516 | 29.90 |

Near-onset discrimination is stronger; around 48 ps, matched AP is close to the 0.5 reference. Timing errors here include every event and 96 ps survival mass, so they are not the earlier detected-only 12 ps timing errors. These are exploratory, one-seed results.

## Completed forecast comparison

![Completed forecast comparison](plots/00_model_comparison.png)

Eight completed one-seed fits, 44,385 at-risk origins from 30 held-out source trajectories. Equal-source integrated Brier spans 128 bins through 96 ps; 12 ps AP uses the natural window population. Predictors start from scratch with frozen encoders, identical 25-query context and observed descriptor auxiliaries. MACE/GATr retain different local supports, so this is not an architecture-only comparison. Lines connect categorical model choices; no interpolation or uncertainty inference is intended.

## Structured context from real atomic clouds

![Structured context from real atomic clouds](plots/07_symmetric_context.png)

Source 938, tracked atom 41382, time 222 ps. Panel a uses the original orthographic-cloud style: all 25 nominal queries (open markers), assigned atom centers (filled markers), and short assignment offsets. Dashed/dotted circles mark the 20/10 Å spherical query radii in projection; maximum assignment offset here is 1.72 Å. All 25 MACE neighborhoods are highlighted, each with its full local support circle and real atoms: tracked center 0, inner-shell slots 1–12, and outer-shell slots 13–24. Overlapping neighborhoods share atoms. Gray background atoms extend to approximately 32 Å for illustration and do not define the model support. Panel b shows all 25 corresponding real atomic crops in slot order, with one common camera and identical magnification; every crop has a 7.94 Å support radius. Both panels show the entire MACE spatial context at this observed time. Projected overlaps do not mean atoms coincide in 3D. Panel c shows all 25 spatial slots at all four input times. Squares denote spatial attention retaining the slots; the ×2 indicates two alternating spatial/causal temporal blocks, followed by a single pooling stage and forecast. The stencil is box-fixed; queries are symmetric, real atom assignments need not be. Slot identity is a spatial query, not a tracked neighboring atom. No future frame enters prediction.

## Same events, changing forecast lead

![Same events, changing forecast lead](plots/08_event_offset_skill.png)

The identical 287 distinct first local onset events and 287 matched control records from 27 sources are reused at every offset and for every model. A control is another tracked center in the same source, liquid/at risk at each forecast origin, with no first onset by the reference event time; it may crystallize later. One seeded control is chosen using availability/labels only and reused at every offset. Some controls repeat across pairs. Each source has equal total weight, its events share this weight, and cases/controls have equal class weight. Left: pooled weighted AP using predicted probability of onset by the reference event time (forecast horizon equals the actual lead); the balanced prevalence baseline is 0.5. This is a case-control diagnostic, not the previous population AP. Right: absolute error of the 96 ps restricted-mean predicted onset time across ALL events, including missed alarms; survival mass stays at 96 ps and no true event time truncates the timing estimator. Each requested origin rounds down to the existing 3 ps grid, making actual leads nominal to nominal+2.25 ps. Shading: 95% paired source-bootstrap intervals, 1,000 draws; no seed uncertainty. Different offsets change the horizon used for AP, but the cohort and estimator stay fixed. Events without all offsets or a control are excluded and listed in the cohort manifest.

## Probability and conditional timing checks

![Probability and conditional timing checks](plots/09_event_offset_probability.png)

Same fixed event/control cohort and source weighting as Figure 8. Solid lines: average predicted onset probability by the reference event; dashed lines: matched controls. The horizon grows with lead, so a larger cumulative probability alone does not establish earlier warning. Right: predicted mean time conditional on onset somewhere in the full next 96 ps, evaluated on ALL true events regardless of alarm. This conditional error can conceal low event probability; the restricted-mean error in Figure 8 retains survival mass. Neither estimator is conditioned on onset by the known true event time.

## The same event viewed from different origins

![The same event viewed from different origins](plots/10_same_event_forecasts.png)

Three score-independent examples: first choose the median-onset event per source, then the 20/50/80% source representatives ordered by onset time. Left/right show MACE autoregressive and GATr mixture on exactly the same events. Each curve is an archived open-loop CDF launched approximately 3, 12, 24 or 48 ps before onset; its start is marked at probability zero. The x-axis is aligned to actual onset (dashed zero). The future onset is used only to align this retrospective figure; predictors receive only prior observations. These selected predictor examples do not replace the all-eight-model aggregate curves.

## MACE structural forecasts

![MACE structural forecasts](mace/plots/01_structural_trajectories.png)

Identical four source-separated examples for both backbones: early onset, later onset, no onset within 96 ps, and an early event missed by the MACE direct model. Examples are median-error windows within sources, then median across eligible sources, using MACE direct only for illustration. Gray past is observed; zero is forecast origin; dashed line is first onset. Direct/AR predict q6 alongside latent and physical states, not by decoding a latent or reconstructing coordinates. All methods observe −48, −12, −3 and 0 ps and roll out open loop through 96 ps. Dense black past is shown only for orientation.

## MACE onset probabilities

![MACE onset probabilities](mace/plots/02_onset_probabilities.png)

Saved onset CDFs for the same four windows. Onset is the tracked atom first crystalline for three consecutive 0.75 ps frames. The “missed early onset” panel is defined by MACE direct’s calibration-only 5%-FPR threshold and is shared across backbones; it is not necessarily missed by every model. No onset means none within 96 ps, not permanent survival.

## MACE probabilistic trajectories

![MACE probabilistic trajectories](mace/plots/03_predictive_spread.png)

Mixture/diffusion mean q6 and pointwise 5–95% sample interval from 64 fresh fixed-seed inference draws; five predetermined sample paths are faintly shown. Two examples match Figures 1–2. All predictors use the same four observation times. Shading is model spread, not a confidence interval or a simultaneous coverage claim; no best-of-samples selection. Aggregates use the original archived evaluation predictions.

## MACE held-out forecast quality

![MACE held-out forecast quality](mace/plots/04_forecast_quality.png)

Shared physical128 standardized mean-path MSE and dense onset Brier by forecast horizon, using original saved predictions on all 44,385 windows. Every source has equal total weight; bands are 95% paired whole-source bootstrap intervals from 1,000 draws. Persistence keeps current physical state constant. Latent error is deliberately excluded from cross-backbone comparisons. Selection uses development sources; the test population has been examined previously.

## MACE embedding space

![MACE embedding space](mace/plots/05_embedding_umap.png)

128D frozen center embeddings. Standardization and UMAP fit only on 5,760 states from 90 training sources; 1,920 states from 30 test sources are transformed afterward. Both maps use exactly the same outcome-independent sampled identities and timeline range 0–594 ps, including post-onset states; color shows PTM status, q6 and temperature. Separate maps are fitted for MACE/GATr, so axes are not aligned and cross-map distances are meaningless. Euclidean UMAP: 30 neighbors, min_dist 0.15, fixed seed. Colors do not enter fitting.

## MACE embedding trajectories

![MACE embedding trajectories](mace/plots/06_forecast_umap_paths.png)

Observed and predicted paths for the same four windows in that backbone’s training-fitted UMAP. White circle: origin; squares: 96 ps endpoints. Gray dotted: observed history; black: observed future; blue/orange: direct/AR predicted future. The 128D predictive mean is transformed before plotting. UMAP can distort or compress off-manifold errors; apparent 2D agreement is not a forecast-quality metric.

## GATR structural forecasts

![GATR structural forecasts](gatr/plots/01_structural_trajectories.png)

Identical four source-separated examples for both backbones: early onset, later onset, no onset within 96 ps, and an early event missed by the MACE direct model. Examples are median-error windows within sources, then median across eligible sources, using MACE direct only for illustration. Gray past is observed; zero is forecast origin; dashed line is first onset. Direct/AR predict q6 alongside latent and physical states, not by decoding a latent or reconstructing coordinates. All methods observe −48, −12, −3 and 0 ps and roll out open loop through 96 ps. Dense black past is shown only for orientation.

## GATR onset probabilities

![GATR onset probabilities](gatr/plots/02_onset_probabilities.png)

Saved onset CDFs for the same four windows. Onset is the tracked atom first crystalline for three consecutive 0.75 ps frames. The “missed early onset” panel is defined by MACE direct’s calibration-only 5%-FPR threshold and is shared across backbones; it is not necessarily missed by every model. No onset means none within 96 ps, not permanent survival.

## GATR probabilistic trajectories

![GATR probabilistic trajectories](gatr/plots/03_predictive_spread.png)

Mixture/diffusion mean q6 and pointwise 5–95% sample interval from 64 fresh fixed-seed inference draws; five predetermined sample paths are faintly shown. Two examples match Figures 1–2. All predictors use the same four observation times. Shading is model spread, not a confidence interval or a simultaneous coverage claim; no best-of-samples selection. Aggregates use the original archived evaluation predictions.

## GATR held-out forecast quality

![GATR held-out forecast quality](gatr/plots/04_forecast_quality.png)

Shared physical128 standardized mean-path MSE and dense onset Brier by forecast horizon, using original saved predictions on all 44,385 windows. Every source has equal total weight; bands are 95% paired whole-source bootstrap intervals from 1,000 draws. Persistence keeps current physical state constant. Latent error is deliberately excluded from cross-backbone comparisons. Selection uses development sources; the test population has been examined previously.

## GATR embedding space

![GATR embedding space](gatr/plots/05_embedding_umap.png)

128D frozen center embeddings. Standardization and UMAP fit only on 5,760 states from 90 training sources; 1,920 states from 30 test sources are transformed afterward. Both maps use exactly the same outcome-independent sampled identities and timeline range 0–594 ps, including post-onset states; color shows PTM status, q6 and temperature. Separate maps are fitted for MACE/GATr, so axes are not aligned and cross-map distances are meaningless. Euclidean UMAP: 30 neighbors, min_dist 0.15, fixed seed. Colors do not enter fitting.

## GATR embedding trajectories

![GATR embedding trajectories](gatr/plots/06_forecast_umap_paths.png)

Observed and predicted paths for the same four windows in that backbone’s training-fitted UMAP. White circle: origin; squares: 96 ps endpoints. Gray dotted: observed history; black: observed future; blue/orange: direct/AR predicted future. The 128D predictive mean is transformed before plotting. UMAP can distort or compress off-manifold errors; apparent 2D agreement is not a forecast-quality metric.

## Example identities

[
  {
    "label": "Early onset",
    "index": 117035,
    "test_row": 17033,
    "source": 938,
    "center": 11,
    "frame": 296,
    "temperature_K": 450.0,
    "onset_ps": 4.5,
    "direct_path_error": 0.6226298213005066,
    "direct_12ps_threshold": 0.2928493916988373
  },
  {
    "label": "Later onset",
    "index": 117661,
    "test_row": 17659,
    "source": 939,
    "center": 8,
    "frame": 204,
    "temperature_K": 450.0,
    "onset_ps": 42.75,
    "direct_path_error": 0.9679901003837585,
    "direct_12ps_threshold": 0.2928493916988373
  },
  {
    "label": "No onset within 96 ps",
    "index": 34171,
    "test_row": 2514,
    "source": 883,
    "center": 7,
    "frame": 64,
    "temperature_K": 520.0,
    "onset_ps": null,
    "direct_path_error": 0.5921463370323181,
    "direct_12ps_threshold": 0.2928493916988373
  },
  {
    "label": "Missed early onset",
    "index": 182478,
    "test_row": 37349,
    "source": 1000,
    "center": 14,
    "frame": 640,
    "temperature_K": 510.0,
    "onset_ps": 5.25,
    "direct_path_error": 0.9685139656066895,
    "direct_12ps_threshold": 0.2928493916988373
  }
]

Event-aligned illustration records:

[
  {
    "source": 943,
    "center": 5,
    "control_center": 3,
    "onset_frame": 248,
    "onset_ps": 186.0,
    "origins_frames": [
      244,
      240,
      236,
      232,
      224,
      216,
      200,
      184
    ],
    "control_onset_frame": 569
  },
  {
    "source": 883,
    "center": 6,
    "control_center": 10,
    "onset_frame": 363,
    "onset_ps": 272.25,
    "origins_frames": [
      356,
      352,
      348,
      344,
      336,
      328,
      312,
      296
    ],
    "control_onset_frame": 475
  },
  {
    "source": 909,
    "center": 11,
    "control_center": 1,
    "onset_frame": 539,
    "onset_ps": 404.25,
    "origins_frames": [
      532,
      528,
      524,
      520,
      512,
      504,
      488,
      472
    ],
    "control_onset_frame": 577
  }
]
