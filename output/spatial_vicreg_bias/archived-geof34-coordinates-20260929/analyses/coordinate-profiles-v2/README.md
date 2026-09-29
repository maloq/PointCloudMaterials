# GeoFormer epoch-34 coordinate transitions

Completed descriptive inference on the archived checkpoint. Its saved recipe disables neighbor shifting and includes FactorVAE. This is not the proposed causal training comparison.

Coordinates were selected on the fitting half of 174 ps and evaluated on the opposite spatial half of all three snapshots. These snapshots were in encoder training; this is not independent-source validation. Indices are zero-based.

| Snapshot | Export | Coordinate | Held-region rho | Liquid-only rho | Crystal-clear rho |
| --- | --- | ---: | ---: | ---: | ---: |
| 166ps | encoder | 74 | 0.008 | 0.015 | 0.007 |
| 166ps | encoder | 59 | 0.010 | 0.014 | -0.015 |
| 166ps | encoder | 24 | 0.010 | 0.004 | 0.009 |
| 166ps | projector | 105 | 0.006 | 0.010 | 0.016 |
| 166ps | projector | 3 | -0.008 | -0.017 | -0.012 |
| 166ps | projector | 50 | -0.036 | -0.035 | -0.046 |
| 174ps | encoder | 74 | -0.422 | -0.047 | 0.003 |
| 174ps | encoder | 59 | -0.416 | -0.025 | 0.018 |
| 174ps | encoder | 24 | 0.429 | 0.055 | 0.042 |
| 174ps | projector | 105 | -0.423 | -0.040 | -0.006 |
| 174ps | projector | 3 | 0.417 | 0.039 | 0.005 |
| 174ps | projector | 50 | 0.403 | 0.022 | 0.021 |
| 177ps | encoder | 74 | -0.772 | -0.057 | 0.004 |
| 177ps | encoder | 59 | -0.743 | -0.096 | -0.031 |
| 177ps | encoder | 24 | 0.778 | 0.083 | -0.002 |
| 177ps | projector | 105 | -0.764 | -0.023 | 0.030 |
| 177ps | projector | 3 | 0.700 | 0.039 | -0.013 |
| 177ps | projector | 50 | 0.685 | -0.025 | -0.028 |

Correlations retain the original coordinate sign. Figures orient selected coordinates toward increasing liquid distance. High pooled correlation can arise from phase separation or shared support; inspect individual paths and liquid-only results before calling a transition smooth.

The distance coordinate is half the difference between distance to crystal core and distance to bulk-like liquid, not an exact interface distance. Bands show observation spread. See [frozen metric definitions](tables/METRICS.md).

[frame_00_encoder-coordinate-profiles](plots/frame_00_encoder-coordinate-profiles.png)

![frame_00_encoder-coordinate-profiles](plots/frame_00_encoder-coordinate-profiles.png)

[frame_00_projector-coordinate-profiles](plots/frame_00_projector-coordinate-profiles.png)

![frame_00_projector-coordinate-profiles](plots/frame_00_projector-coordinate-profiles.png)

[frame_01_encoder-coordinate-profiles](plots/frame_01_encoder-coordinate-profiles.png)

![frame_01_encoder-coordinate-profiles](plots/frame_01_encoder-coordinate-profiles.png)

[frame_01_encoder-transects](plots/frame_01_encoder-transects.png)

![frame_01_encoder-transects](plots/frame_01_encoder-transects.png)

[frame_01_projector-coordinate-profiles](plots/frame_01_projector-coordinate-profiles.png)

![frame_01_projector-coordinate-profiles](plots/frame_01_projector-coordinate-profiles.png)

[frame_01_projector-transects](plots/frame_01_projector-transects.png)

![frame_01_projector-transects](plots/frame_01_projector-transects.png)

[frame_02_encoder-coordinate-profiles](plots/frame_02_encoder-coordinate-profiles.png)

![frame_02_encoder-coordinate-profiles](plots/frame_02_encoder-coordinate-profiles.png)

[frame_02_encoder-transects](plots/frame_02_encoder-transects.png)

![frame_02_encoder-transects](plots/frame_02_encoder-transects.png)

[frame_02_projector-coordinate-profiles](plots/frame_02_projector-coordinate-profiles.png)

![frame_02_projector-coordinate-profiles](plots/frame_02_projector-coordinate-profiles.png)

[frame_02_projector-transects](plots/frame_02_projector-transects.png)

![frame_02_projector-transects](plots/frame_02_projector-transects.png)
