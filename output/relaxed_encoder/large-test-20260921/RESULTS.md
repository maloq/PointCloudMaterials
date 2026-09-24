# Larger matched crystallization test

11256 natural at-risk test windows; 338 distinct local atom onsets within 12 ps, in 27 event-bearing sources out of 30 test sources. Local atom onsets within a trajectory are correlated; they are not independent nucleation events.

Selection, calibration and test origins use a fixed 12 ps cadence. No transition oversampling, test-source reassignment or selection by quench completion time. Original MD labels and source splits are retained. Training readouts retain the 15-origin training cohort. This is a historically used source split, not a newly untouched dataset.

| Encoder / input | Readout | Event NLL | 12 ps AP | AUROC | Timing MAE (ps) | Misses / events |
|---|---|---:|---:|---:|---:|---:|
| cold-control | linear | 0.2226 | 0.1533 | 0.7372 | 2.6977 | 292/338 |
| cold-control | mlp | 0.2040 | 0.2529 | 0.8097 | 2.7846 | 187/338 |
| cold-sig-temp1 | linear | 0.2216 | 0.1547 | 0.7404 | 2.8099 | 290/338 |
| cold-sig-temp1 | mlp | 0.2046 | 0.2522 | 0.8092 | 2.7327 | 184/338 |
| cold-sig-temp3 | linear | 0.2186 | 0.1919 | 0.7424 | 2.8524 | 237/338 |
| cold-sig-temp3 | mlp | 0.2054 | 0.2496 | 0.8066 | 2.7170 | 181/338 |
| conditions | linear | 0.2405 | 0.0593 | 0.6106 | 3.1707 | 249/338 |
| conditions | mlp | 0.2382 | 0.0801 | 0.6414 | 3.1228 | 265/338 |
| earlier-epijepa-mace | linear | 0.2343 | 0.0890 | 0.6736 | 3.1811 | 300/338 |
| earlier-epijepa-mace | mlp | 0.2254 | 0.1534 | 0.7231 | 3.0148 | 234/338 |
| earlier-sigreg-mace | linear | 0.2339 | 0.0886 | 0.6735 | 3.1877 | 296/338 |
| earlier-sigreg-mace | mlp | 0.2244 | 0.1433 | 0.7208 | 3.0377 | 243/338 |
| earlier-vicreg-mace | linear | 0.2301 | 0.1366 | 0.6886 | 3.1433 | 250/338 |
| earlier-vicreg-mace | mlp | 0.2247 | 0.1650 | 0.7211 | 3.1966 | 220/338 |
| geometry_cold | linear | 0.2095 | 0.2472 | 0.8170 | 2.8100 | 178/338 |
| geometry_cold | mlp | 0.1860 | 0.3911 | 0.8603 | 2.7756 | 130/338 |
| geometry_hot | linear | 0.2179 | 0.1834 | 0.7730 | 3.0187 | 212/338 |
| geometry_hot | mlp | 0.2067 | 0.2452 | 0.8035 | 2.9693 | 171/338 |
| hot-control | linear | 0.2328 | 0.0928 | 0.6799 | 3.1631 | 296/338 |
| hot-control | mlp | 0.2259 | 0.1419 | 0.7315 | 2.9275 | 213/338 |
| old-vicreg-gatr | linear | 0.2363 | 0.0721 | 0.6540 | 2.9746 | 302/338 |
| old-vicreg-gatr | mlp | 0.2339 | 0.0822 | 0.6799 | 3.0400 | 289/338 |
| old-vicreg-mace | linear | 0.2366 | 0.0896 | 0.6972 | 3.3578 | 318/338 |
| old-vicreg-mace | mlp | 0.2212 | 0.1669 | 0.7571 | 3.0989 | 226/338 |
| original_geometry | linear | 0.2264 | 0.1287 | 0.7332 | 3.1458 | 266/338 |
| original_geometry | mlp | 0.2186 | 0.2090 | 0.7769 | 2.8130 | 193/338 |

All encoders are frozen. Neural heads and linear heads are compared separately. Calibration-only thresholds target 5% source-weighted false-positive rate. Timing MAE covers detections only; also report misses and timing-within-3-ps recall. Whole-source bootstrap intervals preserve temperature strata and exclude training-seed uncertainty. Legacy cached encoders use unrelaxed inputs; parent_hot/parent_cold isolate input relaxation.

Completed readouts: 26/44. Bootstrap exported: False. Full horizon metrics: readouts/tables/. Paired intervals: uncertainty/tables/.

Excluded encoders: hot-to-cold-vic-temp01: User stopped remaining encoder training; evaluate completed checkpoints only
