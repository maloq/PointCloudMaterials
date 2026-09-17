# Exploratory predictive-memory pilot

All fits use the same 3,000-update budget and one training seed. Lower physical-path NLL is better.

| Model | Selected update | Test joint NLL | Test future MSE |
|---|---:|---:|---:|
| xv-H0 | 2000 | 0.92405 | 0.93122 |
| xv-H12 | 2250 | 0.92847 | 0.93737 |
| xv-H48 | 2250 | 0.92867 | 0.93797 |
| xv-H48-repeat | 2250 | 0.92363 | 0.93742 |

Paired NLL gains: positive values favor real history. Intervals resample whole sources.

| Comparison | Gain | 95% source interval |
|---|---:|---|
| xv_H12_gain_over_snapshot | -0.00442 | [-0.01856, 0.01224] |
| xv_H48_gain_over_snapshot | -0.00462 | [-0.01930, 0.01223] |
| xv_H48_gain_over_repeated_anchor | -0.00504 | [-0.00954, -0.00117] |

These are previously examined sources, one center per source and three adjacent anchors. The intervals do not cover training-seed variability. Existing full-box float16 observations cannot distinguish physical memory from quantization-noise averaging. No compression, sufficiency, kinetic-closure or confirmatory crystallization claim follows from these fits.
