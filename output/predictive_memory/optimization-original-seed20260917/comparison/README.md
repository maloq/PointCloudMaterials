# Exploratory predictive-memory pilot

All fits use the same 12,000-update budget and one training seed. Lower physical-path NLL is better.

| Model | Selected update | Test joint NLL | Test future MSE |
|---|---:|---:|---:|
| xv-H0 | 2250 | 0.92531 | 0.93632 |
| xv-H12 | 2250 | 0.92226 | 0.93548 |
| xv-H48 | 2250 | 0.92256 | 0.93494 |
| xv-H48-repeat | 6000 | 0.91780 | 0.93127 |

Paired NLL gains: positive values favor real history. Intervals resample whole sources.

| Comparison | Gain | 95% source interval |
|---|---:|---|
| xv_H12_gain_over_snapshot | 0.00305 | [0.00118, 0.00463] |
| xv_H48_gain_over_snapshot | 0.00275 | [0.00060, 0.00475] |
| xv_H48_gain_over_repeated_anchor | -0.00476 | [-0.01440, 0.00397] |

These are previously examined sources, one center per source and three adjacent anchors. The intervals do not cover training-seed variability. Existing full-box float16 observations cannot distinguish physical memory from quantization-noise averaging. No compression, sufficiency, kinetic-closure or confirmatory crystallization claim follows from these fits.
