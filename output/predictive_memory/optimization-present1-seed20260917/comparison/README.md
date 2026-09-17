# Exploratory predictive-memory pilot

All fits use the same 12,000-update budget and one training seed. Lower physical-path NLL is better.

| Model | Selected update | Test joint NLL | Test future MSE |
|---|---:|---:|---:|
| xv-H0 | 6250 | 0.94213 | 0.85148 |
| xv-H12 | 3500 | 0.91071 | 0.87939 |
| xv-H48 | 4000 | 0.90483 | 0.84429 |
| xv-H48-repeat | 2250 | 0.92719 | 0.93349 |

Paired NLL gains: positive values favor real history. Intervals resample whole sources.

| Comparison | Gain | 95% source interval |
|---|---:|---|
| xv_H12_gain_over_snapshot | 0.03142 | [0.01352, 0.05218] |
| xv_H48_gain_over_snapshot | 0.03729 | [0.01930, 0.05762] |
| xv_H48_gain_over_repeated_anchor | 0.02236 | [0.01173, 0.03409] |

These are previously examined sources, one center per source and three adjacent anchors. The intervals do not cover training-seed variability. Existing full-box float16 observations cannot distinguish physical memory from quantization-noise averaging. No compression, sufficiency, kinetic-closure or confirmatory crystallization claim follows from these fits.
