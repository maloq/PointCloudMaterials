# H100 causal MACE runtime measurements

All 18 cases completed on the current H100 NVL (95,830 MiB exposed). The original pilot shared the GPU. These measurements are provisional contended throughput, not isolated hardware benchmarks.

| Width | Batch | Original examples/s | Packed resident examples/s | Speed ratio | Resident peak allocated / reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 | 2 | 14.3 | 22.0 | 1.53x | 6.0 / 6.6 |
| 16 | 8 | 14.6 | 30.1 | 2.06x | 12.5 / 14.7 |
| 16 | 16 | 16.6 | 34.1 | 2.06x | 21.1 / 25.2 |
| 32 | 2 | 10.4 | 15.3 | 1.47x | 7.6 / 8.7 |
| 32 | 8 | 14.0 | 26.7 | 1.90x | 19.6 / 23.9 |
| 32 | 16 | 19.3 | 31.7 | 1.64x | 35.1 / 43.6 |

Width-16 batch-8 validation inference increased from 24.6 to 101.0 examples/s (4.10x). It includes encoder, heads, transfer and CPU output copies; it excludes physical metric calculation. CPU preflight and one-time input residency are excluded from training timings. Device residency costs about 3.85 GiB of allocated history storage on this cache.

Across packed GPU cases, maximum output absolute difference was 1.19e-06; maximum full parameter-gradient relative L2 difference was 1.06e-06. FP32/TF32-disabled outputs and absent-gradient patterns passed the declared checks. No AMP, label, graph-support or task-loss change was used.

62 tests passed across causal packing/training/resume/probes/comparison/layout, neighboring velocity/data protocols, portability and allocation handling.

The selected new cohort uses batch 8, width 16, 5,000 updates with resident packing. Batch 16 improved the measured width-16 throughput only about 13%, while changing optimization and increasing memory; it is not silently substituted for the matched batch-8 recipe.

![Training throughput](plots/throughput.png)

Definitions and source hashes: [METRICS.md](tables/METRICS.md). Raw repetitions: [results](technical/results.json). GPU process snapshots: [metadata](technical/metadata.json).
