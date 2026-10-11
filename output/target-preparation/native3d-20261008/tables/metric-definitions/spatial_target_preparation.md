# Spatial target preparation: timing and numerical agreement

Target definitions remain the 40 bond-order and 258 TDA fields in
[rich liquid descriptors](liquid_descriptors.md), plus the original 99 geometry
and 45 CNA fields when all families are requested. Compiled float64 reductions
preserve neighborhoods, scaling, filtration radii and feature ordering; stored
outputs remain float32. See the [workflow](../target_preparation.md).

Timing seconds is wall time to calculate the entire same sample, excluding
sampling, JIT warmup and publication. Report the median of the declared repeats.
`patches_per_second = rows / median_seconds` and
`speedup = reference_median_seconds / variant_median_seconds`, with identical
worker counts. Multi-process timing includes scheduling/result transfer;
startup is reported separately. These are CPU preparation measurements.
Input loading, shard hashing/writing and native compilation are excluded.
`additional_speedup` compares native variants to the previous optimized safe
producer with the same column set and worker count. `time_reduction_fraction`
is `1 - 1/additional_speedup`. Original-reference speedups for spatial-only
variants include the removal of unused geometry/CNA fields; they are not purely
algorithm improvements at fixed output dimension.

`native_full` and `native_spatial` use native safe alpha persistence with fused
summaries, one scalar call per patch. `native_batch` uses the spatial 298-column
schema and one native call per bounded batch (32 patches per process-pool task
in the October 8 audit). Both complexes, finite H0/H1/H2 intervals and all 258
TDA statistics retain the reference definitions; double filtration is explicit.
The native build receipt pins headers, dependency builds, compiler and CPU flags.

For family F, errors compare the optimized producer with the pre-edit reference
on the same CPU and exact input coordinates. `max_absolute = max(abs(error))`,
`rms_standardized = sqrt(mean((error / frozen_training_SD)^2))`, and
`max_standardized = max(abs(error / frozen_training_SD))`. The SD is the original
fitting transform, including its 1e-4 floor. Unequal values counts nonzero float32
differences. Safe-mode acceptance requires maximum standardized error <=1e-5.
The optional fast-alpha mode is measured, not automatically promoted or asserted
to pass that gate.

Reference-to-sealed-cache drift is recorded separately because nearly-zero tensor
coherence already depends on CPU arithmetic. No historical targets, predictions
or metrics are rewritten. New implementation hashes apply to new computations;
historical reproduction uses its recorded source snapshot.

The sample contains eight rows spread through the first packed shard per training
source. Exact indices, sources, material, ancestry, hashes, reference code hash,
versions and CPU are retained. No event labels or model outputs select the sample.


Table export: 2026-10-08T11:59:50.943500+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/spatial_target_preparation.json` relative to the analysis root.
