# Spatial target preparation: timing and numerical agreement

The shared command dispatcher also exposes `spatial-relaxed` for the separate
paired relaxed dataset workflow. Its population accounting is defined in
`spatial_paired_relaxed.md`; it reuses these descriptor calculations without
changing this family's timing or numerical-agreement definitions.

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

`native_selected` calculates only the training-active spatial columns named in a
checksum-bound selection contract (40 bond-order and 113 TDA for the current
1,056,768-row fit). Reference comparisons select those exact named coordinates,
including their original training scales. The `native_selected` additional-speedup
denominator is `native_batch` on the same sample/CPU. Omitted TDA summaries are
not calculated or stored, but shared alpha complexes/persistence are retained.
`target_storage_reduction_fraction = 1 - selected_columns / reference_columns`
refers only to float32 target arrays, excluding coordinates and metadata.

The CNA-inclusive variant selects 40 bond-order, 45 CNA and 113 TDA coordinates
(198 total). `native_cna_batch` calculates all 343 values of those three families;
this is its equal-family comparison baseline. Its original-reference speedup
still includes removal of the 99 unused geometry values. Selection uses only the
training-transform active mask; all selected CNA values retain the same producer.

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
