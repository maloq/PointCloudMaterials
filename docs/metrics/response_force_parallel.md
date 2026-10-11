# Parallel force-only response latency

Three warm repeats of the same first training parent,24Å environment,
two pre-existing common-noise seeds and epsilon0.01Å. The five paths
(central and two signed pairs) are dispatched to a persistent pool of the
declared number of single-thread CPU engines. `seconds` includes dispatch,
all five trajectories, feature calculation, result transfer and signed finite
differences. Process startup, potential initialization and the first warmup
bundle are excluded and recorded separately. This benchmark creates no W&B run.

`serial_seconds` is the recorded single-thread complete bundle time from the
separate serial campaign on exactly the same source/seed/radius/epsilon.
`serial_to_parallel_ratio` divides that by the measured parallel seconds.
`mace_gpu_seconds` is the saved two-shot MACE AD gate divided by two.
`mace_gpu_to_parallel_cpu_ratio` divides that by parallel seconds. These are
wall-latency comparisons with different CPU counts/GPU resources, not equal
hardware efficiency or MACE-physics equivalence. Repeats are hardware repeats,
not additional independent scientific samples.

`value_rms_to_serial` is RMS over256 features;
`response_relative_to_serial` is the Frobenius norm of the signed256×2
response difference divided by the serial response norm (floor1e-15).
Every query requires value RMS≤1e-10 and response relative error≤1e-6.
The serial campaign supplies independent finite-difference and halo gates;
parallel equivalence does not replace them. Potential changes retain their
true labels and no encoder/predictor is trained by this benchmark.
