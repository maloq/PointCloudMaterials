# Force-only local-response teacher comparison

This is a train-only numerical and throughput experiment, with no fitted
encoder or predictive generalization score. Four predetermined training-source
queries, two common-noise seeds, three radii (18/24/36 Å), and two local80
perturbation directions retain the fixed Al64 ancestry and original MACE gate
identities. Noise comes from the exact full-cell CUDA float64 generator used
by the reference; both force-only engines integrate in float64 on CPU.

`engine-comparison.csv` has one row per engine, parent, radius, seed, epsilon,
and feature prefix. The first128 coordinates describe20fs; the second128 are
joint20/100fs Fourier features. Neither prefix is a crystallization label.
Epsilons0.03/0.01/0.003 Å multiply a unit-norm direction distributed over79
noncentral atoms. The central atom stays fixed initially; all environment
atoms subsequently move. Perturbed trajectories use their own initial anchors.

- `value_seconds`: measured complete central100fs trajectory and both feature
  prefixes, including101 force evaluations; persistent engine initialization
  and input/noise loading are outside the clock.
- `bundle_seconds`: central value time plus four perturbed trajectory times at
  the stated epsilon. Three epsilon trials reuse the same central value;
  reported bundle time is the cost of one production two-direction response.
- `reference_mace_ad_seconds_per_shot`: cached original GPU AD gate wall time
  divided by its two shots, including both feature prefixes and directions.
- `reference_gpu_to_cpu_wall_ratio`: previous field / bundle_seconds.
  This compares one CPU thread with the original GPU allocation. It is neither
  a same-hardware speedup nor a claim of equivalent MACE physics. CPU contention
  and reference execution dates are retained in execution receipts. Horizon
  rows repeat the same full100fs timing; do not sum those times.
- `fd_relative_to_smallest_epsilon`: Frobenius norm of the signed256×2
  response difference, divided by the norm at epsilon0.003 (floor1e-15).
  The smallest-epsilon row is trivially zero; this is finite-difference
  convergence, not proof of agreement with an analytic derivative.
- `response_relative_to_mace`: norm of the signed prefix-response difference,
  divided by the original MACE norm (floor1e-15).
- `response_cosine_to_mace`: flattened signed dot product divided by the
  product of norms (denominator floor1e-30). A negative value means opposed
  directions, regardless of their response magnitudes.
- `value_rms_to_mace`: RMS difference of the128 prefix coordinates.

`boundary-convergence.csv` compares each candidate with every larger radius
using the same engine, original
atom identities, seed and epsilon, separately for each feature prefix.
`value_rms` is RMS feature difference; `response_relative` is signed Frobenius
relative difference with36Å as denominator. `passed` requires value RMS≤1e-4
and response relative error≤0.05. This is nested open-environment convergence,
not exact full-periodic equivalence. MEAM/UNEP cannot inherit MACE's halo gate.

Potential files and all calculations are hashed. UNEP-v1 is a different MLIP;
MEAM is the parent generating potential, but this short open-cluster BAOAB
assay differs from the parent full-cell MD. All results stay local; no W&B
runs are created for these engineering checks. No AP objectives or checkpoint
selection occur.
