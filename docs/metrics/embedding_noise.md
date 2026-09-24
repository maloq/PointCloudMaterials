# Controlled coordinate-noise response, version 1

This supplements `embedding_dynamics_v1` without changing its 0.75 ps lag or
historical exports. Producers: `trajectory_stability/noise.py`,
`noise_metrics.py`, `noise_worker.py`, with the existing native snapshot driver.

## Observation and perturbation protocol

Use the exact ten evaluation sources / forty tracked atoms of the matched dense
assay. Seed 20260924 selects eight saved origin frames per atom without replacement,
independently of physical outcomes, from frames 0–799. The next frame is always
available at **0.75 ps**. This gives 320 original observations, 32 per source.
Use four independent Gaussian noise realizations per observation, giving 1,280
responses per model and noise level. These are repeated perturbations, not 1,280
independent physical environments or experimental replications.

For each noncentral atom and coordinate draw epsilon ~ N(0,1). At physical noise
level sigma, x_noisy = x_clean + sigma*epsilon; the tracked center stays exactly
at zero. Use sigma = 0.0001, 0.001, **0.01 (primary)** and 0.1 Angstrom per
coordinate. The expected noncentral-atom 3D RMS displacement is sqrt(3)*sigma.
The same epsilon realization is scaled across noise levels and reused across
all models, with the same atom IDs. Add noise to the stored centered float32
coordinates without re-quantizing; this does not recover precision absent in
the original MD coordinate storage.

The finite candidate atom list is the original stored neighborhood (approximately
16.87 Angstrom). Its identities remain fixed. The pipeline recomputes nearest-80
membership (ties resolved by atom ID), each model's native spatial crop, graph
edges and taper weights from the perturbed positions. Thus this includes neighbor
selection and graph changes. It is not a fixed-neighbor differentiability test.
The shared nearest-80 replacement fraction is exported to contextualize changes.
For the older large-support models, atoms outside the stored candidate list
cannot enter; this assay makes no full-cell boundary-crossing claim.

Geoformer exports encoder and projector separately, with its original scaling.
Current geometry MACE retains its native 8 Angstrom support and checkpoint buffers.
Older MACE/GATr are instantiated from hash-matched v6 inference code and the exact
saved checkpoints, using their original BF16/compilation path. The necessary v6
model definitions are recovered from the pinned Git revision, and the historical
observation/batching code is reused from the saved assay. No architecture adapter
or shape-compatible state-dict substitution is used.

Physical descriptors reuse the original producers: nearest-80 persistence images,
single-species Al SOAP (7 Angstrom, n_max8, l_max6, sigma0.3), native radial32 and
angular16 packet components, and six bond-order coordinates. Bond order rebuilds
the twelve nearest bonds of the center and each of its twelve nearest neighbors.
The large candidate chart supplies this local context. Cached clean descriptors
are checked through the replay diagnostic below.

## Calculations

All metric arithmetic uses float64. Sources carry equal total weight; perturbation
rows inside a source have equal weight. Four equal draws per original observation
preserve equal origin weighting. Noncrystalline results condition on the **clean**
origin's PTM label outside {1,2,3}, with the same source weighting. This is not a
verified equilibrium-liquid classification.

Let V be the source-weighted clean reference covariance trace for that model,
read from its existing dense dynamics report (420 observations on five separate
reference sources). No reference moments are refitted using noisy inputs.
Let delta_noise = E(x_noisy)-E(x_clean).

- `response_rms` = sqrt(E_w ||delta_noise||² / (2V)). A value of 0.01 means noise
  moves the state by 1% of that model's reference independent-pair RMS distance.
- `response_rms_raw` = sqrt(E_w ||delta_noise||²), in raw embedding units.
- `response_p95` = weighted empirical 95th percentile of
  ||delta_noise|| / sqrt(2V), using inverse empirical CDF without interpolation.
- `input_rms_displacement_A` = sqrt(E_w mean_noncentral_atoms ||x_noisy-x_clean||²),
  measured on the shared finite candidate chart after float32 rounding. It uses
  the actual displacement, not a nominal sigma or a sum that grows with atom count.
- `sensitivity_per_A` = response_rms / input_rms_displacement_A. This is normalized
  embedding displacement per Angstrom of RMS input noise, not a dimensionless gain.
- `repeat_rms` compares two separately executed clean exports in identical batch
  shape/order, normalized with the same V. `response_to_repeat_ratio` is undefined
  if this measured floor is exactly zero. No subtraction hides the execution floor.
- `replay_rms` compares the new clean export with the corresponding original
  dense cached clean features. It exposes producer, batching or numerical replay
  differences; repeated-input agreement alone would not catch a wrong export.
- `temporal_rms_matched_075` = sqrt(E_w ||z(t+0.75)-z(t)||²/(2V)), using the exact
  sampled origin/atom IDs and cached original dense features. Repeated noise draws
  repeat those temporal pairs without changing their weights. Differences never
  cross atoms or sources.
- `noise_to_temporal_ratio` = response_rms / temporal_rms_matched_075. This is
  undefined for zero natural movement. It compares two magnitudes; it does **not**
  estimate the fraction of actual motion caused by noise.
- `nearest80_replacement_fraction` is the source-weighted mean fraction of the
  clean nearest-80 identities replaced after perturbation. It is an input diagnostic,
  not evidence that every model has a nearest-80 architecture.

The combined table preserves the parent's full-population 0.75 ps RMS, state ranks,
movement rank and d95. Its noise/motion ratio uses the **matched 320-origin**
denominator, which is separately exported, rather than dividing by the full
32,000-pair temporal column. The new primary noise columns always identify sigma.

## Artifacts and interpretation

`tables/noise-response.csv` contains every model, sigma and domain;
`tables/comparison.csv` adds the primary 0.01 Angstrom metrics to the parent table.
Technical artifacts retain the exact selected source/frame/atom/draw IDs, noise
inputs, feature checksums, producer identities, clean-reference identity and
replayed feature values. Original trajectory data and previous reports are unchanged.

No confidence interval is inferred from correlated perturbations. Source-level
uncertainty would require resampling whole sources; training-seed uncertainty is
separate. Lower response means greater robustness to this artificial coordinate
noise. It does not by itself imply better physical sensitivity, preserved present
information, onset AP, or permission to suppress true thermal motion. Current MACE
was trained with relaxed inputs and is measured on observed MD here, as in the
parent dynamics table.
