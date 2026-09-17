# al homogeneous unseeded 2nn meam 70304 400-500K 600ps 9independent runs 20260901

[All datasets](../README.md) · [Browsable card](al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901-cedaf030.html) · [Full metadata](../records/al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901-cedaf030.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901`
- Materials: Al
- Classification: **review_required**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 9; these are not independent-source counts.
- Stored frames: 1809; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 11.927 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "al homogeneous unseeded 2nn meam 70304 400-500K 600ps 9independent runs 20260901",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "review_required",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| ensemble | ["NPT"] |
| measurement_duration_ps | [600.0] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0] |
| velocity_seed | [35911, 35923, 35933, 35951, 35963, 35977, 35993, 36007, 36013] |
| equilibration_duration_ps | [15.0] |
| sample_interval_ps | [3.0] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| frame_count | [201] |
| storage_dtype | ["float32"] |

## Evidence

All 29 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901-cedaf030.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-17T22:51:50.502756+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
