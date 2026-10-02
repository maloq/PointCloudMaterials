# al homogeneous unseeded 2nn meam 70304 400-500K 600ps 9independent runs 20260901

[All datasets](../README.md) · [Browsable card](al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901-cedaf030.html) · [Full metadata](../records/al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901-cedaf030.json)

MD payloads retired at user request on 2026-09-26. Historical plots, metrics, logs and provenance remain; no usable trajectory is retained here. See docs/simulations/cleanup_20260926/README.md.

- ID: `al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901`
- Materials: Al
- Classification: **retired**; role: provenance_only
- Location: `/work/PERSO/vmorozov/simulations/al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.046 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "al homogeneous unseeded 2nn meam 70304 400-500K 600ps 9independent runs 20260901",
  "materials": [
    "Al"
  ],
  "role": "provenance_only",
  "classification": "retired",
  "description": "MD payloads retired at user request on 2026-09-26. Historical plots, metrics, logs and provenance remain; no usable trajectory is retained here. See docs/simulations/cleanup_20260926/README.md.",
  "evidence": [
    "${dataset:al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901}",
    "${dataset:al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901}/retirement.json"
  ],
  "retirement_record": "docs/simulations/cleanup_20260926/receipt.json"
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

## Evidence

All 20 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901-cedaf030.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
