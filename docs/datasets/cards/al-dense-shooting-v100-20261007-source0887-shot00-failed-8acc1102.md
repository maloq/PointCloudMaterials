# al-dense-shooting-v100-20261007-source0887-shot00-failed

[All datasets](../README.md) · [Browsable card](al-dense-shooting-v100-20261007-source0887-shot00-failed-8acc1102.html) · [Full metadata](../records/al-dense-shooting-v100-20261007-source0887-shot00-failed-8acc1102.json)

Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.

- ID: `al-dense-shooting-v100-20261007-source0887-shot00-failed`
- Materials: Al
- Classification: **incomplete_or_rejected**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/simulations/al-dense-shooting-v100-20261007-source0887-shot00-failed`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.031 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "raw_dynamics",
  "classification": "incomplete_or_rejected",
  "ancestry": "independent_melt_750494487",
  "split": "train",
  "source_id": 887,
  "sampling_ps": 0.1,
  "protocol": "al_dense_half_stop_010ps_v1",
  "description": "Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.",
  "evidence": [
    "/store/PERSO/vmorozov/simulations/al-dense-shooting-v100-20261007-source0887-shot00-failed/input_metadata.json"
  ],
  "execution_backend": "torch_meam_lammps_iso_npt_v1",
  "execution": {
    "engine": "FastAl analytic GPU MEAM with native LAMMPS isotropic NPT",
    "force_dtype": "float32",
    "integration_dtype": "float64",
    "energy_reduction_dtype": "float64",
    "gpu_count": 1,
    "mpi_ranks": 1,
    "runtime_profile": "gpu_meam_v100",
    "validation": "/store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007/technical/audit/validation.json",
    "thermostat_ps": 0.3,
    "barostat_ps": 3.0,
    "pressure_bar": 0.0,
    "stress_scope": "Normal virials; shear not computed",
    "initial_state": "Full-precision prepared-liquid coordinates; new Maxwell velocities; fresh NPT bath",
    "ancestry": "Two velocity reshoots per retained melt; descendants are not independent ancestors"
  },
  "sampling_protocol": "prepared_liquid_velocity_reshoot_v1",
  "source_origin": "shooting",
  "collection_id": "al-dense-shooting-v100-20261007",
  "shot_index": 0,
  "velocity_seed": 751058833
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-dense-shooting-v100-20261007-source0887-shot00-failed-8acc1102.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
