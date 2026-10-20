# al-birth-focused-v100-20261007-source0902-shot01

[All datasets](../README.md) · [Browsable card](al-birth-focused-v100-20261007-source0902-shot01-4d9ed85a.html) · [Full metadata](../records/al-birth-focused-v100-20261007-source0902-shot01-4d9ed85a.json)

Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.

- ID: `al-birth-focused-v100-20261007-source0902-shot01`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/simulations/al-birth-focused-v100-20261007-source0902-shot01`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 1; these are not independent-source counts.
- Stored frames: 1051; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.162 GiB
- Missing metadata: None in the core fields

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
  "classification": "research",
  "ancestry": "independent_melt_837421620",
  "split": "selection",
  "source_id": 902,
  "sampling_ps": 0.1,
  "protocol": "al_birth_focused_010ps_v1",
  "description": "Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.",
  "evidence": [
    "/store/PERSO/vmorozov/simulations/al-birth-focused-v100-20261007-source0902-shot01/input_metadata.json"
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
    "initial_state": "Full-precision prepared-liquid coordinates, fresh Maxwell velocities and NPT bath; record before any integration",
    "ancestry": "Two velocity reshoots per retained melt; descendants are not independent ancestors",
    "control_path_audit": "/store/PERSO/vmorozov/simulation-launches/al-birth-focused-v100-20261007/technical"
  },
  "sampling_protocol": "prepared_liquid_velocity_reshoot_v1",
  "source_origin": "shooting",
  "collection_id": "al-birth-focused-v100-20261007",
  "shot_index": 1,
  "velocity_seed": 729214079
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| root_lineage | ["independent_melt_837421620"] |
| split | ["selection"] |
| temperature_K | [400.0] |
| preparation_seed | [837421620] |
| velocity_seed | [729214079] |
| protocol | ["al_birth_focused_010ps_v1"] |
| ptm_rmsd_cutoff | [0.1] |
| material | ["Al"] |
| atom_count | [70304] |
| timestep_ps | [0.002] |
| ensemble | ["NPT"] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| frame_count | [1051] |
| storage_dtype | ["float16"] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| source_run_id | ["al-birth-focused-v100-20261007-source0902-shot01"] |

## Evidence

All 5 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-birth-focused-v100-20261007-source0902-shot01-4d9ed85a.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
