# Five dense Al sources transferred to CUDA MEAM with native isotropic NPT

[All datasets](../README.md) · [Browsable card](al-dense-gpu-20261004-7ce224cb.html) · [Full metadata](../records/al-dense-gpu-20261004-7ce224cb.json)

70304 atoms; original 2 fs integration, exact 0.1 ps positions/velocities, confirmed-halfway stopping and peer caps. Float32 GPU force/virial, float64 native integration.

- ID: `al-dense-gpu-20261004`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al-dense-gpu-20261004`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
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
  "classification": "research",
  "title": "Five dense Al sources transferred to CUDA MEAM with native isotropic NPT",
  "description": "70304 atoms; original 2 fs integration, exact 0.1 ps positions/velocities, confirmed-halfway stopping and peer caps. Float32 GPU force/virial, float64 native integration.",
  "ancestry": "Five retained original train ancestors; no additional independent melts.",
  "execution_backend": "torch_meam_lammps_iso_npt_v1",
  "evidence": [
    "/store/PERSO/vmorozov/simulation-launches/al-dense-gpu-20261004/manifest.json",
    "/store/PERSO/vmorozov/simulation-launches/al-dense-gpu-20261004/submission.json",
    "/store/PERSO/vmorozov/simulation-launches/al-dense-gpu-20261004/hardware-transfer-20261004/attempt-v2/submission.json"
  ],
  "hardware_transfer": {
    "source_ids": [
      914,
      974
    ],
    "partitions": [
      "A100",
      "L40S",
      "A40"
    ],
    "submission": "/store/PERSO/vmorozov/simulation-launches/al-dense-gpu-20261004/hardware-transfer-20261004/attempt-v2/submission.json",
    "original_active_sources_preserved": [
      894,
      944
    ],
    "updated_at": "2026-10-04T13:41:22.539133+00:00"
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-dense-gpu-20261004-7ce224cb.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
