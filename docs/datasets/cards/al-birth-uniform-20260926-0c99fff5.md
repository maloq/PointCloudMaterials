# Al spontaneous births: 22 independent melts

[All datasets](../README.md) · [Browsable card](al-birth-uniform-20260926-0c99fff5.html) · [Full metadata](../records/al-birth-uniform-20260926-0c99fff5.json)

Continuous position/velocity observations from target-temperature initialization; 0.15 ps cadence; early stopping after 10% PTM crystalline plus 12 ps.

- ID: `al-birth-uniform-20260926`
- Materials: Al
- Classification: **building**; role: raw_dynamics
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al-birth-uniform-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.001 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Al spontaneous births: 22 independent melts",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "raw_dynamics",
  "classification": "building",
  "evidence": [
    "${dataset:al-birth-uniform-20260926}/manifest.json"
  ],
  "lineage": "22 fresh independently melted roots; every source is development/train only",
  "description": "Continuous position/velocity observations from target-temperature initialization; 0.15 ps cadence; early stopping after 10% PTM crystalline plus 12 ps.",
  "limitations": [
    "Prepared/running sources are not completed data.",
    "Outcome-dependent training collection; not a representative fixed-duration evaluation cohort.",
    "Temperature and age are audit metadata, never model inputs."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al_spontaneous_birth_sources_v1"] |
| temperatures_K | [[400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]] |
| timestep_ps | [0.003] |
| ptm_rmsd_cutoff | [0.1] |
| temperature_K | [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500] |
| split | ["train"] |
| root_lineage | ["independent_melt_180989530", "independent_melt_186372838", "independent_melt_203886074", "independent_melt_22359867", "independent_melt_275557365", "independent_melt_2877629", "independent_melt_355088647", "independent_melt_416180197", "independent_melt_443247778", "independent_melt_480113979", "independent_melt_49726191", "independent_melt_530991513"] … (22 values; see JSON) |
| parent_trajectory_id | [null] |
| preparation_seed | [180989530, 186372838, 203886074, 22359867, 275557365, 2877629, 355088647, 416180197, 443247778, 480113979, 49726191, 530991513] … (22 values; see JSON) |
| velocity_seed | [150988134, 219271997, 248007662, 248990724, 25203271, 253546319, 286078853, 287063453, 298286875, 326780954, 336433309, 35654120] … (22 values; see JSON) |
| atom_count | [70304] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-birth-uniform-20260926-0c99fff5.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
