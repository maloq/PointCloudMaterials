# Known Al nucleus recovery from saved full-cell positions and velocities

[All datasets](../README.md) · [Browsable card](al-nucleus-recovery-20261004-e493b4ed.html) · [Full metadata](../records/al-nucleus-recovery-20261004-e493b4ed.json)

81 initial full-cell snapshots; 671 planned continuations; original test-birth cohort and a separate paired float32 shooting precision panel. Exact ancestry, native cadence, offsets and potential hashes in plan.json.

- ID: `al-nucleus-recovery-20261004`
- Materials: Al
- Classification: **research**; role: restart_recovery
- Location: `/store/PERSO/vmorozov/simulations/nucleus_recovery-saved-state-20261004`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 671; these are not independent-source counts.
- Stored frames: 13026; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 21.831 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Known Al nucleus recovery from saved full-cell positions and velocities",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "restart_recovery",
  "classification": "research",
  "description": "81 initial full-cell snapshots; 671 planned continuations; original test-birth cohort and a separate paired float32 shooting precision panel. Exact ancestry, native cadence, offsets and potential hashes in plan.json.",
  "provenance": {
    "recipe": "configs/simulation/nucleus_recovery_20261004.json",
    "plan": "${dataset:al-nucleus-recovery-20261004}/plan.json"
  },
  "limitations": [
    "Original float16 states have no prebirth native bath restart. NPT resets thermostat/barostat; precision attribution uses only retained float32 shooting states.",
    "Diagnostic subset of shared Al pool; other cohorts remain untested."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["known_nucleus_restart_recovery_v1"] |
| seed | [1000523509, 1006905150, 1008048052, 1009437228, 1009637791, 1015095571, 1017633679, 103457196, 103872093, 105598445, 111391238, 112494390] … (326 values; see JSON) |
| timestep_fs | [1.0, 3.0] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| lineage | [{"interface_distance_A": 8.0, "local_birth_radius_A": 8.0, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.6}] |
| temperature_K | [450.0, 500.0, 510.0, 520.0] |
| cadence_ps | [0.3, 0.75] |
| position_dtype | ["float16", "float32"] |
| velocity_dtype | ["float16", "float32"] |
| atom_count | [70304] |
| duration_ps | [10.499999999999996, 10.5, 10.500000000000004, 16.5, 19.5, 21.75, 24.0, 24.75, 25.5, 27.0, 27.75, 28.5] … (17 values; see JSON) |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [11, 15, 17, 23, 26, 27, 30, 33, 34, 35, 36, 37] … (15 values; see JSON) |
| storage_dtype | ["float16", "float32"] |

## Evidence

All 1343 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nucleus-recovery-20261004-e493b4ed.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
