# Present-selected Al embryo birth-rate calibration

[All datasets](../README.md) · [Browsable card](al-nucleus-rate-calibration-20261005-49c5cd96.html) · [Full metadata](../records/al-nucleus-rate-calibration-20261005-49c5cd96.json)

64 original ancestors, one present 8–63 atom embryo state each, selected without future outcomes. Source-temperature NPT plus two source-temperature BAOAB continuations, 192 twelve-ps runs; common float16 future classification. No new independent MD preparation ancestry.

- ID: `al-nucleus-rate-calibration-20261005`
- Materials: Al
- Classification: **research**; role: restart_calibration
- Location: `/store/PERSO/vmorozov/simulations/nucleus-rate-calibration-20261005`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 192; these are not independent-source counts.
- Stored frames: 3264; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 11.561 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Present-selected Al embryo birth-rate calibration",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "restart_calibration",
  "classification": "research",
  "description": "64 original ancestors, one present 8\u201363 atom embryo state each, selected without future outcomes. Source-temperature NPT plus two source-temperature BAOAB continuations, 192 twelve-ps runs; common float16 future classification. No new independent MD preparation ancestry.",
  "provenance": {
    "recipe": "configs/simulation/nucleus_rate_calibration_20261005.json",
    "plan": "${dataset:al-nucleus-rate-calibration-20261005}/plan.json"
  },
  "limitations": [
    "Source ancestry overlaps historical campaigns and must not be pooled as independent additional evidence.",
    "NPT resets missing bath state; BAOAB changes thermostat and box dynamics.",
    "Separate three-hour shared deadline; incomplete observations remain explicit."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["present_embryo_birth_rate_calibration_v1"] |
| duration_ps | [12.0] |
| seed | [100253934300161245, 1005420913704276683, 1010217263356106436, 1025361794517692396, 1074573463752529222, 1081501157887260372, 1110786878039747462, 1112013273903166874, 1113769862538511217, 1122768050713089868, 1135486743071251588, 1144418595781872104] … (130 values; see JSON) |
| timestep_fs | [1.0, 3.0] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| lineage | [{"interface_distance_A": 8.0, "local_birth_radius_A": 8.0, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.6}] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| cadence_ps | [0.75] |
| atom_count | [70304] |
| position_dtype | ["float16"] |
| velocity_dtype | ["float16"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [17] |
| storage_dtype | ["float16", "float32"] |

## Evidence

All 385 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nucleus-rate-calibration-20261005-49c5cd96.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
