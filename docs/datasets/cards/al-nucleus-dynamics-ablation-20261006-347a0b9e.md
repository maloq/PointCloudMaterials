# Al box and thermostat continuation controls

[All datasets](../README.md) · [Browsable card](al-nucleus-dynamics-ablation-20261006-347a0b9e.html) · [Full metadata](../records/al-nucleus-dynamics-ablation-20261006-347a0b9e.json)

384 new 12 ps branches from the same 64 present-selected source states: NVT 3/1 fs and two BAOAB shots at 300/1000 fs damping. Reuses 192 NPT/BAOAB calibration branches. Original source temperatures and common-float16 observations.

- ID: `al-nucleus-dynamics-ablation-20261006`
- Materials: Al
- Classification: **research**; role: restart_calibration
- Location: `/store/PERSO/vmorozov/simulations/nucleus-dynamics-ablation-20261006`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 384; these are not independent-source counts.
- Stored frames: 6528; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 22.767 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Al box and thermostat continuation controls",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "restart_calibration",
  "classification": "research",
  "description": "384 new 12 ps branches from the same 64 present-selected source states: NVT 3/1 fs and two BAOAB shots at 300/1000 fs damping. Reuses 192 NPT/BAOAB calibration branches. Original source temperatures and common-float16 observations.",
  "provenance": {
    "recipe": "configs/simulation/nucleus_dynamics_ablation_20261006.json",
    "plan": "${dataset:al-nucleus-dynamics-ablation-20261006}/plan.json"
  },
  "lineage": "All 64 original ancestors and exact starting states reused from the sealed calibration; no independent ancestry added.",
  "limitations": [
    "Native NVT uses float64 integration; BAOAB response states/forces/noise remain float32.",
    "Damping settings reuse paired noise seeds; repeats are not independent starting states.",
    "Nine-hour shared deadline includes scheduler wait; partial coverage remains explicit.",
    "Original 0.75 ps cohort only; dense/shooting/million-atom/relaxed data are outside this diagnostic."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| duration_ps | [12.0] |
| timestep_fs | [1.0, 3.0] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| lineage | [{"interface_distance_A": 8.0, "local_birth_radius_A": 8.0, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.6}] |
| protocol | ["nucleus_box_thermostat_ablation_v1"] |
| seed | [100253934300161245, 1005420913704276683, 1010217263356106436, 1025361794517692396, 1074573463752529222, 1081501157887260372, 1110786878039747462, 1112013273903166874, 1113769862538511217, 1122768050713089868, 1135486743071251588, 1144418595781872104] … (130 values; see JSON) |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| cadence_ps | [0.75] |
| atom_count | [70304] |
| position_dtype | ["float16"] |
| velocity_dtype | ["float16"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [17] |
| storage_dtype | ["float16", "float32"] |

## Evidence

All 769 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nucleus-dynamics-ablation-20261006-347a0b9e.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
