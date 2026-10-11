# Al dissolved-embryo and random-liquid continuation controls

[All datasets](../README.md) · [Browsable card](al-nucleus-controls-20261005-98fa4964.html) · [Full metadata](../records/al-nucleus-controls-20261005-98fa4964.json)

36 complete-cell Q,V initial states in 12 matched establishing/dissolved/random-liquid blocks, 180 twelve-ps continuations; no new independent ancestors. Original cadence and source roles retained.

- ID: `al-nucleus-controls-20261005`
- Materials: Al
- Classification: **research**; role: restart_controls
- Location: `/store/PERSO/vmorozov/simulations/nucleus-controls-20261005`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 180; these are not independent-source counts.
- Stored frames: 3060; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 7.560 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Al dissolved-embryo and random-liquid continuation controls",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "restart_controls",
  "classification": "research",
  "description": "36 complete-cell Q,V initial states in 12 matched establishing/dissolved/random-liquid blocks, 180 twelve-ps continuations; no new independent ancestors. Original cadence and source roles retained.",
  "provenance": {
    "recipe": "configs/simulation/nucleus_controls_20261005.json",
    "plan": "${dataset:al-nucleus-controls-20261005}/plan.json"
  },
  "limitations": [
    "Dissolved cases are small sub-established embryos, not melted critical nuclei.",
    "Liquid frames are random conditional on present full-cell structure in matched source ancestors.",
    "Sixteen-hour global limit includes queue time; unfinished coverage remains explicit."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["dissolved_embryo_random_liquid_continuation_v1"] |
| duration_ps | [12.0] |
| seed | [1017030962, 102118555, 103807238, 116394377, 128724691, 140804758, 148748555, 149915680, 151262062, 158256515, 163733626, 184807887] … (146 values; see JSON) |
| timestep_fs | [1.0, 3.0] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| lineage | ["independent_melt_200290231", "independent_melt_228616647", "independent_melt_234029248", "independent_melt_387474214", "independent_melt_468207512", "independent_melt_511618678", "independent_melt_622811055", "independent_melt_63365650", "independent_melt_691988874", "independent_melt_729671835", "independent_melt_745135969", "independent_melt_866573682"] … (13 values; see JSON) |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| cadence_ps | [0.75] |
| atom_count | [70304] |
| material | ["Al"] |
| position_dtype | ["float16"] |
| velocity_dtype | ["float16"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [17] |
| storage_dtype | ["float16", "float32"] |

## Evidence

All 361 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nucleus-controls-20261005-98fa4964.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
