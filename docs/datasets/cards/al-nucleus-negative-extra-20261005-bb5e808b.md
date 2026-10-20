# Additional Al random-liquid and failed-embryo continuations

[All datasets](../README.md) · [Browsable card](al-nucleus-negative-extra-20261005-bb5e808b.html) · [Full metadata](../records/al-nucleus-negative-extra-20261005-bb5e808b.json)

64 additional original ancestors with one present-liquid state each (NPT plus two BAOAB shots), and four fresh BAOAB shots for each of 12 reused failed embryos: 240 new 12 ps continuations. No new independent MD preparation ancestry; original 0.75 ps cadence and roles retained.

- ID: `al-nucleus-negative-extra-20261005`
- Materials: Al
- Classification: **research**; role: restart_controls
- Location: `/store/PERSO/vmorozov/simulations/nucleus-negative-extra-20261005`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 240; these are not independent-source counts.
- Stored frames: 4080; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 12.299 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Additional Al random-liquid and failed-embryo continuations",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "restart_controls",
  "classification": "research",
  "description": "64 additional original ancestors with one present-liquid state each (NPT plus two BAOAB shots), and four fresh BAOAB shots for each of 12 reused failed embryos: 240 new 12 ps continuations. No new independent MD preparation ancestry; original 0.75 ps cadence and roles retained.",
  "provenance": {
    "recipe": "configs/simulation/nucleus_negative_extra_20261005.json",
    "plan": "${dataset:al-nucleus-negative-extra-20261005}/plan.json"
  },
  "limitations": [
    "Random-liquid eligibility uses present structure, not a negative future label.",
    "Repeated failed embryos are the same starting states as the previous campaign, not additional independent data.",
    "Shared three-hour deadline includes scheduler wait; partial coverage remains explicit."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["additional_negative_continuations_v1"] |
| duration_ps | [12.0] |
| seed | [1005630283309613487, 1009619075213829050, 1009951802886927839, 1023507259967956593, 1024468151146506670, 1048837272644111107, 1049292121082638788, 1051513587128795393, 1052732099560127130, 1053809318975492898, 1061512552590275603, 1069288405158229998] … (178 values; see JSON) |
| timestep_fs | [1.0, 3.0] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| lineage | ["independent_melt_200290231", "independent_melt_228616647", "independent_melt_234029248", "independent_melt_387474214", "independent_melt_468207512", "independent_melt_511618678", "independent_melt_622811055", "independent_melt_63365650", "independent_melt_691988874", "independent_melt_729671835", "independent_melt_745135969", "independent_melt_866573682"] … (13 values; see JSON) |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| cadence_ps | [0.75] |
| atom_count | [70304] |
| position_dtype | ["float16"] |
| velocity_dtype | ["float16"] |
| material | ["Al"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [17] |
| storage_dtype | ["float16", "float32"] |

## Evidence

All 481 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nucleus-negative-extra-20261005-bb5e808b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
