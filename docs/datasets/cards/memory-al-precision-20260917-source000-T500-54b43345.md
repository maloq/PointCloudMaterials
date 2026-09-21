# memory-al-precision-20260917-source000-T500

[All datasets](../README.md) · [Browsable card](memory-al-precision-20260917-source000-T500-54b43345.html) · [Full metadata](../records/memory-al-precision-20260917-source000-T500-54b43345.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `memory-al-precision-20260917-source000-T500`
- Materials: Al
- Classification: **review_required**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/simulations/memory-al-precision-20260917-source000-T500`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 2; these are not independent-source counts.
- Stored frames: 5122; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 8.144 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "memory-al-precision-20260917-source000-T500",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "review_required",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:memory-al-precision-20260917-source000-T500}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["predictive_memory_precision_sources_v1"] |
| atom_count | [70304] |
| timestep_ps | [0.003] |
| ensemble | ["NPT"] |
| thermostat | ["Nose-Hoover"] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| root_lineage | ["independent_melt_860878647"] |
| parent_trajectory_id | [null] |
| temperature_K | [500.0] |
| split | ["train"] |
| preparation_seed | [860878647] |
| velocity_seed | [112005714] |
| frame_count | [2561] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)", "positions decode to float32 coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| storage_dtype | ["float16", "float32"] |

## Evidence

All 5 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/memory-al-precision-20260917-source000-T500-54b43345.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
