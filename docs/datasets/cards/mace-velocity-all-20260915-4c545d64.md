# mace-velocity-all-20260915

[All datasets](../README.md) · [Browsable card](mace-velocity-all-20260915-4c545d64.html) · [Full metadata](../records/mace-velocity-all-20260915-4c545d64.json)

Previously unregistered derived data. Inspect schemas, source plans and cache protocol before reuse.

- ID: `mace-velocity-all-20260915`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/mace-velocity-all-20260915`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 1; these are not independent-source counts.
- Stored frames: 6; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.867 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "mace-velocity-all-20260915",
  "role": "training_cache",
  "classification": "derived",
  "materials": [
    "Al"
  ],
  "description": "Previously unregistered derived data. Inspect schemas, source plans and cache protocol before reuse.",
  "evidence": [
    "${dataset:mace-velocity-all-20260915}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| split | ["test", "train", "val"] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (151 values; see JSON) |
| timestep_fs | [2.0, 3.0] |
| frame_count | [112, 113, 118, 119, 120, 128, 130, 135, 140, 142, 144, 149] … (75 values; see JSON) |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| atom_count | [70304] |
| seed | [20260915] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| protocol | ["selected_paired_velocity_v1"] |
| storage_dtype | ["float16"] |

## Evidence

All 1127 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/mace-velocity-all-20260915-4c545d64.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-18T18:40:33.535128+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
