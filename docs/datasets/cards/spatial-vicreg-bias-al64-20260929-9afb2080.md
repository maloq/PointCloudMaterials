# Matched spatial VICReg Al64 parents and structure assays

[All datasets](../README.md) · [Browsable card](spatial-vicreg-bias-al64-20260929-9afb2080.html) · [Full metadata](../records/spatial-vicreg-bias-al64-20260929-9afb2080.json)

Frozen 128-atom parents at fixed Al64 anchors, eight nearby 80-atom views, complete unfiltered structural training population, uniform plus exact original evaluation rows, independent full-cell PTM visibility and 412 local geometric/bond/CNA/TDA descriptors.

- ID: `spatial-vicreg-bias-al64-20260929`
- Materials: Al
- Classification: **derived**; role: training_and_evaluation_cache
- Location: `/home/ids/vmorozov/training-cache/spatial-vicreg-bias/al64-v1-20260929`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 5.571 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Matched spatial VICReg Al64 parents and structure assays",
  "materials": [
    "Al"
  ],
  "role": "training_and_evaluation_cache",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Frozen 128-atom parents at fixed Al64 anchors, eight nearby 80-atom views, complete unfiltered structural training population, uniform plus exact original evaluation rows, independent full-cell PTM visibility and 412 local geometric/bond/CNA/TDA descriptors.",
  "lineage": "Same 150 independent melts and frozen source roles as fixed Al64 v1; no new MD, descendants or resplit.",
  "evidence": [
    "${dataset:spatial-vicreg-bias-al64-20260929}/manifest.json",
    "configs/spatial_vicreg_bias/al64_20260929.json",
    "docs/spatial_vicreg_bias.md"
  ],
  "limitations": [
    "Observed geometry; not relaxed archived static data.",
    "Evaluation sources historically examined.",
    "All 80 input atoms are retained without the historical radius-8 mask.",
    "Neighbor crop is nearest80 within the central nearest128 pool.",
    "PTM-clear is an evaluation annotation, not an encoder-training filter or proof of absence of order."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["spatial_vicreg_bias_al64_v1"] |
| periodic | [true] |
| seed | [20260929] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 152 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/spatial-vicreg-bias-al64-20260929-9afb2080.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
