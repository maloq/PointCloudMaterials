# Neighborhood JEPA v2 native Al, fixed 0.75 ps lag and smooth anchored moments

[All datasets](../README.md) · [Browsable card](neighborhood-jepa-v2-native-al-20260920-7a8cb50b.html) · [Full metadata](../records/neighborhood-jepa-v2-native-al-20260920-7a8cb50b.json)

8,995 training anchors in 36 independent native lineages and 480 selection anchors in 15 lineages. Inherited checked radius8 graphs and physical85/TDA144 labels. New degree1/2/4/6 multiscale C2 moments, training-only per-block scales. Known temperature conditions. No simulations added.

- ID: `neighborhood-jepa-v2-native-al-20260920`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/neighborhood_jepa/v2-native-al-20260920-final`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.089 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Neighborhood JEPA v2 native Al, fixed 0.75 ps lag and smooth anchored moments",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "8,995 training anchors in 36 independent native lineages and 480 selection anchors in 15 lineages. Inherited checked radius8 graphs and physical85/TDA144 labels. New degree1/2/4/6 multiscale C2 moments, training-only per-block scales. Known temperature conditions. No simulations added.",
  "evidence": [
    "${dataset:neighborhood-jepa-v2-native-al-20260920}/manifest.json",
    "configs/neighborhood_jepa/v2_native_al_20260920.json"
  ],
  "limitations": [
    "Selection previously used in encoder development.",
    "Inherited float16 coordinate quantization; float32 graph storage does not recover precision.",
    "TDA nearest80 membership remains a separate nonsmooth target; no relaxed TDA."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| lineage | ["independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_176364202", "independent_melt_184384186", "independent_melt_197585375", "independent_melt_23525568", "independent_melt_235751553", "independent_melt_24993146"] … (51 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al"] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/neighborhood-jepa-v2-native-al-20260920-7a8cb50b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
