# Expanded native Al JEPA 32,768 anchors: v2

[All datasets](../README.md) · [Browsable card](neighborhood-jepa-native-al-32768-v2-20260920-229e21af.html) · [Full metadata](../records/neighborhood-jepa-native-al-32768-v2-20260920-229e21af.json)

32,768 source-balanced training anchors from 90 native Al lineages, plus the same 480 selection anchors from 15 held-out lineages. Existing Lee 2003 MEAM dynamics, fixed 0.75 ps lag. Physical85 and instantaneous TDA144 labels; 21 tracked atom-centered views. V2 adds fixed smooth multiscale angular moments. No new simulation.

- ID: `neighborhood-jepa-native-al-32768-v2-20260920`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/neighborhood_jepa/native-al-32768-v2-20260920`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.319 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Expanded native Al JEPA 32,768 anchors: v2",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "32,768 source-balanced training anchors from 90 native Al lineages, plus the same 480 selection anchors from 15 held-out lineages. Existing Lee 2003 MEAM dynamics, fixed 0.75 ps lag. Physical85 and instantaneous TDA144 labels; 21 tracked atom-centered views. V2 adds fixed smooth multiscale angular moments. No new simulation.",
  "evidence": [
    "${dataset:neighborhood-jepa-native-al-32768-v2-20260920}/manifest.json",
    "configs/neighborhood_jepa/large_20260920/data.json"
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
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297", "independent_melt_160380922", "independent_melt_176364202"] … (105 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al"] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/neighborhood-jepa-native-al-32768-v2-20260920-229e21af.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T18:31:23.169342+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
