# Native Al tracked future-center graphs at 3, 6, 9 ps

[All datasets](../README.md) · [Browsable card](neighborhood-jepa-native-al-horizons369-20260920-a9e70b5d.html) · [Full metadata](../records/neighborhood-jepa-native-al-horizons369-20260920-a9e70b5d.json)

Future local graph teacher targets for joint invariant/equivariant JEPA prediction on the existing 32768/480 anchor split; 97328 valid training and 1440 validation future observations.

- ID: `neighborhood-jepa-native-al-horizons369-20260920`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/neighborhood_jepa/native-al-32768-horizons369-20260920`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.461 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Native Al tracked future-center graphs at 3, 6, 9 ps",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Future local graph teacher targets for joint invariant/equivariant JEPA prediction on the existing 32768/480 anchor split; 97328 valid training and 1440 validation future observations.",
  "provenance": {
    "producer": "src/training_methods/neighborhood_jepa/multihorizon/data.py",
    "parent": "neighborhood-jepa-native-al-32768-v2-20260920",
    "ancestry": "Same 90 training and 15 development native-Al independent roots; no calibration/test roots",
    "raw_precision": "Inherited float16 positions, reconstructed float32 local graphs; no precision recovery",
    "targets": "Physical85 and instantaneous TDA144 use existing physical-coordinate producers; train-only inherited normalizers; future embedding targets computed online by joint encoder"
  },
  "limitations": [
    "Late trajectory origins have no 6/9 ps follow-up and are explicitly masked.",
    "Only the tracked center is added at long horizons; six spatial-neighbor tasks retain their existing current/.75 ps lags.",
    "One seed; no new independent test cohort."
  ],
  "evidence": [
    "${dataset:neighborhood-jepa-native-al-horizons369-20260920}/manifest.json"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| horizons_ps | [[3.0, 6.0, 9.0]] |
| split | ["selection", "train"] |
| materials | [["Al"]] |

## Evidence

All 615 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/neighborhood-jepa-native-al-horizons369-20260920-a9e70b5d.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
