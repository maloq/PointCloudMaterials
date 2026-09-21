# BCR full-radius Al implementation fixture

[All datasets](../README.md) · [Browsable card](bcr-real-overfit-20260921-06f1c708.html) · [Full metadata](../records/bcr-real-overfit-20260921-06f1c708.json)

Eight radius-8 angstrom patches from two full-precision frames of one 400K Al trajectory. Fixed parent ancestry; implementation/overfit only, no held-out scientific claim.

- ID: `bcr-real-overfit-20260921`
- Materials: Al
- Classification: **diagnostic**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/bcr/real-overfit-20260921`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "BCR full-radius Al implementation fixture",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "diagnostic",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Eight radius-8 angstrom patches from two full-precision frames of one 400K Al trajectory. Fixed parent ancestry; implementation/overfit only, no held-out scientific claim.",
  "evidence": [
    "configs/bcr/real_overfit.json"
  ],
  "limitations": [
    "Single shared prepared-liquid root, no independent test split. Positions float32; explicit native-ULP rounding bound and periodic image identities."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [20260921] |
| radius_A | [8.0] |
| split | ["train"] |
| temperature_K | [400] |
| timestep_fs | [3.0] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/bcr-real-overfit-20260921-06f1c708.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
