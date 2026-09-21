# embedding-forecast-full-20260911

[All datasets](../README.md) · [Browsable card](embedding-forecast-full-20260911-1bfca47c.html) · [Full metadata](../records/embedding-forecast-full-20260911-1bfca47c.json)

Derived inputs or targets. Source lineages and target protocol remain part of the dataset definition.

- ID: `embedding-forecast-full-20260911`
- Materials: Unknown
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/embedding-forecast-full-20260911`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 48.912 GiB
- Missing metadata: materials, generating potential identity

## Notes and relationships

```json
{
  "title": "embedding-forecast-full-20260911",
  "materials": [],
  "role": "training_cache",
  "classification": "derived",
  "description": "Derived inputs or targets. Source lineages and target protocol remain part of the dataset definition.",
  "evidence": [
    "${dataset:embedding-forecast-full-20260911}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cadence_ps | [0.75] |
| history_ps | [6] |
| seed | [20260911] |
| storage_dtype | ["float16"] |
| preparation_seed | [114745743, 129223029, 134462729, 135035943, 13937749, 139885636, 142641279, 146234076, 147978527, 151871197, 15458297, 176364202] … (125 values; see JSON) |
| split | ["test", "train", "val"] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 127 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/embedding-forecast-full-20260911-1bfca47c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T09:05:46.021728+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
