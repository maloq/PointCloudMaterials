# embedding-forecast-full-20260911

[All datasets](../README.md) · [Browsable card](embedding-forecast-full-20260911-1bfca47c.html) · [Full metadata](../records/embedding-forecast-full-20260911-1bfca47c.json)

Bulk forecasting cache arrays deleted at user request on 2026-09-27. Metadata, identity/timeline arrays and rebuild provenance remain. This cache must be rebuilt before use; see docs/storage/cache_cleanup_20260927/README.md.

- ID: `embedding-forecast-full-20260911`
- Materials: Unknown
- Classification: **retired**; role: provenance_only
- Location: `/store/PERSO/vmorozov/training-cache-archive-20260926/embedding-forecast-full-20260911`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.021 GiB
- Missing metadata: materials, generating potential identity

## Notes and relationships

```json
{
  "title": "embedding-forecast-full-20260911",
  "materials": [],
  "role": "provenance_only",
  "classification": "retired",
  "description": "Bulk forecasting cache arrays deleted at user request on 2026-09-27. Metadata, identity/timeline arrays and rebuild provenance remain. This cache must be rebuilt before use; see docs/storage/cache_cleanup_20260927/README.md.",
  "evidence": [
    "${dataset:embedding-forecast-full-20260911}",
    "${dataset:embedding-forecast-full-20260911}/retirement.json"
  ],
  "retirement_record": "docs/storage/cache_cleanup_20260927/receipt.json"
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

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
