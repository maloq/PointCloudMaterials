# New native 0.10 ps Al observations with revised nucleus lineages

[All datasets](../README.md) · [Browsable card](al-nuclei-dense-additions-20261005-a08c71ff.html) · [Full metadata](../records/al-nuclei-dense-additions-20261005-a08c71ff.json)

New completed native 0.10 ps production MD absent from the frozen all-Al search; same four structural detectors and v3 lineage rules at exact 0.10 and 0.30 ps. Recovery, selected-state restarts, seeded controls and engine diagnostics explicitly excluded.

- ID: `al-nuclei-dense-additions-20261005`
- Materials: Al
- Classification: **derived**; role: label_audit
- Location: `/store/PERSO/vmorozov/experiments/nuclei_search/al-dense-additions-20261005-v3`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 3.808 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "New native 0.10 ps Al observations with revised nucleus lineages",
  "materials": [
    "Al"
  ],
  "role": "label_audit",
  "classification": "derived",
  "potential_ids": [],
  "description": "New completed native 0.10 ps production MD absent from the frozen all-Al search; same four structural detectors and v3 lineage rules at exact 0.10 and 0.30 ps. Recovery, selected-state restarts, seeded controls and engine diagnostics explicitly excluded.",
  "provenance": "configs/analysis/al_nuclei_dense_additions_20261005.json; parent extraction plan binds arrays and source producer metadata",
  "ancestry": "Original production melt ancestry retained; related dense descendants and paired cadence views are not independent samples.",
  "evidence": [
    "docs/al_nuclei_lineage.md",
    "${dataset:al-nuclei-dense-additions-20261005}/technical/plan.json",
    "${dataset:al-nuclei-dense-additions-20261005}/technical/launch.json",
    "${storage:training_storage}/nuclei_search/al-dense-additions-20261005-extraction/technical/production-audit.json"
  ],
  "limitations": [
    "Operational appearance/establishment/fate labels, not critical-nucleus estimates.",
    "Static and relaxed data are paired views; live/incomplete trajectories need a later sealed population release.",
    "No prediction-history filter; existing prediction datasets are unchanged.",
    "2026-10-07 user-authorized cleanup: redundant full events.csv export removed; authoritative event shards, consolidated al-nuclei-birth inputs, summaries and provenance remain."
  ],
  "source_records": 28,
  "frame_count": 129028,
  "storage_cleanup": {
    "date": "2026-10-07",
    "payload_files_removed": 1,
    "receipt": "docs/storage/store_cleanup_20261007/payload-cleanup-receipt.json"
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al_nucleus_lineage_v3"] |

## Evidence

All 228 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nuclei-dense-additions-20261005-a08c71ff.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
