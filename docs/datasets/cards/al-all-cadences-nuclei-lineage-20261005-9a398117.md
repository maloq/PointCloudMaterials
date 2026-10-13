# Revised Al embryo birth and interaction fate catalogue

[All datasets](../README.md) · [Browsable card](al-all-cadences-nuclei-lineage-20261005-9a398117.html) · [Full metadata](../records/al-all-cadences-nuclei-lineage-20261005-9a398117.json)

Reprocesses all 1042 frozen Al source records and exact 0.75/0.10/0.30 ps views using sealed structural labels; separates emergence, ordinary fragmentation, minor absorption, coalescence and dissolution.

- ID: `al-all-cadences-nuclei-lineage-20261005`
- Materials: Al
- Classification: **derived**; role: label_audit
- Location: `/store/PERSO/vmorozov/experiments/nuclei_search/al-all-20261005-v3`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 26.026 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "Revised Al embryo birth and interaction fate catalogue",
  "materials": [
    "Al"
  ],
  "role": "label_audit",
  "classification": "derived",
  "potential_ids": [],
  "description": "Reprocesses all 1042 frozen Al source records and exact 0.75/0.10/0.30 ps views using sealed structural labels; separates emergence, ordinary fragmentation, minor absorption, coalescence and dissolution.",
  "provenance": "src.research.crystallization_origin.lineage_search; frozen v2 extraction plan and checksums",
  "lineage": "Original melt ancestry, shooting branches, cadence views and historical partial releases retained; no additional independent trajectories.",
  "evidence": [
    "configs/analysis/al_nuclei_lineage_20261005.json",
    "docs/al_nuclei_lineage.md",
    "docs/metrics/al_nuclei_lineage.md",
    "${dataset:al-all-cadences-nuclei-lineage-20261005}/technical/plan.json",
    "${dataset:al-all-cadences-nuclei-lineage-20261005}/technical/launch.json"
  ],
  "limitations": [
    "Operational birth/establishment/fate labels, not critical-nucleus estimates.",
    "Thresholds need physical and cadence sensitivity review; same historical generating potentials recorded per source.",
    "Composite records are not extra births; independent and descendant establishment are separate.",
    "New or ongoing simulations outside the frozen parent release remain unanalysed by this rerun.",
    "Relaxed views are paired observations, not independent temporal outcomes.",
    "2026-10-07 user-authorized cleanup: unindexed 0.75-ps per-source event/interaction/trajectory/size payloads and redundant events.csv exports were deleted. All v3 0.10/0.30-ps shards needed by consolidated al-nuclei-birth reading and refresh remain, along with plans, summaries and provenance."
  ],
  "storage_cleanup": {
    "date": "2026-10-07",
    "payload_files_removed": 2866,
    "receipt": "docs/storage/store_cleanup_20261007/payload-cleanup-receipt.json"
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al_nucleus_lineage_v3"] |

## Evidence

All 4472 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-all-cadences-nuclei-lineage-20261005-9a398117.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
