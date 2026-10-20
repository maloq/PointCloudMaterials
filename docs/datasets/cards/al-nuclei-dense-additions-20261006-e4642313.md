# Verified native MEAM production additions to Al birth search

[All datasets](../README.md) · [Browsable card](al-nuclei-dense-additions-20261006-e4642313.html) · [Full metadata](../records/al-nuclei-dense-additions-20261006-e4642313.json)

New completed native LAMMPS Lee2003 Al MEAM production verified from executed pair style, potential-file hashes, prepared-liquid provenance and conversion receipts. Same four structure detectors and v3 lineage search at exact 0.10/0.30 ps. Recovery/control/diagnostic purposes excluded.

- ID: `al-nuclei-dense-additions-20261006`
- Materials: Al
- Classification: **derived**; role: label_audit
- Location: `/store/PERSO/vmorozov/experiments/nuclei_search/al-dense-additions-20261006-v3`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 2.606 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Verified native MEAM production additions to Al birth search",
  "materials": [
    "Al"
  ],
  "role": "label_audit",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "New completed native LAMMPS Lee2003 Al MEAM production verified from executed pair style, potential-file hashes, prepared-liquid provenance and conversion receipts. Same four structure detectors and v3 lineage search at exact 0.10/0.30 ps. Recovery/control/diagnostic purposes excluded.",
  "ancestry": "Original production melt groups and fixed roles retained; cadence views and descendants are not independent samples.",
  "provenance": "src.research.crystallization_origin.production_inventory and lineage_search; source-level evidence and SHA-256 in extraction technical/inventory/production-audit.json",
  "evidence": [
    "configs/analysis/al_nuclei_dense_additions_extract_20261006.json",
    "configs/analysis/al_nuclei_dense_additions_20261006.json",
    "docs/al_nuclei_lineage.md",
    "${storage:training_storage}/nuclei_search/al-dense-additions-20261006-extraction/technical/inventory/production-audit.json",
    "${dataset:al-nuclei-dense-additions-20261006}/technical/launch.json"
  ],
  "limitations": [
    "Operational event/fate definitions, not independently validated critical nuclei.",
    "Newly completed sources after the snapshot require a separate release; this is not all-Al completion.",
    "No relaxation, predictive fitting or resplitting; held-out production source labels remain intact.",
    "2026-10-07 user-authorized cleanup: redundant full events.csv export removed; authoritative event shards, consolidated al-nuclei-birth inputs, summaries and provenance remain."
  ],
  "source_records": 29,
  "frame_count": 77399,
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

All 236 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nuclei-dense-additions-20261006-e4642313.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
