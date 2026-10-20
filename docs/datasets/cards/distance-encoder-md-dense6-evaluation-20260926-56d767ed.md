# Six closest MD frames for crystal distance: evaluation

[All datasets](../README.md) · [Browsable card](distance-encoder-md-dense6-evaluation-20260926-56d767ed.html) · [Full metadata](../records/distance-encoder-md-dense6-evaluation-20260926-56d767ed.json)

Six preceding/current local MD observations for unchanged fixed Al64 held-out rows and spatial scan-position atoms. Float32 relative coordinates, reused from existing trajectories without additional quantization.

- ID: `distance-encoder-md-dense6-evaluation-20260926`
- Materials: Al
- Classification: **retired**; role: provenance_only
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/md-dense6-evaluation-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Six closest MD frames for crystal distance: evaluation",
  "materials": [
    "Al"
  ],
  "role": "provenance_only",
  "classification": "retired",
  "description": "Six preceding/current local MD observations for unchanged fixed Al64 held-out rows and spatial scan-position atoms. Float32 relative coordinates, reused from existing trajectories without additional quantization.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Unchanged fixed Al64 selection/calibration/test source roles and spatial scan-position atoms.",
  "evidence": [
    "configs/distance_encoder/md_dense6_075nominal_20260926.json",
    "docs/distance_encoder_history.md"
  ],
  "limitations": [
    "All observations have uniform 0.75-ps cadence.",
    "Held-out performance is Al only."
  ],
  "retirement_record": "docs/storage/cache_cleanup_20261002/receipt.json",
  "storage_status": "Cache arrays deleted at user request on 2026-10-02. Historical scientific description and provenance retained; rebuild before reuse.",
  "relaxation_source_availability_note": "Some historical relaxed-cell/neighborhood inputs were deleted at user request on 2026-10-03; retained downstream artifacts are unchanged. See docs/storage/relaxation_unverified_removal_20261003/receipt.json."
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 43 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-md-dense6-evaluation-20260926-56d767ed.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
