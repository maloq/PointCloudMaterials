# Six closest MD frames for crystal distance: evaluation

[All datasets](../README.md) · [Browsable card](distance-encoder-md-dense6-075ps-evaluation-20260926-64e0b867.html) · [Full metadata](../records/distance-encoder-md-dense6-075ps-evaluation-20260926-64e0b867.json)

Six preceding/current local MD observations for unchanged fixed Al64 held-out rows and spatial scan-position atoms. Float32 relative coordinates, reused from existing trajectories without additional quantization. Includes verified geometry reused from the stopped D6 preparation; run receipts bind old/new producer identities.

- ID: `distance-encoder-md-dense6-075ps-evaluation-20260926`
- Materials: Al
- Classification: **derived**; role: evaluation_geometry
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/md-dense6-075ps-evaluation-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.411 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Six closest MD frames for crystal distance: evaluation",
  "materials": [
    "Al"
  ],
  "role": "evaluation_geometry",
  "classification": "derived",
  "description": "Six preceding/current local MD observations for unchanged fixed Al64 held-out rows and spatial scan-position atoms. Float32 relative coordinates, reused from existing trajectories without additional quantization. Includes verified geometry reused from the stopped D6 preparation; run receipts bind old/new producer identities.",
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
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 61 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-md-dense6-075ps-evaluation-20260926-64e0b867.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-28T23:47:41.345301+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
