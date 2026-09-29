# Tracked-atom MD histories for unchanged crystal-front evaluation

[All datasets](../README.md) · [Browsable card](distance-encoder-md-history6-evaluation-20260926-88f1a19c.html) · [Full metadata](../records/distance-encoder-md-history6-evaluation-20260926-88f1a19c.json)

Geometry at -6/-3/0 ps for every original held-out fixed observation and controlled scan-position atom. Same atom ID followed through MD; no new simulation.

- ID: `distance-encoder-md-history6-evaluation-20260926`
- Materials: Al
- Classification: **derived**; role: evaluation_geometry
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/md-history6-evaluation-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.210 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Tracked-atom MD histories for unchanged crystal-front evaluation",
  "materials": [
    "Al"
  ],
  "role": "evaluation_geometry",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Original fixed Al64 selection/calibration/test source roles; no source resplitting.",
  "description": "Geometry at -6/-3/0 ps for every original held-out fixed observation and controlled scan-position atom. Same atom ID followed through MD; no new simulation.",
  "evidence": [
    "configs/distance_encoder/md_history6_20260926.json",
    "docs/distance_encoder_history.md"
  ],
  "limitations": [
    "Al held-out evaluation only; external material generalization is not evaluated.",
    "History visibility includes all observed MD frames; not interchangeable with snapshot visibility."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 61 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-md-history6-evaluation-20260926-88f1a19c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-28T23:47:41.345301+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
