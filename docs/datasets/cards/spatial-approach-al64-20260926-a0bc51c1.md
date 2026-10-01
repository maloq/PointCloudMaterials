# Fixed-snapshot Al64 crystal approach labels and probe paths

[All datasets](../README.md) · [Browsable card](spatial-approach-al64-20260926-a0bc51c1.html) · [Full metadata](../records/spatial-approach-al64-20260926-a0bc51c1.json)

Same 126545 fixed sample IDs/source roles with present crystal-distance labels; additional calibration/test scan routes. Preparation pending until technical/prepared.json is complete. No new simulation.

- ID: `spatial-approach-al64-20260926`
- Materials: Al
- Classification: **derived**; role: spatial_distance_labels
- Location: `/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.327 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Fixed-snapshot Al64 crystal approach labels and probe paths",
  "materials": [
    "Al"
  ],
  "role": "spatial_distance_labels",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Same 126545 fixed sample IDs/source roles with present crystal-distance labels; additional calibration/test scan routes. Preparation pending until technical/prepared.json is complete. No new simulation.",
  "lineage": "Inherits 150 independent-melt source roles from fixed Al64. Path positions within a source are correlated.",
  "evidence": [
    "${dataset:spatial-approach-al64-20260926}/technical/prepared.json",
    "configs/analysis/spatial_approach_20260926.json",
    "docs/metrics/spatial_approach.md"
  ],
  "limitations": [
    "Historical at-risk population, not uniform liquid volume.",
    "Known-target scan routes are conditional diagnostics.",
    "Previously inspected test sources, not a fresh test."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["fixed_snapshot_spatial_approach_v1"] |
| seed | [20260926] |

## Evidence

All 156 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/spatial-approach-al64-20260926-a0bc51c1.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
