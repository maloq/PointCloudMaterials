# Uniform atom centers for continuous crystal-distance readouts

[All datasets](../README.md) · [Browsable card](spatial-distance-uniform-al64-20260926-8f0d1e09.html) · [Full metadata](../records/spatial-distance-uniform-al64-20260926-8f0d1e09.json)

16 uniformly sampled atom centers per frozen observation frame in 90 train and 15 selection sources; 25 observed geometry patches per center. Current confirmed-crystal distance/visibility labels; no new simulation.

- ID: `spatial-distance-uniform-al64-20260926`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/spatial-distance-geometry`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.321 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Uniform atom centers for continuous crystal-distance readouts",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "16 uniformly sampled atom centers per frozen observation frame in 90 train and 15 selection sources; 25 observed geometry patches per center. Current confirmed-crystal distance/visibility labels; no new simulation.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Exact fixed Al64 source roles and raw frame identities; seed 20260926; calibration/test sources excluded. Version directory is the bound experiment identity.",
  "evidence": [
    "configs/analysis/spatial_distance_20260926.json",
    "docs/spatial_distance.md"
  ],
  "limitations": [
    "Observation frames inherit the historical at-risk sampling; atom sampling is uniform within these frames, not an unbiased simulation-time population.",
    "Nearest80 cropped at 8 Angstrom; spatial support is not a complete radius ball."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 105 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/spatial-distance-uniform-al64-20260926-8f0d1e09.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
