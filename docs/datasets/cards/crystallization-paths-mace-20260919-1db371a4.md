# Future-center MACE trajectories for local crystallization path forecasts

[All datasets](../README.md) · [Browsable card](crystallization-paths-mace-20260919-1db371a4.html) · [Full metadata](../records/crystallization-paths-mace-20260919-1db371a4.json)

Same 150 independent Al trajectories and fixed 90/15/15/30 source split as the transfer assay. Frozen parent MACE center embeddings extend to 594 ps on the existing 3 ps grid. Existing full-timeline physical packets, bond order and PTM labels remain in the parent cache. No new simulations.

- ID: `crystallization-paths-mace-20260919`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystallization-paths-mace-20260919`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.229 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Future-center MACE trajectories for local crystallization path forecasts",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Same 150 independent Al trajectories and fixed 90/15/15/30 source split as the transfer assay. Frozen parent MACE center embeddings extend to 594 ps on the existing 3 ps grid. Existing full-timeline physical packets, bond order and PTM labels remain in the parent cache. No new simulations.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "configs/crystallization_transfer/mace_paths_20260919.json",
    "output/crystallization_transfer/mace-paths-20260919/technical/plan.json"
  ],
  "limitations": [
    "Derived predictions target local structural states, not atomic coordinates.",
    "Window overlap does not create independent sources.",
    "Future targets are separated from causal observations; extraction completion is recorded per source."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 150 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystallization-paths-mace-20260919-1db371a4.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
