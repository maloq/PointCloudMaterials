# Dense local onset transfer: 150 Al sources, 362400 candidate windows

[All datasets](../README.md) · [Browsable card](crystallization-transfer-mace-20260919-55cdb4ac.html) · [Full metadata](../records/crystallization-transfer-mace-20260919-55cdb4ac.json)

Existing 150 independent 70304-atom Al runs at 400–520 K; 16 outcome-independent tracked centers, 3 ps origins, fixed 90/15/15/30 lineage split. Local MACE graphs and native scalar/tensor features at sparse historical times and seven geometric context centers. Cached verified PTM labels from the local-predictability assay. No simulations generated.

- ID: `crystallization-transfer-mace-20260919`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystallization-transfer-mace-20260919`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 41.186 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Dense local onset transfer: 150 Al sources, 362400 candidate windows",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Existing 150 independent 70304-atom Al runs at 400\u2013520 K; 16 outcome-independent tracked centers, 3 ps origins, fixed 90/15/15/30 lineage split. Local MACE graphs and native scalar/tensor features at sparse historical times and seven geometric context centers. Cached verified PTM labels from the local-predictability assay. No simulations generated.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "output/crystallization_transfer/mace-expanded-20260919/technical/plan.json",
    "configs/crystallization_transfer/mace_20260919.json"
  ],
  "limitations": [
    "Candidate windows are correlated; independent sample unit is source lineage. At-risk filtering reduces the candidate count.",
    "Sparse history snapshots and seven representative context centers, not dense spacetime observations.",
    "Positions enter MACE; the separate physical descriptor baseline additionally uses existing velocities. No relaxed-TDA targets."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 150 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystallization-transfer-mace-20260919-55cdb4ac.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
