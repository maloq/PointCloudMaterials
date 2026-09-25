# Equivariant spatial-context Al feature cache

[All datasets](../README.md) · [Browsable card](equivariant-context-prepared-20260925-91ef98cd.html) · [Full metadata](../records/equivariant-context-prepared-20260925-91ef98cd.json)

Frozen shared-MACE scalar/tensor feature cache at 25 patches for 31,609 prospective windows across 150 independent Al sources. Observed and same-frame relaxed inputs; extraction follows fresh likelihood-trained width-128 encoders. Node59 batch/microbatch 512 queue authorized on 2026-09-25.

- ID: `equivariant-context-prepared-20260925`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/equivariant-context/node59-b512-v2-20260925`
- Present on this machine: False
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Equivariant spatial-context Al feature cache",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Frozen shared-MACE scalar/tensor feature cache at 25 patches for 31,609 prospective windows across 150 independent Al sources. Observed and same-frame relaxed inputs; extraction follows fresh likelihood-trained width-128 encoders. Node59 batch/microbatch 512 queue authorized on 2026-09-25.",
  "evidence": [
    "configs/equivariant_context/comparison_20260925.json",
    "src/research/equivariant_context/data.py",
    "${storage:analysis}/equivariant_context/node59-b512-v2-20260925/technical/inventory.json"
  ],
  "limitations": [
    "Not generated yet.",
    "Historical test sources reused.",
    "Box-fixed query stencil and hard nearest-80 membership are not globally rotation-invariant under resampling.",
    "Inherited float16 absolute-coordinate quantization; no new simulation or quench."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/equivariant-context-prepared-20260925-91ef98cd.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
