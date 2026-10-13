# Native0.10ps birth transfer of October6 response encoders

[All datasets](../README.md) · [Browsable card](al-response-dense010-birth-20261006-6eb08b6c.html) · [Full metadata](../records/al-response-dense010-birth-20261006-6eb08b6c.json)

Exact0.1/0.5/1ps pre-embryo snapshots with actual velocities; root embryos later established64 atoms for1.5ps versus matched liquid controls. Three v3 lineage releases; newest labels remain pending until complete.

- ID: `al-response-dense010-birth-20261006`
- Materials: Al
- Classification: **derived**; role: frozen_encoder_transfer_inputs
- Location: `/home/ids/vmorozov/training-cache/response-birth/native010-v3-20261006`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.003 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Native0.10ps birth transfer of October6 response encoders",
  "materials": [
    "Al"
  ],
  "role": "frozen_encoder_transfer_inputs",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Exact0.1/0.5/1ps pre-embryo snapshots with actual velocities; root embryos later established64 atoms for1.5ps versus matched liquid controls. Three v3 lineage releases; newest labels remain pending until complete.",
  "provenance": "src.research.response_dense_birth; source manifests, event hashes, exact atom identities and extracted bytes bound in per-source receipts.",
  "lineage": "83 dense production trajectories from original fixed Al64 melt ancestors; inherited roles54/9/9/11; no source resplit or independent-count inflation.",
  "evidence": [
    "configs/analysis/response_dense_birth_20261006.json",
    "docs/response_dense_birth.md",
    "${storage:training_storage}/response_velocity/native500-f64-20261006-v1/analyses/dense010-birth-v1/technical/plan.json"
  ],
  "limitations": [
    "Case/control sampling is retrospective; not natural nucleation risk.",
    "Small crystal precursors allowed; no eight-atom component within8A at each input.",
    "MD positions and velocities stored float16, converted to float32 inference.",
    "Sparse, shooting, historical partial, million-atom and relaxed populations excluded from this native main-production protocol."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 54 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-response-dense010-birth-20261006-6eb08b6c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
