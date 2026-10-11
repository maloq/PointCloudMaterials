# All velocity-bearing 0.10 ps birth views for October 6 response encoders

[All datasets](../README.md) · [Browsable card](al-response-all010-birth-20261006-65d1a6a1.html) · [Full metadata](../records/al-response-all010-birth-20261006-65d1a6a1.json)

92 exact 0.10 ps labelled trajectory views with saved velocities across three pinned v3 releases: 88 native 0.10 ps and four views of native 0.01 ps histories; includes seven historical partial archives. Frozen Q/QV encoders, exact 0.1/0.5/1 ps inputs before root emergence, four matched liquid controls.

- ID: `al-response-all010-birth-20261006`
- Materials: Al
- Classification: **derived**; role: frozen_encoder_transfer_inputs
- Location: `/home/ids/vmorozov/training-cache/response-birth/all010-v3-20261006-v2`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.013 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "All velocity-bearing 0.10 ps birth views for October 6 response encoders",
  "materials": [
    "Al"
  ],
  "role": "frozen_encoder_transfer_inputs",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "92 exact 0.10 ps labelled trajectory views with saved velocities across three pinned v3 releases: 88 native 0.10 ps and four views of native 0.01 ps histories; includes seven historical partial archives. Frozen Q/QV encoders, exact 0.1/0.5/1 ps inputs before root emergence, four matched liquid controls.",
  "provenance": "src.research.response_dense_birth; 54 checksum-bound preparations reused from the superseded native-production evaluation. Source manifests, event identities, original frame mappings and extracted Q/V bytes retained.",
  "lineage": "Fixed Al64 ancestor roles inherited: 60/10/9/13 train/selection/calibration/test trajectory views. Related archives are not independent melts. Exact repeated event/center/three-lead QV observations deduplicated; physical events equally weighted; whole-ancestor bootstrap.",
  "evidence": [
    "configs/analysis/response_all010_birth_20261006.json",
    "docs/response_dense_birth.md",
    "${storage:training_storage}/response_velocity/native500-f64-20261006-v1/analyses/all010-birth-v2/technical/plan.json",
    "${storage:training_storage}/response_velocity/native500-f64-20261006-v1/analyses/all010-birth-v2/technical/launch.json"
  ],
  "limitations": [
    "Seven million-atom views and one older 100k-atom view excluded because velocities are absent; no replacement velocities.",
    "Eligibility and complete follow-up still required; 92 views do not imply 92 independent events or nonempty rows from every source.",
    "Latest October 6 lineage labels pending at launch; independent upstream jobs unchanged.",
    "Retrospective matched case/control task; not population nucleation risk. Small crystal precursors allowed; no eight-atom component within 8 A at inputs.",
    "Positions and velocities archived float16; float32 encoder inference. Native cadence retained; no interpolation.",
    "Other cadences, shooting, relaxed views and collections outside these three frozen releases remain separate."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 93 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-response-all010-birth-20261006-65d1a6a1.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
