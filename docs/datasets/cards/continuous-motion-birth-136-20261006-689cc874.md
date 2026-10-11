# Continuous-motion nucleus-birth transfer at1/3/6ps

[All datasets](../README.md) · [Browsable card](continuous-motion-birth-136-20261006-689cc874.html) · [Full metadata](../records/continuous-motion-birth-136-20261006-689cc874.json)

New exact1/3/6ps observations from the92 pinned saved-velocity v3 source views; frozen continuous-motion encoders and1:4 matched root-birth controls. Four historical source views have unknown generating potential and retain that uncertainty.

- ID: `continuous-motion-birth-136-20261006`
- Materials: Al
- Classification: **derived**; role: frozen_encoder_transfer_inputs
- Location: `/home/ids/vmorozov/training-cache/continuous-motion/birth-136-20261006-v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.009 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Continuous-motion nucleus-birth transfer at1/3/6ps",
  "materials": [
    "Al"
  ],
  "role": "frozen_encoder_transfer_inputs",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "New exact1/3/6ps observations from the92 pinned saved-velocity v3 source views; frozen continuous-motion encoders and1:4 matched root-birth controls. Four historical source views have unknown generating potential and retain that uncertainty.",
  "provenance": "src.research.continuous_motion.birth reuses response_dense_birth source extraction, event alias rules and training-only pooled readouts. Native manifests, timelines and event labels checksum bound.",
  "lineage": "Existing fixed Al64 melt ancestry and original roles. Validation/calibration/test pooled only for evaluation; no independent new MD sources.",
  "evidence": [
    "configs/analysis/continuous_motion_birth_20261006.json",
    "docs/continuous_motion.md",
    "${storage:training_storage}/continuous_motion/all010-20261006-v3/analyses/birth-136-v1/technical/plan.json"
  ],
  "limitations": [
    "Event counts pending preparation; filters and censored followup remain explicit.",
    "No velocity-free million-atom or other-cadence/relaxed inputs.",
    "Retrospective matched case/control prediction, not population nucleation risk.",
    "Birth cache velocities are A/fs; MotionModel inference explicitly converts to A/ps."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 93 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/continuous-motion-birth-136-20261006-689cc874.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
