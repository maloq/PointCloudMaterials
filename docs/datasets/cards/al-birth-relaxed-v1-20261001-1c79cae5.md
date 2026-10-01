# Full-cell relaxed Al birth histories, original cohort v1

[All datasets](../README.md) · [Browsable card](al-birth-relaxed-v1-20261001-1c79cae5.html) · [Full metadata](../records/al-birth-relaxed-v1-20261001-1c79cae5.json)

1030 fixed-box full-cell MEAM quenches; 11793 aligned nearest-80 patches for the original 1475 histories. Original labels, source roles and folds retained.

- ID: `al-birth-relaxed-v1-20261001`
- Materials: Al
- Classification: **derived**; role: derived_quenched_birth_histories
- Location: `/home/ids/vmorozov/training-cache/birth-prediction/relaxed-temporal-20261001`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.011 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Full-cell relaxed Al birth histories, original cohort v1",
  "materials": [
    "Al"
  ],
  "role": "derived_quenched_birth_histories",
  "classification": "derived",
  "description": "1030 fixed-box full-cell MEAM quenches; 11793 aligned nearest-80 patches for the original 1475 histories. Original labels, source roles and folds retained.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "provenance": {
    "config": "configs/birth_prediction/relaxed_temporal_20261001.json",
    "producer": "src/research/birth_prediction/relaxed_data.py",
    "archive": "${storage:archive}/birth_prediction/relaxed-temporal-20261001",
    "dataset_identity": "ab3cbac7a1fefa431e4d55d2eadc1169553654300d6a242f54bd636df53ee886"
  },
  "evidence": [
    "${dataset:al-birth-relaxed-v1-20261001}/manifest.json"
  ],
  "limitations": [
    "Derived from original MD, not independent additional trajectories. Original retrospective sampled sites and 20% case-control prevalence remain.",
    "Full periodic-cell minimization uses broader current-frame computational context. Frozen encoders are not retrained on this input domain."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 601 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-birth-relaxed-v1-20261001-1c79cae5.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
