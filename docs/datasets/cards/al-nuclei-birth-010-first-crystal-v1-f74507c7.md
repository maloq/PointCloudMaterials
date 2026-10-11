# Additional first-crystal and persistent-appearance labels for Al nuclei births

[All datasets](../README.md) · [Browsable card](al-nuclei-birth-010-first-crystal-v1-f74507c7.html) · [Full metadata](../records/al-nuclei-birth-010-first-crystal-v1-f74507c7.json)

Versioned permanent observation labels at 0.1/0.5/1/3/6 ps. Copies computed PTM arrays and onset labels from fixed-time uniformly sampled local80 sites. First crystal counts transient order; persistence requires the same atom in three consecutive frames. Unknown generating-potential metadata remain explicit in copied source provenance. No new simulations or independent ancestry. One final publication follows sealing of all 456 windows and the final cohort; no incremental dataset updates.

- ID: `al-nuclei-birth-010-first-crystal-v1`
- Materials: Al
- Classification: **derived**; role: additional_prediction_labels
- Location: `/store/PERSO/vmorozov/experiments/nuclei_birth/al/additional_labels/first-crystal-010-v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.418 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Additional first-crystal and persistent-appearance labels for Al nuclei births",
  "materials": [
    "Al"
  ],
  "role": "additional_prediction_labels",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Versioned permanent observation labels at 0.1/0.5/1/3/6 ps. Copies computed PTM arrays and onset labels from fixed-time uniformly sampled local80 sites. First crystal counts transient order; persistence requires the same atom in three consecutive frames. Unknown generating-potential metadata remain explicit in copied source provenance. No new simulations or independent ancestry. One final publication follows sealing of all 456 windows and the final cohort; no incremental dataset updates.",
  "provenance": "${dataset:al-nuclei-birth-010-first-crystal-v1}/manifest.json",
  "lineage": "114 pinned velocity-bearing source views and 106 recorded ancestors, with unchanged original roles. Source/frame/center-atom join to the parent; no manufactured nucleus-event IDs. Separate fingerprint leaves original nucleus births unchanged.",
  "evidence": [
    "docs/datasets/nuclei_birth_labels.md",
    "configs/data/nuclei_birth_first_crystal_20261007.json",
    "${dataset:al-nuclei-birth-010-first-crystal-v1}/manifest.json"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cadence_ps | [0.1] |
| horizons_ps | [[0.1, 0.5, 1, 3, 6]] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nuclei-birth-010-first-crystal-v1-f74507c7.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
