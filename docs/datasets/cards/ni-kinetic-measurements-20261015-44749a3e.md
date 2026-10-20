# Ni native kinetic-latent measurements and exact local phase annotations

[All datasets](../README.md) · [Browsable card](ni-kinetic-measurements-20261015-44749a3e.html) · [Full metadata](../records/ni-kinetic-measurements-20261015-44749a3e.json)

Derived train-only affine/asinh/affine normalization of the existing 14,400 by 64 by 244 BO/TDA/density/velocity measurement tensor. Same central atom at native 0.1 ps, nearest 80 reselected per observation, 0.3 ps history and 6 ps future. Preserved 16 training ancestors/3,840 windows; former selection and test pooled into 44 test ancestors/10,560 windows. A resumable annotation view retains exact central PTM type, local crystalline fraction and central tracked component size. Reuses the separate 19,800-row birth assay; no new independent MD or raw trajectory copy.

- ID: `ni-kinetic-measurements-20261015`
- Materials: Ni
- Classification: **research**; role: motion_training_and_evaluation
- Location: `/home/ids/vmorozov/training-cache/ni-kinetic-latent/native-20261015-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.845 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni native kinetic-latent measurements and exact local phase annotations",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "motion_training_and_evaluation",
  "classification": "research",
  "description": "Derived train-only affine/asinh/affine normalization of the existing 14,400 by 64 by 244 BO/TDA/density/velocity measurement tensor. Same central atom at native 0.1 ps, nearest 80 reselected per observation, 0.3 ps history and 6 ps future. Preserved 16 training ancestors/3,840 windows; former selection and test pooled into 44 test ancestors/10,560 windows. A resumable annotation view retains exact central PTM type, local crystalline fraction and central tracked component size. Reuses the separate 19,800-row birth assay; no new independent MD or raw trajectory copy.",
  "ancestry": "Original 60 Ni ancestors unchanged; native 16 train/44 test. Separate 90-source birth contract 28 train/62 test, with actual assay rows from 21 train/42 test ancestors.",
  "source_plan_sha256": "91812d6d7ab012b1a24221cf9b40d90ccdb8105389097b7c242f39adcc2ca5d8",
  "birth_history_plan_sha256": "122296aa06f638acc8a4224eaf9335b15f51ef11216afed03a455bc4c70756c8",
  "sample_interval_ps": 0.1,
  "evidence": [
    "configs/ni_latent_sim/kinetic_latent_20261015.json",
    "docs/ni_kinetic_latent.md",
    "${storage:training_storage}/ni_kinetic_latent/native-20261015-v1/technical/plan.json",
    "${storage:cache}/ni-kinetic-latent/native-20261015-v1/data.json",
    "${storage:cache}/ni-kinetic-latent/native-20261015-v1/labels.json"
  ],
  "limitations": [
    "4,800 liquid-persistent and 4,800 growing-front windows are outcome-enriched; cohort frequencies are not natural incidence.",
    "Actual birth inputs use present measurements; cached four-frame birth history has fixed present neighbor IDs, unlike native reselected neighborhoods.",
    "PTM labels and connected components annotate evaluation only. Component size 0 means untracked below 4 atoms, not necessarily liquid.",
    "Annotation completion is determined by labels.json, independently of normalized data availability.",
    "Generated independent local measurement paths do not establish full-cell connected nucleus formation or equilibrium kinetics."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 60 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-kinetic-measurements-20261015-44749a3e.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
