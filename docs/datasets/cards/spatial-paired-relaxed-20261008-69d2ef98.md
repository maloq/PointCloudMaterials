# Paired dynamic and static relaxed structural training collection

[All datasets](../README.md) · [Browsable card](spatial-paired-relaxed-20261008-69d2ef98.html) · [Full metadata](../records/spatial-paired-relaxed-20261008-69d2ef98.json)

1339157 training neighborhoods; exact dynamic hot/cold pairs plus 24 user-confirmed already-relaxed static Mg/Al/Zr/Ta configurations. All 2670500 target rows are complete, including held-out and diagnostic roles; all 6571 Ni quenches are complete.

- ID: `spatial-paired-relaxed-20261008`
- Materials: Al, Mg, Ni, Ta, Zr
- Classification: **prepared**; role: structural_pretraining
- Location: `/home/ids/vmorozov/training-cache/spatial-pretraining/paired-relaxed-static-20261008-v2`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 8.790 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Paired dynamic and static relaxed structural training collection",
  "materials": [
    "Al",
    "Ni",
    "Mg",
    "Ta",
    "Zr"
  ],
  "potential_ids": [
    "al-lee2003-meam",
    "ni-mendelev2012-eam"
  ],
  "role": "structural_pretraining",
  "classification": "prepared",
  "description": "1339157 training neighborhoods; exact dynamic hot/cold pairs plus 24 user-confirmed already-relaxed static Mg/Al/Zr/Ta configurations. All 2670500 target rows are complete, including held-out and diagnostic roles; all 6571 Ni quenches are complete.",
  "provenance": "configs/data/spatial_paired_relaxed_20261008.json; shared indexed quenches and retained paired caches",
  "ancestry": "Inherited dynamic roles; static archived material groups remain train-only, with unknown within-material preparation ancestry. No fitting/held-out group overlap; selection/test pooled for diagnostics only.",
  "identity": "3793681fcdba5f7a359302ceac3e5e3b14bdc76ff5ca32b89a020bdc3d00a6db",
  "planned_counts": {
    "train/Mg": 62916,
    "train/Al": 216861,
    "train/Zr": 61440,
    "train/Ta": 510260,
    "evaluation/Al": 90511,
    "calibration/Al": 28672,
    "diagnostic/Al": 17664,
    "train/Ni": 487680,
    "evaluation/Ni": 1194496
  },
  "precision_rows": {
    "static_float32": 636152,
    "prequantization_float32": 149265,
    "float16_cell_derived_float32": 202907,
    "native_restart": 1682176
  },
  "evidence": [
    "docs/datasets/spatial_paired_relaxed.md",
    "/store/PERSO/vmorozov/experiments/spatial_architecture/esen-small-relaxed-static-20261008-v2/technical/launch.json",
    "${storage:cache}/spatial-pretraining/paired-relaxed-static-20261008-v2/complete.json",
    "configs/spatial_architecture/esen_small_relaxed_static_20261009.json"
  ],
  "limitations": [
    "Ni component is birth-window sampled, not outcome blind.",
    "Historical float16-cell-derived Al patches have explicit precision; prequantization float32 patches have deduplication priority.",
    "Static generating potentials and minimization settings are unknown; no new relaxation is run.",
    "Held-out monitoring covers Al/Ni, not Mg/Ta/Zr generalization."
  ],
  "static_relaxation_status": "Explicit user confirmation on 2026-10-08; generating potential and minimizer unknown.",
  "launch_jobs": {
    "al": "1029528",
    "ni": "1029529",
    "publish": "1029530",
    "queued": "1029531"
  },
  "completed_counts": {
    "train/Mg": 62916,
    "train/Al": 216861,
    "train/Zr": 61440,
    "train/Ta": 510260,
    "evaluation/Al": 90511,
    "calibration/Al": 28672,
    "diagnostic/Al": 17664,
    "train/Ni": 487680,
    "evaluation/Ni": 1194496
  },
  "training_retry": {
    "prior_job": "1029531",
    "prior_state": "failed_before_training",
    "reason": "Probe transformed positions inside BF16 autocast; corrected to FP32 geometry without changing threshold or training",
    "output": "${storage:training_storage}/spatial_architecture/esen-small-relaxed-static-20261009-v3",
    "state": "superseded_at_user_request_for_two_gpus",
    "continuation_job": "1030439",
    "local_allocation": "1029087",
    "local_host": "node58",
    "local_gpu": 0,
    "launch_receipt": "/store/PERSO/vmorozov/experiments/spatial_architecture/esen-small-relaxed-static-20261009-v3/technical/launch.json",
    "continuation_state": "cancelled_1030439",
    "next_config": "configs/spatial_architecture/esen_small_relaxed_static_2gpu_20261009.json"
  },
  "two_gpu_training": {
    "state": "running",
    "config": "configs/spatial_architecture/esen_small_relaxed_static_2gpu_20261009.json",
    "output": "${storage:training_storage}/spatial_architecture/esen-small-relaxed-static-2gpu-20261009-v4",
    "launch_receipt": "/store/PERSO/vmorozov/experiments/spatial_architecture/esen-small-relaxed-static-2gpu-20261009-v4/technical/launch.json",
    "local_host": "node58",
    "local_allocation": "1029087",
    "devices": [
      0,
      1
    ],
    "continuation_job": "1030471",
    "global_batch": 256,
    "rows_per_gpu": 128,
    "updates": 62784,
    "epochs": 12
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Zr-archived-root", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_005_velocity_35879", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839"] … (220 values; see JSON) |
| material | ["Al", "Mg", "Ni", "Ta", "Zr"] |
| protocol | ["fixed_material_cutoff_al_reference_v1", "paired-and-static-relaxed-spatial-esen-20261008-v2", "spatial_esen_small_ni_cna_12epoch_v1"] |
| seed | [20261007, 20261008] |
| radius | [8.0] |
| materials | [["Ni"]] |

## Evidence

All 12218 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/spatial-paired-relaxed-20261008-69d2ef98.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
