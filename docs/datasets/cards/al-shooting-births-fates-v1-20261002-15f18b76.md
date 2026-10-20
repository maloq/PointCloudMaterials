# Al shooting full-cell birth and dissolution catalogue

[All datasets](../README.md) · [Browsable card](al-shooting-births-fates-v1-20261002-15f18b76.html) · [Full metadata](../records/al-shooting-births-fates-v1-20261002-15f18b76.json)

All completed registered Al shooting futures; exact0.30ps full-cell PTM/lineage search, parent-present embryo and failed-cluster fate labels. Original nested plus fixed24 counted as one future.

- ID: `al-shooting-births-fates-v1-20261002`
- Materials: Al
- Classification: **derived**; role: label_audit
- Location: `/store/PERSO/vmorozov/experiments/nucleus_fates/al-shooting-20261002`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.796 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al shooting full-cell birth and dissolution catalogue",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "label_audit",
  "classification": "derived",
  "description": "All completed registered Al shooting futures; exact0.30ps full-cell PTM/lineage search, parent-present embryo and failed-cluster fate labels. Original nested plus fixed24 counted as one future.",
  "provenance": "src/research/crystallization_origin/shooting.py; checksum-bound campaign/outcome/binary manifests and actual timelines",
  "ancestry": "Original parent hashes and canonical source ancestors retained; sibling shots and parent states from one source are correlated; no split is assigned.",
  "evidence": [
    "configs/analysis/shooting_births_20261002.json",
    "docs/metrics/shooting_births.md",
    "${dataset:al-shooting-births-fates-v1-20261002}/technical/plan.json",
    "${dataset:al-shooting-births-fates-v1-20261002}/technical/complete.json"
  ],
  "limitations": [
    "Operational crystal episodes, not validated critical nuclei.",
    "Separate cadence and actual durations from original0.75ps Al64 audit.",
    "Outcome-selected nested parents/continuations remain identifiable; not an unbiased rate ensemble. Deleted smoke campaigns are excluded with a recorded retirement receipt.",
    "Candidate histories have not yet passed strict pre-appearance eligibility; no predictor fits added."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al-shooting-births-fates-v1"] |
| cadence_ps | [0.3] |
| ptm_rmsd_cutoff | [0.1] |
| lineage | [{"interface_distance_A": 8.0, "local_birth_radius_A": 8.0, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.6, "thresholds": [{"name": "primary", "persistence_frames": 6, "persistence_ps": 1.5, "size": 64}]}, {"interface_distance_A": 8.0, "local_birth_radius_A": 8.0, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.6}] |
| material | ["Al"] |
| parent_id | ["parent_000_T400_cluster00020_transition_candidate", "parent_000_T400_v35831_pre_nucleation_12ps", "parent_001_T400_cluster00029_transition_candidate", "parent_001_T400_v35831_pre_nucleation_3ps", "parent_002_T400_cluster00038_transition_candidate", "parent_002_T400_v35839_pre_nucleation_12ps", "parent_003_T400_cluster00046_transition_candidate", "parent_003_T400_v35839_pre_nucleation_3ps", "parent_004_T400_cluster00055_transition_candidate", "parent_004_T400_v35851_pre_nucleation_12ps", "parent_005_T400_cluster00064_transition_candidate", "parent_005_T400_v35851_pre_nucleation_3ps"] … (76 values; see JSON) |
| temperature_K | [400.0, 450.0, 500.0] |
| timestep_fs | [3.0] |
| storage_dtype | ["float16", "float32"] |
| atom_count | [70304] |
| frame_count | [101, 103, 108, 113, 115, 117, 122, 131, 133, 134, 138, 143] … (27 values; see JSON) |
| source_run_id | ["al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_005_velocity_35879", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_005_velocity_35879"] … (49 values; see JSON) |
| source_split | ["final_validation", "model_selection", "optimization", "train", "validation"] |
| duration_ps | [15.0, 24.0, 48.0] |
| ensemble | ["fixed-cell Langevin NVT", "fixed-cell stochastic canonical NVT"] |
| sample_interval_ps | [0.3] |
| velocity_seed | [101485013, 102979543, 103028903, 103617398, 103742080, 106686085, 111585476, 112138207, 112506268, 112956818, 116558962, 119253255] … (680 values; see JSON) |

## Evidence

All 827 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-shooting-births-fates-v1-20261002-15f18b76.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
