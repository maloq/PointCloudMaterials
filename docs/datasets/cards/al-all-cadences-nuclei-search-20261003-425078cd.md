# All-Al cluster birth and fate search, exact 0.75 / 0.10 / 0.30 ps views

[All datasets](../README.md) · [Browsable card](al-all-cadences-nuclei-search-20261003-425078cd.html) · [Full metadata](../records/al-all-cadences-nuclei-search-20261003-425078cd.json)

Frozen all-Al registry inventory, four structural detectors, physical-time cluster tracking and paired cadence comparisons. No prediction-eligibility filter.

- ID: `al-all-cadences-nuclei-search-20261003`
- Materials: Al
- Classification: **derived**; role: label_audit
- Location: `/store/PERSO/vmorozov/experiments/nuclei_search/al-all-20261003-v2`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.538 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "All-Al cluster birth and fate search, exact 0.75 / 0.10 / 0.30 ps views",
  "materials": [
    "Al"
  ],
  "role": "label_audit",
  "classification": "derived",
  "potential_ids": [],
  "description": "Frozen all-Al registry inventory, four structural detectors, physical-time cluster tracking and paired cadence comparisons. No prediction-eligibility filter.",
  "provenance": "src.research.crystallization_origin.search; source manifests, producer metadata and checksums bound in technical/plan.json",
  "lineage": "Historical sources, shooting branches, dense descendants and historical prefixes retain their original ancestors. Protocol/detector views are correlated; unresolved ancestry stays explicit.",
  "evidence": [
    "configs/analysis/al_nuclei_search_20261003.json",
    "docs/al_nuclei_search.md",
    "docs/metrics/al_nuclei_search.md",
    "${dataset:al-all-cadences-nuclei-search-20261003}/technical/plan.json",
    "${dataset:al-all-cadences-nuclei-search-20261003}/technical/launch.json"
  ],
  "limitations": [
    "Generating potentials are recorded per source; this mixed-potential collection has no single potential.",
    "Operational embryo/establishment thresholds, not validated critical nuclei.",
    "Only completed available arrays with documented exact timelines enter the queue.",
    "Raw MD outcomes; relaxed/static views do not supply independent temporal events.",
    "Historic prefixes and cadence views must not be summed into independent event counts.",
    "2026-10-07 user-authorized cleanup: obsolete v2 per-source event/interaction/trajectory/size payloads and redundant events.csv exports were deleted. Frozen extraction plans, structural extraction, code, summaries and provenance remain; use the consolidated al-nuclei-birth dataset for current labels."
  ],
  "storage_cleanup": {
    "date": "2026-10-07",
    "payload_files_removed": 8913,
    "receipt": "docs/storage/store_cleanup_20261007/payload-cleanup-receipt.json"
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al_cluster_births_all_cadences_v2", "al_dense_half_stop_001ps_v1", "al_dense_half_stop_010ps_v1", "al_dense_replay_v1", "al_spontaneous_birth_sources_v1", "predictive_memory_precision_sources_v1"] |
| ptm_rmsd_cutoff | [0.1] |
| timestep_fs | [1.0, 2.0, 3.0] |
| temperature_K | [400, 400.0, 410, 420, 430, 440, 450, 450.0, 460, 470, 480, 490] … (17 values; see JSON) |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (151 values; see JSON) |
| frame_count | [101, 103, 1061, 108, 113, 115, 117, 122, 1241, 1251, 1261, 131] … (65 values; see JSON) |
| atom_count | [100000, 1000000, 1048576, 70304] |
| parent_id | ["parent_000_T400_cluster00020_transition_candidate", "parent_000_T400_v35831_pre_nucleation_12ps", "parent_001_T400_cluster00029_transition_candidate", "parent_001_T400_v35831_pre_nucleation_3ps", "parent_002_T400_cluster00038_transition_candidate", "parent_002_T400_v35839_pre_nucleation_12ps", "parent_003_T400_cluster00046_transition_candidate", "parent_003_T400_v35839_pre_nucleation_3ps", "parent_004_T400_cluster00055_transition_candidate", "parent_004_T400_v35851_pre_nucleation_12ps", "parent_005_T400_cluster00064_transition_candidate", "parent_005_T400_v35851_pre_nucleation_3ps"] … (76 values; see JSON) |
| source_run_id | ["al-main-001ps-20261001-source0886", "al-main-001ps-20261001-source1004", "al-main-010ps-20260927-source0884", "al-main-010ps-20260927-source0885", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_005_velocity_35879", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839"] … (53 values; see JSON) |
| source_split | ["final_validation", "model_selection", "optimization", "train", "validation"] |
| storage_dtype | ["float16", "float32"] |
| velocity_seed | [101485013, 102979543, 103028903, 103617398, 103742080, 106686085, 111585476, 112138207, 112506268, 112956818, 116558962, 119253255] … (680 values; see JSON) |
| root_lineage | ["independent_melt_144370470", "independent_melt_160380922", "independent_melt_180989530", "independent_melt_184384186", "independent_melt_186372838", "independent_melt_200290231", "independent_melt_203886074", "independent_melt_206158192", "independent_melt_22359867", "independent_melt_235751553", "independent_melt_248699045", "independent_melt_24993146"] … (53 values; see JSON) |
| split | ["calibration", "selection", "test", "train"] |
| timestep_ps | [0.002, 0.003] |
| material | ["Al"] |
| cadence_ps | [0.3] |
| ensemble | ["fixed-cell Langevin NVT"] |
| duration_ps | [48.0] |
| sample_interval_ps | [0.3] |

## Evidence

All 4470 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-all-cadences-nuclei-search-20261003-425078cd.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
