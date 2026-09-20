# al meam position shooting 70304 400-500K 48ps 4shot topup to16 20260903 prepared without legacy window spec

[All datasets](../README.md) · [Browsable card](al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_wind-0ccd5bca.html) · [Full metadata](../records/al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_wind-0ccd5bca.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_window_spec`
- Materials: Al
- Classification: **prepared**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_window_spec`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.003 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "al meam position shooting 70304 400-500K 48ps 4shot topup to16 20260903 prepared without legacy window spec",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "prepared",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_window_spec}"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Siblings share parent/source ancestry; mixed shooting protocols and converted continuations are not independent sources."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| ptm_rmsd_cutoff | [0.1] |
| parent_id | ["parent_000_T400_v35831_pre_nucleation_12ps", "parent_001_T400_v35831_pre_nucleation_3ps", "parent_002_T400_v35839_pre_nucleation_12ps", "parent_003_T400_v35839_pre_nucleation_3ps", "parent_004_T400_v35851_pre_nucleation_12ps", "parent_005_T400_v35851_pre_nucleation_3ps", "parent_006_T400_v35863_pre_nucleation_12ps", "parent_007_T400_v35863_pre_nucleation_3ps", "parent_008_T400_v35869_pre_nucleation_12ps", "parent_009_T400_v35869_pre_nucleation_3ps", "parent_010_T400_v35879_pre_nucleation_12ps", "parent_011_T400_v35879_pre_nucleation_3ps"] … (40 values; see JSON) |
| source_run_id | ["al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_005_velocity_35879", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_005_velocity_35879"] … (20 values; see JSON) |
| source_split | ["train", "validation"] |
| temperature_K | [400.0, 450.0, 500.0] |
| velocity_seed | [104862625, 108083334, 114344128, 116558962, 117126948, 11769494, 138213247, 14134890, 14505840, 147869426, 150716425, 151057967] … (160 values; see JSON) |
| duration_ps | [48.0] |
| ensemble | ["fixed-cell Langevin NVT"] |
| sample_interval_ps | [0.3] |
| storage_dtype | ["float32"] |
| timestep_fs | [3.0] |
| scientific_contract | [{"bootstrap_unit": "root_source_lineage", "exact_restart": false, "first_passage_does_not_stop_fixed_trajectory": true, "interpretation": "Fixed-cell Langevin-NVT futures conditioned on immutable positions, cell, history, shooting temperature, sampled momentum, and thermostat stream.", "no_equilibration_after_branching": true, "overlapping_windows_are_correlated": true}] |
| temperatures_K | [[400.0, 450.0, 500.0]] |

## Evidence

All 202 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_wind-0ccd5bca.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T01:13:02.416719+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
