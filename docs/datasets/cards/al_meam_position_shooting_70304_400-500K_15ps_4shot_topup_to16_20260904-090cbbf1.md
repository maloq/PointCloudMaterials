# Al · accepted 15 ps shooting top-up

[All datasets](../README.md) · [Browsable card](al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904-090cbbf1.html) · [Full metadata](../records/al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904-090cbbf1.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 160; these are not independent-source counts.
- Stored frames: 8160; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 13.713 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al \u00b7 accepted 15 ps shooting top-up",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904}"
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
| velocity_seed | [112138207, 112956818, 119253255, 119394763, 12042714, 126789114, 132354054, 13459509, 158621729, 163493059, 169382709, 172638987] … (160 values; see JSON) |
| duration_ps | [15.0] |
| ensemble | ["fixed-cell stochastic canonical NVT"] |
| sample_interval_ps | [0.3] |
| storage_dtype | ["float32"] |
| timestep_fs | [3.0] |
| scientific_contract | [{"bootstrap_unit": "root_source_lineage", "exact_extension_requires_same_mpi_ranks": 24, "exact_short_to_extension_restart": true, "exact_source_restart": false, "first_passage_does_not_stop_fixed_trajectory": true, "interpretation": "Fixed-cell stochastic canonical futures conditioned on immutable positions, cell, history, shooting temperature, sampled momentum, and thermostat stream.", "no_equilibration_after_branching": true, "overlapping_windows_are_correlated": true}] |
| temperatures_K | [[400.0, 450.0, 500.0]] |
| frame_count | [51] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 682 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904-090cbbf1.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
