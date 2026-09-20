# Al · fixed-24 ps nested continuations

[All datasets](../README.md) · [Browsable card](al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible-a8df15b6.html) · [Full metadata](../records/al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible-a8df15b6.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible`
- Materials: Al
- Classification: **review_required**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 144; these are not independent-source counts.
- Stored frames: 11664; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 9.665 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al \u00b7 fixed-24 ps nested continuations",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "review_required",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible}"
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
| parent_id | ["parent_000_T400_cluster00020_transition_candidate", "parent_001_T400_cluster00029_transition_candidate", "parent_002_T400_cluster00038_transition_candidate", "parent_003_T400_cluster00046_transition_candidate", "parent_004_T400_cluster00055_transition_candidate", "parent_005_T400_cluster00064_transition_candidate", "parent_006_T400_cluster00073_transition_candidate", "parent_007_T400_cluster00075_transition_candidate", "parent_008_T400_cluster00099_transition_candidate", "parent_009_T400_cluster00099_transition_candidate", "parent_010_T400_cluster00005_liquid_control", "parent_011_T400_cluster00340_crystal_control"] … (36 values; see JSON) |
| ptm_rmsd_cutoff | [0.1] |
| source_run_id | ["source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_001_velocity_35839", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_004_velocity_35869", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_005_velocity_35879", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_004_velocity_35869", "source_group_00/al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_005_velocity_35879"] … (29 values; see JSON) |
| source_split | ["final_validation", "model_selection", "optimization"] |
| storage_dtype | ["float16", "float32"] |
| temperature_K | [400.0, 450.0, 500.0] |
| timestep_fs | [3.0] |
| duration_ps | [24.0] |
| ensemble | ["fixed-cell Langevin NVT"] |
| sample_interval_ps | [0.3] |
| scientific_contract | [{"continuation_randomness": "Positions and velocities are retained at the restart; only the Markov Langevin noise stream receives a new checksum-derived seed.", "first_passage_labels": "Original outcomes and times are copied without reclassification.", "no_fabricated_frames": "Every frame is either from the original nested LAMMPS path or a LAMMPS continuation from its saved first-passage restart."}] |
| frame_count | [81] |
| coordinate_convention | ["positions decode to float32 coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 578 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible-a8df15b6.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T01:13:02.416719+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
