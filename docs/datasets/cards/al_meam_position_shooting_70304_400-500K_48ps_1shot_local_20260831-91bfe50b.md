# Al · 48 ps shooting, one-shot supplement

[All datasets](../README.md) · [Browsable card](al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831-91bfe50b.html) · [Full metadata](../records/al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831-91bfe50b.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 40; these are not independent-source counts.
- Stored frames: 6440; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 10.448 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al \u00b7 48 ps shooting, one-shot supplement",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831}"
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
| scientific_contract | [{"exact_restart": false, "interpretation": "Independent fixed-cell Langevin-NVT futures conditioned on each archived position under the same 2NN-MEAM Hamiltonian.", "no_equilibration_after_branching": true, "reason": "Archived sources contain positions and cells but not velocities or serialized NPT thermostat/barostat state."}] |
| atom_count | [70304] |
| temperatures_K | [[400.0, 450.0, 500.0]] |
| ensemble | ["fixed-cell Langevin NVT"] |
| timestep_fs | [3.0] |
| duration_ps | [48.0] |
| sample_interval_ps | [0.3] |
| temperature_K | [400.0, 450.0, 500.0] |
| velocity_seed | [103617398, 123518261, 124750126, 157216088, 161401839, 184668838, 225295350, 230024520, 245147714, 246135621, 258207404, 272577041] … (48 values; see JSON) |
| split | ["train", "validation"] |
| parent_id | ["parent_000_T400_v35831_pre_nucleation_12ps", "parent_001_T400_v35831_pre_nucleation_3ps", "parent_002_T400_v35839_pre_nucleation_12ps", "parent_003_T400_v35839_pre_nucleation_3ps", "parent_004_T400_v35851_pre_nucleation_12ps", "parent_005_T400_v35851_pre_nucleation_3ps", "parent_006_T400_v35863_pre_nucleation_12ps", "parent_007_T400_v35863_pre_nucleation_3ps", "parent_008_T400_v35869_pre_nucleation_12ps", "parent_009_T400_v35869_pre_nucleation_3ps", "parent_010_T400_v35879_pre_nucleation_12ps", "parent_011_T400_v35879_pre_nucleation_3ps"] … (40 values; see JSON) |
| source_run_id | ["al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_005_velocity_35879", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_005_velocity_35879"] … (20 values; see JSON) |
| source_split | ["train", "validation"] |
| frame_count | [161] |
| storage_dtype | ["float32"] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 202 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831-91bfe50b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-24T10:36:51.012083+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
