# al meam predictive dynamics fixed15 smoke 1parent 16branches float32 20260904

[All datasets](../README.md) · [Browsable card](al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904-af170dc2.html) · [Full metadata](../records/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904-af170dc2.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904`
- Materials: Al
- Classification: **fixture**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 21; these are not independent-source counts.
- Stored frames: 1130; duplicate-group records: 1
- Allocated storage, excluding registered nested datasets: 2.673 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "al meam predictive dynamics fixed15 smoke 1parent 16branches float32 20260904",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "fixture",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904}"
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
| parent_id | ["smoke_shifted_parent_T400_ancestor_branch004_t12ps"] |
| source_run_id | ["al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831"] |
| source_split | ["optimization"] |
| temperature_K | [400.0] |
| velocity_seed | [181574221, 324648890, 51548785, 528143416, 578737219, 578757272, 626402527, 753808016] |
| frame_count | [30, 41, 51, 81] |
| storage_dtype | ["float32"] |
| duration_ps | [15.0] |
| ensemble | ["fixed-cell stochastic canonical NVT"] |
| sample_interval_ps | [0.3] |
| timestep_fs | [3.0] |
| scientific_contract | [{"bootstrap_unit": "root_source_lineage", "exact_extension_requires_same_mpi_ranks": 24, "exact_short_to_extension_restart": true, "exact_source_restart": false, "first_passage_does_not_stop_fixed_trajectory": true, "interpretation": "Fixed-cell stochastic canonical futures conditioned on immutable positions, cell, history, shooting temperature, sampled momentum, and thermostat stream.", "no_equilibration_after_branching": true, "overlapping_windows_are_correlated": true}] |
| temperatures_K | [[400.0]] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)", "positions decode to float32 coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 75 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904-af170dc2.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
