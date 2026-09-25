# superseded preparation al meam predictive dynamics fixed15 smoke wave979929 20260904T0955Z

[All datasets](../README.md) · [Browsable card](superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z-be4065aa.html) · [Full metadata](../records/superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z-be4065aa.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z`
- Materials: Unknown
- Classification: **fixture**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.001 GiB
- Missing metadata: materials

## Notes and relationships

```json
{
  "title": "superseded preparation al meam predictive dynamics fixed15 smoke wave979929 20260904T0955Z",
  "materials": [],
  "role": "raw_dynamics",
  "classification": "fixture",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z}"
  ]
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
| velocity_seed | [238906586, 28418216, 353135530, 449133463, 524937898, 548560715, 594947822, 872881233] |
| frame_count | [41] |
| storage_dtype | ["float32"] |
| duration_ps | [15.0] |
| ensemble | ["fixed-cell stochastic canonical NVT"] |
| sample_interval_ps | [0.3] |
| timestep_fs | [3.0] |
| scientific_contract | [{"bootstrap_unit": "root_source_lineage", "exact_extension_requires_same_mpi_ranks": 24, "exact_short_to_extension_restart": true, "exact_source_restart": false, "first_passage_does_not_stop_fixed_trajectory": true, "interpretation": "Fixed-cell stochastic canonical futures conditioned on immutable positions, cell, history, shooting temperature, sampled momentum, and thermostat stream.", "no_equilibration_after_branching": true, "overlapping_windows_are_correlated": true}] |
| temperatures_K | [[400.0]] |
| coordinate_convention | ["positions decode to float32 coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 21 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z-be4065aa.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
