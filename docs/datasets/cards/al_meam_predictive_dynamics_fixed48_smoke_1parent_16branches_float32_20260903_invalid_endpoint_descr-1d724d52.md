# al meam predictive dynamics fixed48 smoke 1parent 16branches float32 20260903 invalid endpoint descriptor preparation

[All datasets](../README.md) · [Browsable card](al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descr-1d724d52.html) · [Full metadata](../records/al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descr-1d724d52.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descriptor_preparation`
- Materials: Al
- Classification: **fixture**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descriptor_preparation`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 1; these are not independent-source counts.
- Stored frames: 41; duplicate-group records: 1
- Allocated storage, excluding registered nested datasets: 0.069 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "al meam predictive dynamics fixed48 smoke 1parent 16branches float32 20260903 invalid endpoint descriptor preparation",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "fixture",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descriptor_preparation}"
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
| velocity_seed | [185531845, 372192563, 422375434, 434008895, 500096656, 612379165, 724145866, 809779785] |
| frame_count | [41] |
| storage_dtype | ["float32"] |
| duration_ps | [48.0] |
| ensemble | ["fixed-cell Langevin NVT"] |
| sample_interval_ps | [0.3] |
| timestep_fs | [3.0] |
| scientific_contract | [{"bootstrap_unit": "root_source_lineage", "exact_restart": false, "first_passage_does_not_stop_fixed_trajectory": true, "interpretation": "Fixed-cell Langevin-NVT futures conditioned on immutable positions, cell, history, shooting temperature, sampled momentum, and thermostat stream.", "no_equilibration_after_branching": true, "overlapping_windows_are_correlated": true}] |
| temperatures_K | [[400.0]] |
| coordinate_convention | ["positions decode to float32 coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 20 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descr-1d724d52.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T18:31:23.169342+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
