# Al · independent melts, 400–500 K

[All datasets](../README.md) · [Browsable card](al_meam_independent_sources_70304_400-500K_30perT_float16_20260902-09a0e5bb.html) · [Full metadata](../records/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902-09a0e5bb.json)

Independently melted Al histories. Current outcomes, not planned counts, establish available runs. Source and descendants retain the declared split.

- ID: `al_meam_independent_sources_70304_400-500K_30perT_float16_20260902`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 90; these are not independent-source counts.
- Stored frames: 72090; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 59.595 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Al \u00b7 independent melts, 400\u2013500 K",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "Independently melted Al histories. Current outcomes, not planned counts, establish available runs. Source and descendants retain the declared split.",
  "evidence": [
    "${dataset:al_meam_independent_sources_70304_400-500K_30perT_float16_20260902}/manifest.json"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| equilibration_duration_ps | [15.0] |
| measurement_duration_ps | [600.0] |
| melt_duration_ps | [300.0] |
| melt_temperature_K | [1325.0] |
| ptm_rmsd_cutoff | [0.1] |
| sample_interval_ps | [0.75] |
| timestep_fs | [3.0] |
| preparation_seed | [114745743, 134462729, 135035943, 13937749, 142641279, 146234076, 147978527, 15458297, 176364202, 197585375, 206158192, 210998254] … (90 values; see JSON) |
| source_split | ["final_validation", "model_selection", "optimization"] |
| temperature_K | [400.0, 450.0, 500.0] |
| velocity_seed | [10275995, 11340456, 123787821, 12639796, 134706744, 137810049, 149757564, 151670672, 16114127, 163531985, 181016004, 185157491] … (90 values; see JSON) |
| scientific_contract | [{"screening_futures_reused_for_evaluation": false, "split_unit": "source run and every screened parent and future descended from it", "structural_independence": "Every source has its own 300 ps full-box melt trajectory from a unique preparation seed before its independently seeded undercooling history."}] |
| split_unit | ["source run and every screened parent and future descended from it"] |
| structural_independence | ["Every source has its own 300 ps full-box melt trajectory from a unique preparation seed before its independently seeded undercooling history."] |
| temperatures_K | [[400.0, 450.0, 500.0]] |
| frame_count | [801] |
| storage_dtype | ["float16"] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 364 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902-09a0e5bb.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
