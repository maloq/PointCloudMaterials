# al meam independent sources 70304 510-520K 30perT float16 20260903 prepared before manifest checksum

[All datasets](../README.md) · [Browsable card](al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum-84872e32.html) · [Full metadata](../records/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum-84872e32.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum`
- Materials: Al
- Classification: **prepared**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.002 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "al meam independent sources 70304 510-520K 30perT float16 20260903 prepared before manifest checksum",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "prepared",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum}"
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
| preparation_seed | [129223029, 139885636, 144370470, 151871197, 160380922, 184384186, 200290231, 234122652, 24993146, 259940230, 269696945, 270739820] … (60 values; see JSON) |
| source_split | ["final_validation", "model_selection", "optimization"] |
| temperature_K | [510.0, 520.0] |
| velocity_seed | [118047358, 140564254, 150812759, 178631405, 185894899, 19173877, 192647554, 201750649, 226218785, 240372582, 257766501, 263207168] … (60 values; see JSON) |
| scientific_contract | [{"screening_futures_reused_for_evaluation": false, "split_unit": "source run and every screened parent and future descended from it", "structural_independence": "Every source has its own 300 ps full-box melt trajectory from a unique preparation seed before its independently seeded undercooling history."}] |
| split_unit | ["source run and every screened parent and future descended from it"] |
| structural_independence | ["Every source has its own 300 ps full-box melt trajectory from a unique preparation seed before its independently seeded undercooling history."] |
| temperatures_K | [[510.0, 520.0]] |

## Evidence

All 62 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum-84872e32.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-17T22:51:50.502756+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
