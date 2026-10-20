# Al failed-embryo appearance histories (relaxed)

[All datasets](../README.md) · [Browsable card](al-birth-failed-appearance-relaxed-20261002-6356448b.html) · [Full metadata](../records/al-birth-failed-appearance-relaxed-20261002-6356448b.json)

Strict crystal-free histories for eligible members of the nested 184/28 failed-embryo catalogue, matched liquid controls, original source roles. Failed appearances are positive. Full-cell MEAM minimization of observed inputs.

- ID: `al-birth-failed-appearance-relaxed-20261002`
- Materials: Al
- Classification: **derived**; role: event_enriched_classification
- Location: `/home/ids/vmorozov/training-cache/birth-prediction/appearance-transients-relaxed-20261002`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.115 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al failed-embryo appearance histories (relaxed)",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "event_enriched_classification",
  "classification": "derived",
  "description": "Strict crystal-free histories for eligible members of the nested 184/28 failed-embryo catalogue, matched liquid controls, original source roles. Failed appearances are positive. Full-cell MEAM minimization of observed inputs.",
  "provenance": "src/research/birth_prediction/appearance_data.py; completed Al64 fate audit; no new MD",
  "lineage": "Same fixed Al64 source/melt ancestors; events and centers within a source are correlated. No new independent trajectories.",
  "evidence": [
    "configs/birth_prediction/appearance_transfer_20261002.json",
    "docs/metrics/birth_appearance.md",
    "${dataset:al-birth-failed-appearance-relaxed-20261002}/manifest.json"
  ],
  "limitations": [
    "Retrospective selection, operational PTM embryos, 0.75 ps cadence.",
    "Candidate counts decrease after strict history eligibility screening.",
    "Controls are continuously liquid within a declared finite window; class prevalence is not natural incidence."
  ],
  "relaxation_source_availability_note": "Some historical relaxed-cell/neighborhood inputs were deleted at user request on 2026-10-03; retained downstream artifacts are unchanged. See docs/storage/relaxation_unverified_removal_20261003/receipt.json."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["paired_full_cell_relaxed_birth_temporal_v1"] |
| seed | [20260930] |
| timestep_ps | [0.001] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 1360 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-birth-failed-appearance-relaxed-20261002-6356448b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
