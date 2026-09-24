# Al: current full-timeline native release

[All datasets](../README.md) · [Browsable card](local-predictability-native-release-1a28e8bc.html) · [Full metadata](../records/local-predictability-native-release-1a28e8bc.json)

Current 150-source cohort, tracked centers, split roles and native/dense anchor grids. Links to original trajectories and derived physical targets.

- ID: `local-predictability-native-release`
- Materials: Al
- Classification: **research**; role: dataset_release
- Location: `/home/infres/vmorozov/PointCloudMaterials/output/local_predictability/h100-20260917/technical`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.362 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al: current full-timeline native release",
  "materials": [
    "Al"
  ],
  "role": "dataset_release",
  "classification": "research",
  "description": "Current 150-source cohort, tracked centers, split roles and native/dense anchor grids. Links to original trajectories and derived physical targets.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "docs/data_usage/gatr_20260917.md"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["local_crystallization_predictability_v1"] |
| seed | [20260919] |
| split | ["test", "train", "val"] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| cadence_ps | [0.75] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/local-predictability-native-release-1a28e8bc.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-24T10:36:51.012083+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
