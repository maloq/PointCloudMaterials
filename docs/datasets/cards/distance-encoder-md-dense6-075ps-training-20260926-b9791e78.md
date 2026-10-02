# Unsubmitted native-only six-frame candidate

[All datasets](../README.md) · [Browsable card](distance-encoder-md-dense6-075ps-training-20260926-b9791e78.html) · [Full metadata](../records/distance-encoder-md-dense6-075ps-training-20260926-b9791e78.json)

Shared float32 geometry bank for six tracked-atom observations, 0.75 ps apart in every source. 4,561,920 training and 190,080 selection windows; no new simulation.

- ID: `distance-encoder-md-dense6-075ps-training-20260926`
- Materials: Al
- Classification: **derived**; role: training_geometry
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/md-dense6-075ps-training-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 10.954 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Unsubmitted native-only six-frame candidate",
  "materials": [
    "Al"
  ],
  "role": "training_geometry",
  "classification": "derived",
  "description": "Shared float32 geometry bank for six tracked-atom observations, 0.75 ps apart in every source. 4,561,920 training and 190,080 selection windows; no new simulation.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Fixed Al64 native source roles: 90 train, 15 selection. No external trajectory ancestry included. Anchor rows match the native subset of the wider-history arm.",
  "evidence": [
    "docs/distance_encoder_history.md"
  ],
  "limitations": [
    "Al only; external sources with 0.10-ps cadence excluded.",
    "The wider-history arm includes external training sources, so comparing it directly cannot isolate sampling cadence.",
    "Superseded by the user-approved all-data nominal .75/.70-ps experiment before training."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["fixed_material_cutoff_al_reference_v1", "six_consecutive_uniform_cadence_md_frames_v2"] |
| cadence_ps | [0.75] |
| material | ["Al"] |
| split | ["selection", "train"] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297", "independent_melt_160380922", "independent_melt_176364202"] … (105 values; see JSON) |
| timestep_fs | [3.0] |
| frame_count | [801] |
| atom_count | [70304] |

## Evidence

All 96 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-md-dense6-075ps-training-20260926-b9791e78.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
