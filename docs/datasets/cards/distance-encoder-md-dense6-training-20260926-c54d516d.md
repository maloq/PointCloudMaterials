# Withdrawn mixed-cadence six-frame geometry preparation

[All datasets](../README.md) · [Browsable card](distance-encoder-md-dense6-training-20260926-c54d516d.html) · [Full metadata](../records/distance-encoder-md-dense6-training-20260926-c54d516d.json)

Withdrawn before any GPU training after the user required uniform physical observation spacing. Preparation also encountered IDS quota. Frozen preparation receipts are preserved; disposable geometry may be reused or removed.

- ID: `distance-encoder-md-dense6-training-20260926`
- Materials: Al, Mg, Ta, Ti
- Classification: **derived**; role: training_geometry
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/md-dense6-training-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 11.575 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Withdrawn mixed-cadence six-frame geometry preparation",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta"
  ],
  "role": "training_geometry",
  "classification": "derived",
  "description": "Withdrawn before any GPU training after the user required uniform physical observation spacing. Preparation also encountered IDS quota. Frozen preparation receipts are preserved; disposable geometry may be reused or removed.",
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ti-kavousi2019-meam",
    "ta-zhong2014-eam"
  ],
  "lineage": "Inherited fixed Al64 source roles; external branches train-only with their original shared ancestry. No new simulation.",
  "evidence": [
    "${storage:analysis}/distance_encoder/md-dense6-20260926/technical/code/config.json",
    "docs/distance_encoder_history.md"
  ],
  "limitations": [
    "Cadence is .75 ps on native Al, .10 ps on external training sources; no cadence value is a model input.",
    "Held-out performance is evaluated on Al only."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["fixed_material_cutoff_al_reference_v1", "six_consecutive_md_frames_v1"] |
| material | ["Al", "Mg", "Ta", "Ti"] |
| split | ["selection", "train"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "al-1m-independent-melt-911001", "independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076"] … (110 values; see JSON) |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [2401, 241, 3001, 4001, 7241, 801] |
| atom_count | [100000, 1000000, 10000422, 1024000, 1048576, 70304] |
| cadence_ps | [0.09999999999999432, 0.09999999999999964, 0.10000000000000853, 0.75] |

## Evidence

All 96 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-md-dense6-training-20260926-c54d516d.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
