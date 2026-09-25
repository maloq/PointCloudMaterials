# Al/Mg/Ta/Ti · mixed relaxed-MACE training views

[All datasets](../README.md) · [Browsable card](mace-full-081c8f33.html) · [Full metadata](../records/mace-full-081c8f33.json)

Mixed-material derived MACE inputs/targets; original thermal Al/Mg/Ta data plus Ti/Ta branch additions. Source paths and transformations are recorded in manifest.json.

- ID: `mace-full`
- Materials: Al, Mg, Ta, Ti
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/mace-full`
- Present on this machine: True
- Potentials: Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.259 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al/Mg/Ta/Ti \u00b7 mixed relaxed-MACE training views",
  "materials": [
    "Al",
    "Mg",
    "Ta",
    "Ti"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Mixed-material derived MACE inputs/targets; original thermal Al/Mg/Ta data plus Ti/Ta branch additions. Source paths and transformations are recorded in manifest.json.",
  "evidence": [
    "${dataset:mace-full}/manifest.json"
  ],
  "potential_ids": [
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ta-zhong2014-eam",
    "ti-kavousi2019-meam"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [20260910, 42] |
| materials | [["Al", "Mg", "Ta", "Ti"]] |
| material | ["Al", "Mg", "Ta", "Ti"] |
| storage_dtype | ["float16", "float32"] |
| radius | [10.169428, 9.192189, 9.247764279105562, 9.388275] |
| frame_count | [2401, 241] |
| split | ["train", "val"] |
| periodic | [true] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/mace-full-081c8f33.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
