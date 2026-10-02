# Confirmed crystal distances for dynamic multi-material MACE training

[All datasets](../README.md) · [Browsable card](distance-encoder-multimaterial-20260926-d7e3e963.html) · [Full metadata](../records/distance-encoder-multimaterial-20260926-d7e3e963.json)

12,747,736 training and 192,000 Al selection neighborhoods; existing dynamic structural centers at multiple timesteps, excluding initial unconfirmed history. Labels only; coordinates reused from the sealed structural release.

- ID: `distance-encoder-multimaterial-20260926`
- Materials: Al, Mg, Ta, Ti, Zr
- Classification: **derived**; role: supervised_training_labels
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/multimaterial-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.098 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Confirmed crystal distances for dynamic multi-material MACE training",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta"
  ],
  "role": "supervised_training_labels",
  "classification": "derived",
  "description": "12,747,736 training and 192,000 Al selection neighborhoods; existing dynamic structural centers at multiple timesteps, excluding initial unconfirmed history. Labels only; coordinates reused from the sealed structural release.",
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ta-zhong2014-eam",
    "ti-kavousi2019-meam"
  ],
  "lineage": "Frozen native Al 90/15 train/selection ancestry; external branches train-only with their recorded shared preparation groups. No new simulation.",
  "evidence": [
    "configs/distance_encoder/multimaterial_early_20260926.json",
    "docs/distance_encoder.md",
    "${dataset:distance-encoder-multimaterial-20260926}/manifest.json"
  ],
  "limitations": [
    "External branches are not independent replicates; no external-material validation/test claim.",
    "Distances and inputs use fixed Al-reference length normalization; no material/temperature/time input.",
    "Targets refer to already confirmed crystal components, not future nucleation."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["causal-distance-multimaterial-v1", "fixed_material_cutoff_al_reference_v1"] |
| material | ["Al", "Mg", "Ta", "Ti", "Zr"] |
| split | ["selection", "train"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "al-1m-independent-melt-911001", "independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076"] … (114 values; see JSON) |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [1401, 1449, 2401, 241, 3001, 4001, 481, 49, 7241, 801] |
| atom_count | [100000, 1000000, 10000422, 1024000, 1048576, 70304] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| cadence_ps | [0.5] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-multimaterial-20260926-d7e3e963.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
