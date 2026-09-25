# Fixed Al64 structural pretraining: 1,157,760 training neighborhoods

[All datasets](../README.md) · [Browsable card](fixed-al64-structural-v1-20260925-a7f0b736.html) · [Full metadata](../records/fixed-al64-structural-v1-20260925-a7f0b736.json)

Native Al geometry only: 90 training sources, 64 fixed centers, 201 frames at 3 ps sampling. Separate 15-source validation set of 192,960 neighborhoods. Optional 86,400 train/36,480 validation observed-relaxed pairs. No calibration/test ancestors or descendants, no crystallization labels, no outcome filtering.

- ID: `fixed-al64-structural-v1-20260925`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/fixed-cohort/al64-v1-20260925/structural`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.257 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Fixed Al64 structural pretraining: 1,157,760 training neighborhoods",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Native Al geometry only: 90 training sources, 64 fixed centers, 201 frames at 3 ps sampling. Separate 15-source validation set of 192,960 neighborhoods. Optional 86,400 train/36,480 validation observed-relaxed pairs. No calibration/test ancestors or descendants, no crystallization labels, no outcome filtering.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Inherited independent melt ancestry and fixed source roles; exact sources, manifest hashes and atom IDs in ../splits.json. No new simulation.",
  "evidence": [
    "${dataset:fixed-al64-structural-v1-20260925}/manifest.json",
    "${dataset:fixed-al64-structural-v1-20260925}/../plan.json",
    "configs/fixed_cohort/al64_v1.json",
    "docs/datasets/fixed_al64.md"
  ],
  "limitations": [
    "Historical test sources are reused, not a fresh untouched test.",
    "Centers and windows within a source are correlated.",
    "Nearest80 cropped at 8 Angstrom, not a complete radius ball or computational halo.",
    "Structural selection is validation only; structural fit and normalization use only the 90 training ancestors."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297", "independent_melt_160380922", "independent_melt_176364202"] … (105 values; see JSON) |

## Evidence

All 106 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/fixed-al64-structural-v1-20260925-a7f0b736.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
