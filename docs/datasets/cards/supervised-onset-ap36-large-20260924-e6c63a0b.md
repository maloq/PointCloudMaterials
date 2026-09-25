# Supervised onset AP3/AP6 paired Al graph release

[All datasets](../README.md) · [Browsable card](supervised-onset-ap36-large-20260924-e6c63a0b.html) · [Full metadata](../records/supervised-onset-ap36-large-20260924-e6c63a0b.json)

31,609 existing prospective observations; original MD onset labels; paired observed/relaxed nearest-80 graphs in physical Angstroms. 90 train, 15 selection, 15 calibration, 30 historical-test independent melt lineages. All 150 original source manifests and ancestry IDs preserved in manifest.json. No new MD or quenches.

- ID: `supervised-onset-ap36-large-20260924`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/supervised-onset/ap36-large-20260924`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.566 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Supervised onset AP3/AP6 paired Al graph release",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "31,609 existing prospective observations; original MD onset labels; paired observed/relaxed nearest-80 graphs in physical Angstroms. 90 train, 15 selection, 15 calibration, 30 historical-test independent melt lineages. All 150 original source manifests and ancestry IDs preserved in manifest.json. No new MD or quenches.",
  "evidence": [
    "${dataset:supervised-onset-ap36-large-20260924}/manifest.json",
    "configs/supervised_onset/ap36_20260924.json",
    "src/research/supervised_onset/data.py"
  ],
  "limitations": [
    "Historical test reused in earlier studies.",
    "Existing nearest-80 support; no computational halo.",
    "Paired held-out anchor cadence 12 ps; event labels retain 0.75 ps resolution.",
    "Inherited float16 trajectory precision; no recovered lost coordinate precision."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cutoff_A | [5.0] |
| radius_A | [8.0] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/supervised-onset-ap36-large-20260924-e6c63a0b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
