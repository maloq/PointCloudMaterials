# Ta appearance candidates:20A radius and requested12ps before/after

[All datasets](../README.md) · [Browsable card](ta-appearance-crops-12ps-20261003-408c5099.html) · [Full metadata](../records/ta-appearance-crops-12ps-20261003-408c5099.json)

Eight separate ambiguous appearance crops at0.1ps spacing; requested±12ps around establishment, truncated to available observations.

- ID: `ta-appearance-crops-12ps-20261003`
- Materials: Ta
- Classification: **derived**; role: visual_review
- Location: `/work/PERSO/vmorozov/analysis/nucleus_crops/ta-appearance-20A-12ps-20261003`
- Present on this machine: True
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.090 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ta appearance candidates:20A radius and requested12ps before/after",
  "materials": [
    "Ta"
  ],
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "role": "visual_review",
  "classification": "derived",
  "description": "Eight separate ambiguous appearance crops at0.1ps spacing; requested\u00b112ps around establishment, truncated to available observations.",
  "provenance": "Checksum-verified temporal subsets of completed ta-nucleus-crops-20261003; src/research/crystallization_origin/event_crop_windows.py",
  "ancestry": "Retains parent hashes and shared archived-Ta preparation; future-selected centers, no predictor split.",
  "evidence": [
    "configs/analysis/ta_appearance_crops_12ps_20261003.json",
    "docs/metrics/ta_event_crop_windows.md",
    "${dataset:ta-appearance-crops-12ps-20261003}/technical/complete.json"
  ],
  "limitations": [
    "Ambiguous weak-link-dependent establishments, not confirmed independent births.",
    "None has full\u00b112ps coverage; missing observations are explicit.",
    "Existing0.5ps event labels are not refined; intermediate0.1ps PTM is unavailable."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| material | ["Ta"] |
| parent_id | ["2.7ns", "2.8ns", "2.9ns", "3.0ns"] |
| velocity_seed | [145515637, 395820186, 47085415, 779428654, 796497251, 852288974, 869939435] |
| timestep_fs | [2.0] |
| storage_dtype | ["float16"] |
| atom_count | [10000422] |
| frame_count | [49] |
| cadence_ps | [0.5] |
| lineage | [{"interface_distance_A": 8.233252184899222, "local_birth_radius_A": 8.233252184899222, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.70496348320465, "thresholds": [{"name": "primary", "persistence_frames": 4, "persistence_ps": 1.5, "size": 64}]}] |
| radius_A | [20.0] |
| protocol | ["ta_appearance_review_windows_v1"] |

## Evidence

All 18 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-appearance-crops-12ps-20261003-408c5099.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
