# Ta20A event-centered appearance and dissolution movies

[All datasets](../README.md) · [Browsable card](ta-nucleus-crops-20261003-4954a75c.html) · [Full metadata](../records/ta-nucleus-crops-20261003-4954a75c.json)

Separate event crops at native0.1ps, requested18ps before/after, truncated to available24ps branches. Eight ambiguous appearances and28 verified dissolutions.

- ID: `ta-nucleus-crops-20261003`
- Materials: Ta
- Classification: **derived**; role: visual_review
- Location: `/work/PERSO/vmorozov/analysis/nucleus_crops/ta-20A-18ps-20261003`
- Present on this machine: True
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.275 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ta20A event-centered appearance and dissolution movies",
  "materials": [
    "Ta"
  ],
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "role": "visual_review",
  "classification": "derived",
  "description": "Separate event crops at native0.1ps, requested18ps before/after, truncated to available24ps branches. Eight ambiguous appearances and28 verified dissolutions.",
  "provenance": "src/research/crystallization_origin/event_crops.py; frozen completed Ta audit; original full-cell coordinates and partial PTM labels.",
  "ancestry": "Preserves original parent and common preparation ancestry; future-selected event centers; not a causal prediction input release.",
  "evidence": [
    "configs/analysis/ta_nucleus_crops_20261003.json",
    "docs/ta_nucleus_crops.md",
    "${dataset:ta-nucleus-crops-20261003}/technical/plan.json"
  ],
  "limitations": [
    "All requested36ps windows are truncated because sources last24ps.",
    "Event labels remain defined on0.5ps grid; intermediate0.1ps PTM labels unavailable.",
    "Archived-parent generating potential unknown; shooting descendants use Zhong2014 EAM."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| material | ["Ta"] |
| parent_id | ["2.7ns", "2.8ns", "2.9ns", "3.0ns", "3.60ns", "model_1m"] |
| velocity_seed | [145515637, 186999831, 241723527, 254269500, 376325111, 381745835, 395820186, 436100696, 47085415, 535089732, 549417302, 779428654] … (16 values; see JSON) |
| timestep_fs | [2.0] |
| storage_dtype | ["float16"] |
| atom_count | [10000422, 1024000] |
| frame_count | [49] |
| cadence_ps | [0.1, 0.5] |
| lineage | ["archived-Ta-unknown-common-preparation", {"interface_distance_A": 8.233252184899222, "local_birth_radius_A": 8.233252184899222, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.70496348320465, "thresholds": [{"name": "primary", "persistence_frames": 4, "persistence_ps": 1.5, "size": 64}]}] |
| radius_A | [20.0] |
| protocol | ["ta_event_review_crops_v1"] |

## Evidence

All 75 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-nucleus-crops-20261003-4954a75c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
