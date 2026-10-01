# Al pre-appearance birth versus liquid classification histories

[All datasets](../README.md) · [Browsable card](al-birth-preappearance-v1-20260930-290730f2.html) · [Full metadata](../records/al-birth-preappearance-v1-20260930-290730f2.json)

Eight exact 0.75 ps PTM-clear observations, matched continuously liquid controls and four truncated endpoints. Original 90/15/15/30 source roles. Pending full preparation; no new trajectories.

- ID: `al-birth-preappearance-v1-20260930`
- Materials: Al
- Classification: **derived**; role: event_enriched_classification
- Location: `/home/ids/vmorozov/training-cache/birth-prediction/preappearance-20260930`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.056 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al pre-appearance birth versus liquid classification histories",
  "materials": [
    "Al"
  ],
  "role": "event_enriched_classification",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Eight exact 0.75 ps PTM-clear observations, matched continuously liquid controls and four truncated endpoints. Original 90/15/15/30 source roles. Pending full preparation; no new trajectories.",
  "provenance": "src/research/birth_prediction/data.py; completed original full-cell PTM and ancestry audit",
  "lineage": "Inherits original fixed Al64 source/melt ancestors; event anchors and windows are correlated.",
  "evidence": [
    "configs/birth_prediction/preappearance_20260930.json",
    "docs/metrics/birth_prediction.md",
    "${dataset:al-birth-preappearance-v1-20260930}/plan.json",
    "${dataset:al-birth-preappearance-v1-20260930}/manifest.json"
  ],
  "limitations": [
    "Operational isolated establishments, not physical critical nuclei.",
    "Retrospective event-enriched case/control classification is not a prospective natural-risk population.",
    "Bounded-window appearance; full 8 A sphere PTM-clear histories, nearest-80 actual inputs.",
    "Minimum event-support gate can stop fitting; no model-specific row exclusions."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al64-crystallization-origin-audit-v2", "preappearance_birth_classification_v1"] |
| seed | [20260930] |
| cadence_ps | [0.75] |
| radius_A | [8.0] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (151 values; see JSON) |
| horizons_ps | [[3.0, 6.0]] |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 152 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-birth-preappearance-v1-20260930-290730f2.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
