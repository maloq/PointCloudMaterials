# Ni outcome-blind native windows from thirty additional source histories

[All datasets](../README.md) · [Browsable card](ni-native-expansion-20261018-4e340fef.html) · [Full metadata](../records/ni-native-expansion-20261018-4e340fef.json)

7200 fixed-central-atom64-frame measurement windows at exact native0.1ps:15 stratified random anchors after20ps and16 uniform atomIDs peranchor perexisting source. No future-label sampling and no new MD. BO40/TDA113/density74/velocity17 match original244 measurement producer. Exact cached PTM annotations use the original phase/fraction/component definitions. Normalization reuses original16-source training transforms unchanged.

- ID: `ni-native-expansion-20261018`
- Materials: Ni
- Classification: **research**; role: motion_training_and_evaluation
- Location: `/home/ids/vmorozov/training-cache/ni-latent-flow/validation-20261018-v2-expansion`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.785 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni outcome-blind native windows from thirty additional source histories",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "motion_training_and_evaluation",
  "classification": "research",
  "description": "7200 fixed-central-atom64-frame measurement windows at exact native0.1ps:15 stratified random anchors after20ps and16 uniform atomIDs peranchor perexisting source. No future-label sampling and no new MD. BO40/TDA113/density74/velocity17 match original244 measurement producer. Exact cached PTM annotations use the original phase/fraction/component definitions. Normalization reuses original16-source training transforms unchanged.",
  "ancestry": "12additionaltraining/18additionaltest source ancestors; original roles preserved. Matched original-window training uses80windows/source:1280/16 versus2240/28. All4320 additionaltestwindows plus10560existingtestwindows; additionalheldout sources have prior birth-assay exposure.",
  "sample_interval_ps": 0.1,
  "source_plan_sha256": "8ef9abfc99f7a72275b96d0dec702f3a32fa34161c410f9fb45b06ec82514596",
  "evidence": [
    "src/research/ni_latent_flow/expansion.py",
    "${storage:training_storage}/ni_latent_flow/validation-20261018-v2/technical/expansion-plan.json",
    "${storage:training_storage}/ni_latent_flow/validation-20261018-v2/technical/expansion-complete.json",
    "${storage:training_storage}/ni_latent_flow/validation-20261018-v2/technical/data-audit.json"
  ],
  "limitations": [
    "Native simulation stopping rules remain inherited; uniformly sampled saved histories do not establish natural incidence.",
    "Local central-atom persistent PTM entry is distinct from full-cell connected nucleus birth.",
    "Only80of240newtrainingwindows per source enter the declared28-source matched comparison;1920additionaltrainingwindows remain unused.",
    "Old future-enriched native rows remain an explicitly separate training recipe.",
    "No regional87-channel context is added to these new244-measurement windows."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 30 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-native-expansion-20261018-4e340fef.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
