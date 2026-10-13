# Present-selected Al embryo establishment at 1/3/6 ps

[All datasets](../README.md) · [Browsable card](al-embryo-establishment-136-20261007-5c04105b.html) · [Full metadata](../records/al-embryo-establishment-136-20261007-5c04105b.json)

Separate current 8–63 atom embryo cohort at fixed 1 ps landmarks, with common 1/3/6 ps root-establishment labels. All 92 pinned saved-velocity views; four historical views retain unknown potentials. The October 7 search of 22 additional sources is pending and requires a later release.

- ID: `al-embryo-establishment-136-20261007`
- Materials: Al
- Classification: **derived**; role: frozen_encoder_transfer_inputs
- Location: `/home/ids/vmorozov/training-cache/continuous-motion/embryo-establishment-20261007-v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.117 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Present-selected Al embryo establishment at 1/3/6 ps",
  "materials": [
    "Al"
  ],
  "role": "frozen_encoder_transfer_inputs",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Separate current 8\u201363 atom embryo cohort at fixed 1 ps landmarks, with common 1/3/6 ps root-establishment labels. All 92 pinned saved-velocity views; four historical views retain unknown potentials. The October 7 search of 22 additional sources is pending and requires a later release.",
  "provenance": "src.research.continuous_motion.embryo_cohort; checksum-bound v3 PTM families and archived same-frame nearest80 Q/V. Native velocity units A/ps; no new MD, relaxation or interpolation.",
  "lineage": "Original fixed Al64 melt ancestry and roles retained. Exact full-state family and observation aliases deduplicate archived copies. Repeated landmarks and embryo families are not independent melt ancestors.",
  "evidence": [
    "configs/analysis/embryo_establishment_20261007.json",
    "docs/continuous_motion.md",
    "docs/metrics/embryo_establishment_cohort.md",
    "${storage:training_storage}/continuous_motion/embryo-establishment-20261007-v1/technical/plan.json",
    "${storage:training_storage}/continuous_motion/embryo-establishment-20261007-v1/technical/launch.json"
  ],
  "limitations": [
    "Completed cohort; 21,651 training and 12,409 pooled-test observations. Pooled test has 1,776 embryo families across 29 ancestors; positive families at 1/3/6 ps are 29/54/67. Repeated landmarks are not independent.",
    "Conditional embryo-establishment assay, not strict pre-birth prediction or unconditional population nucleation risk.",
    "Velocity-free million-atom and other-cadence/relaxed inputs excluded from this matched Q/V assay.",
    "No new predictive fit in collection; later readouts must use train-only fitting/selection and pooled held-out evaluation."
  ],
  "source_records": 92,
  "observations": 34060,
  "physical_families": 4788,
  "melt_ancestors": 84
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 93 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-embryo-establishment-136-20261007-5c04105b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
