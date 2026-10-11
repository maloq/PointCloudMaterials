# Al nuclei birth dataset: native 0.30 ps

[All datasets](../README.md) · [Browsable card](al-nuclei-birth-030-2e3186d0.html) · [Full metadata](../records/al-nuclei-birth-030-2e3186d0.json)

Only uniformly saved native 0.30 ps trajectories. Excludes subsampled 0.10/0.01/0.075/0.15 ps data and nested shooting histories with finer or irregular raw prefixes. Source timelines and every raw timestep array are checked; upstream excluded data and search results remain preserved. Update through src.data.nuclei_birth refresh.

- ID: `al-nuclei-birth-030`
- Materials: Al
- Classification: **derived**; role: event_dataset
- Location: `/store/PERSO/vmorozov/experiments/nuclei_birth/al/cadences/030`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "Al nuclei birth dataset: native 0.30 ps",
  "materials": [
    "Al"
  ],
  "role": "event_dataset",
  "classification": "derived",
  "potential_ids": [],
  "description": "Only uniformly saved native 0.30 ps trajectories. Excludes subsampled 0.10/0.01/0.075/0.15 ps data and nested shooting histories with finer or irregular raw prefixes. Source timelines and every raw timestep array are checked; upstream excluded data and search results remain preserved. Update through src.data.nuclei_birth refresh.",
  "provenance": "${dataset:al-nuclei-birth}/manifest.json",
  "lineage": "Cadence views share source ancestry; original source roles unchanged.",
  "evidence": [
    "docs/datasets/nuclei_birth.md",
    "${dataset:al-nuclei-birth-030}/manifest.json"
  ],
  "sampling_policy": "native_saved",
  "limitations": [
    "Different source population from the 0.10 ps dataset; raw totals are not a matched cadence comparison.",
    "No 0.30 ps precursor-tail supplement is currently published."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cadence_ps | [0.3] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nuclei-birth-030-2e3186d0.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
