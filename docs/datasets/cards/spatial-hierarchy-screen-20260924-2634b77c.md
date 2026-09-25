# Current-frame 16-A surroundings of fixed 8-A focal patches

[All datasets](../README.md) · [Browsable card](spatial-hierarchy-screen-20260924-2634b77c.html) · [Full metadata](../records/spatial-hierarchy-screen-20260924-2634b77c.json)

2880 current-frame observed patches from exactly the 45-root paired geometry screen, expanded from 8 to 16 A. Preserves focal inputs, identities, frame times and root splits; fixed smooth nested-region summaries and dense trajectory inference inputs. No new simulation or relaxation.

- ID: `spatial-hierarchy-screen-20260924`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/spatial-hierarchy/screen-20260924`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.308 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Current-frame 16-A surroundings of fixed 8-A focal patches",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "2880 current-frame observed patches from exactly the 45-root paired geometry screen, expanded from 8 to 16 A. Preserves focal inputs, identities, frame times and root splits; fixed smooth nested-region summaries and dense trajectory inference inputs. No new simulation or relaxation.",
  "evidence": [
    "configs/spatial_hierarchy/screen_20260924.json",
    "experiments/spatial_hierarchy_20260924/README.md",
    "${storage:cache}/spatial-hierarchy/screen-20260924/manifest.json"
  ],
  "limitations": [
    "Fixed invariant coarse context summaries lose outer atom-level detail.",
    "Small reused development cohort; one seed and only 18 positive onset windows.",
    "Original quantized observed trajectories; dense chart is descriptive and may overlap fitting roots."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| radius_A | [16.0] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/spatial-hierarchy-screen-20260924-2634b77c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
