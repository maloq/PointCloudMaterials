# Fixed Al64 dense observed embedding evaluation

[All datasets](../README.md) · [Browsable card](fixed-al64-dense-observed-20260925-5b7a6043.html) · [Full metadata](../records/fixed-al64-dense-observed-20260925-5b7a6043.json)

All 801 stored frames of 64 fixed centers in the 30 test sources; 0.75 ps cadence. Observed nearest80 geometry and original structural labels. Evaluation only; never training or normalization.

- ID: `fixed-al64-dense-observed-20260925`
- Materials: Al
- Classification: **derived**; role: evaluation_cache
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/training-cache/encoder-context/al64-epochs-20260925/dense-observed`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.972 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Fixed Al64 dense observed embedding evaluation",
  "materials": [
    "Al"
  ],
  "role": "evaluation_cache",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "All 801 stored frames of 64 fixed centers in the 30 test sources; 0.75 ps cadence. Observed nearest80 geometry and original structural labels. Evaluation only; never training or normalization.",
  "lineage": "Same immutable independent Al source ancestry and test roles as fixed-al64-benchmark-v1-20260925. No new simulation or quench.",
  "evidence": [
    "${dataset:fixed-al64-dense-observed-20260925}/manifest.json",
    "configs/fixed_cohort/al64_v1.json",
    "src/research/encoder_context/dense.py"
  ],
  "limitations": [
    "Historically examined test sources; not an untouched evaluation.",
    "Observed geometry only; dense relaxed trajectories unavailable.",
    "Nearest80 cropped at 8 Angstrom, without computational halo."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cadence_ps | [0.75] |

## Evidence

All 31 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/fixed-al64-dense-observed-20260925-5b7a6043.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
