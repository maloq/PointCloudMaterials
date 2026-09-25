# Fixed Al64 crystallization benchmark

[All datasets](../README.md) · [Browsable card](fixed-al64-benchmark-v1-20260925-4fb74686.html) · [Full metadata](../records/fixed-al64-benchmark-v1-20260925-4fb74686.json)

150 independent native Al sources, frozen 90 train/15 selection/15 calibration/30 test roles and 64 tracked centers. Matched observed/relaxed nearest80 patches, original MD sustained-onset labels at 0.75/3/6/9/12 ps, and exact historical 16-center comparison rows.

- ID: `fixed-al64-benchmark-v1-20260925`
- Materials: Al
- Classification: **derived**; role: prediction_benchmark
- Location: `/home/ids/vmorozov/training-cache/fixed-cohort/al64-v1-20260925/benchmark`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.583 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Fixed Al64 crystallization benchmark",
  "materials": [
    "Al"
  ],
  "role": "prediction_benchmark",
  "classification": "derived",
  "description": "150 independent native Al sources, frozen 90 train/15 selection/15 calibration/30 test roles and 64 tracked centers. Matched observed/relaxed nearest80 patches, original MD sustained-onset labels at 0.75/3/6/9/12 ps, and exact historical 16-center comparison rows.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Inherited independent melt ancestry and fixed source roles; exact sources, manifest hashes and atom IDs in ../splits.json. No new simulation.",
  "evidence": [
    "${dataset:fixed-al64-benchmark-v1-20260925}/manifest.json",
    "${dataset:fixed-al64-benchmark-v1-20260925}/../plan.json",
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
| horizons_ps | [[0.75, 3.0, 6.0, 9.0, 12.0]] |

## Evidence

All 151 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/fixed-al64-benchmark-v1-20260925-4fb74686.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
