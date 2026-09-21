# BCR independent-root 1325K Al G1 pilot

[All datasets](../README.md) · [Browsable card](bcr-g1-independent-20260921-471dd663.html) · [Full metadata](../records/bcr-g1-independent-20260921-471dd663.json)

Complete-radius8 A patches from 18 independent 300ps melt endpoints at1325K, one snapshot/root.12 train/6 development;3072/1536 spatially thinned patches. Original nine-significant-digit validation snapshots, not float16 recovery.

- ID: `bcr-g1-independent-20260921`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/bcr/g1-independent-20260921`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.008 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "BCR independent-root 1325K Al G1 pilot",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Complete-radius8 A patches from 18 independent 300ps melt endpoints at1325K, one snapshot/root.12 train/6 development;3072/1536 spatially thinned patches. Original nine-significant-digit validation snapshots, not float16 recovery.",
  "evidence": [
    "configs/bcr/pilot_20260921/study.json",
    "output/bcr/g1-independent-20260921/technical/inventory.json"
  ],
  "limitations": [
    "Historical training roots; no fresh final test. No undercooled trajectory or crystallization claims. Overlapping neighborhoods; bootstrap roots only."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [20260921] |
| radius_A | [8.0] |
| split | ["development", "train"] |
| temperature_K | [1325.0] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/bcr-g1-independent-20260921-471dd663.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
