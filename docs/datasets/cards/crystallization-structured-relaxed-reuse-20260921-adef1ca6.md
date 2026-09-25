# Reused relaxed MACE symmetric-context features

[All datasets](../README.md) · [Browsable card](crystallization-structured-relaxed-reuse-20260921-adef1ca6.html) · [Full metadata](../records/crystallization-structured-relaxed-reuse-20260921-adef1ca6.json)

Frozen inventory of3062 existing quenched cells,150 independent Al sources. Original source split. Three real observations within72 ps. Tracked observed cuboctahedral query identities, actual relaxed geometry. Cold-vic-temp01 frozen encoder. Shared original MD future targets; no new simulation.

- ID: `crystallization-structured-relaxed-reuse-20260921`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystallization-structured-relaxed-reuse-20260921`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.255 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Reused relaxed MACE symmetric-context features",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Frozen inventory of3062 existing quenched cells,150 independent Al sources. Original source split. Three real observations within72 ps. Tracked observed cuboctahedral query identities, actual relaxed geometry. Cold-vic-temp01 frozen encoder. Shared original MD future targets; no new simulation.",
  "evidence": [
    "configs/crystallization_transfer/symmetric_relaxed_reuse_20260921.json",
    "output/crystallization_transfer/symmetric-relaxed-reuse-20260921/technical/plan.json",
    "output/crystallization_transfer/symmetric-relaxed-reuse-20260921/technical/precision.json"
  ],
  "limitations": [
    "Preparing features; inspect source receipts. Archived full-cell float16 precision; four-cell feature comparison median cosine0.999105, relative RMS0.063568. Not equivalent to precise local-cloud extraction. Irregular frozen origin availability; compare matched controls."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 150 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystallization-structured-relaxed-reuse-20260921-adef1ca6.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
