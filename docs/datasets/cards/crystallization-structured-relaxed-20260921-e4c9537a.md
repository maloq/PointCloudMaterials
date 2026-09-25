# Relaxed MACE symmetric-context observations

[All datasets](../README.md) · [Browsable card](crystallization-structured-relaxed-20260921-e4c9537a.html) · [Full metadata](../records/crystallization-structured-relaxed-20260921-e4c9537a.json)

Planned 150 independent Al sources; unchanged source ancestry and split. Full-cell fixed-box FIRE relaxation, fmax <=0.01 eV/A. 25 cuboctahedral queries per center; observed nearest80 identities retained in relaxed float32 clouds. Frozen cold-vic-temp01 features. Original MD defines outcomes.

- ID: `crystallization-structured-relaxed-20260921`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystallization-structured-relaxed-20260921`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.001 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Relaxed MACE symmetric-context observations",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Planned 150 independent Al sources; unchanged source ancestry and split. Full-cell fixed-box FIRE relaxation, fmax <=0.01 eV/A. 25 cuboctahedral queries per center; observed nearest80 identities retained in relaxed float32 clouds. Frozen cold-vic-temp01 features. Original MD defines outcomes.",
  "evidence": [
    "configs/crystallization_transfer/symmetric_relaxed_mace_20260921.json",
    "output/crystallization_transfer/symmetric-relaxed-mace-20260921/technical/plan.json"
  ],
  "limitations": [
    "Preparation incomplete; inspect cell receipts. No new MD integration; initial positions inherit float16 storage. One selected encoder and one forecast seed."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation."] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystallization-structured-relaxed-20260921-e4c9537a.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
