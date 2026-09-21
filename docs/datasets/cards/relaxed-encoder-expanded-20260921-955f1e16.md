# Expanded paired observed/relaxed Al training_cache

[All datasets](../README.md) · [Browsable card](relaxed-encoder-expanded-20260921-955f1e16.html) · [Full metadata](../records/relaxed-encoder-expanded-20260921-955f1e16.json)

90 training and 15 development sources; four encoder-training origins; 15 regular-grid assay origins across 150 sources. Full periodic fixed-box Lee2003 MEAM FIRE quenches at 0.01 eV/Angstrom. Tracked nearest80 candidates with radius8 support. Reuses verified pilot and older target clouds where query identities match.

- ID: `relaxed-encoder-expanded-20260921`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/relaxed_encoder/expanded-20260921`
- Present on this machine: False
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Expanded paired observed/relaxed Al training_cache",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "90 training and 15 development sources; four encoder-training origins; 15 regular-grid assay origins across 150 sources. Full periodic fixed-box Lee2003 MEAM FIRE quenches at 0.01 eV/Angstrom. Tracked nearest80 candidates with radius8 support. Reuses verified pilot and older target clouds where query identities match.",
  "evidence": [
    "configs/analysis/relaxed_encoder_expanded.json",
    "output/relaxed_encoder/expanded-20260921/technical/plan.json"
  ],
  "limitations": [
    "Preparing; consult per-cell completion receipts. One seed. Reused historical source split, no test-source encoder fitting. Full-cell quenching includes external context. Raw MD coordinates inherit float16 storage; relaxed local clouds saved float32 before full-cell conversion. Training-only target-domain normalization."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/relaxed-encoder-expanded-20260921-955f1e16.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T09:05:46.021728+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
