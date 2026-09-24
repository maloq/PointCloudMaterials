# Observed Al bond-coherent cluster and front descriptors

[All datasets](../README.md) · [Browsable card](crystallization-observed-front-20260922-d5a2de0f.html) · [Full metadata](../records/crystallization-observed-front-20260922-d5a2de0f.json)

Existing float16 original MD coordinates,150 independent sources; fixed archived origins and3 ps past. q6 coherence threshold0.7,12 nearest bonds within5 A,ordered if>=7 coherent bonds. Components restricted to12/20/25 A balls;10 A order-evaluation halo.14 invariant descriptors per radius,16 fixed tracked centers/source. No PTM, future labels, new MD or relaxation.

- ID: `crystallization-observed-front-20260922`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystallization-observed-front-20260922`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.034 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Observed Al bond-coherent cluster and front descriptors",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Existing float16 original MD coordinates,150 independent sources; fixed archived origins and3 ps past. q6 coherence threshold0.7,12 nearest bonds within5 A,ordered if>=7 coherent bonds. Components restricted to12/20/25 A balls;10 A order-evaluation halo.14 invariant descriptors per radius,16 fixed tracked centers/source. No PTM, future labels, new MD or relaxation.",
  "evidence": [
    "configs/crystallization_transfer/crystal_front_20260922.json",
    "src/research/crystallization_followup/front.py",
    "output/crystallization_transfer/literature-followup-20260922/technical/plan.json"
  ],
  "limitations": [
    "Preparation in progress; verify per-source receipts. Hard bond-order thresholds are feature definitions, not PTM labels or direct interface surfaces. Components truncated at observation boundary; no global cluster-size information. Single historical source split."
  ],
  "ancestry": "Same150 independent melt lineages and90/15/15/30 source split as archived relaxed-input comparison; identity and original manifest hashes preserved per source."
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 150 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystallization-observed-front-20260922-d5a2de0f.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-24T10:36:51.012083+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
