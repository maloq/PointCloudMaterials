# RD-MACE256-L3-Z256: raw rich-descriptor learning, 60 full epochs

[All datasets](../README.md) · [Browsable card](rd-mace256-rich-descriptors-20260929-cc418a6c.html) · [Full metadata](../records/rd-mace256-rich-descriptors-20260929-cc418a6c.json)

Geometry-only MACE256, three interactions and 256-D patch/context states; same 3536 rich targets. All 183596 raw train contexts, immutable source splits. Large measured batch; 0.004 five-epoch warmup/cosine LR. No relaxation or physical distance supervision.

- ID: `rd-mace256-rich-descriptors-20260929`
- Materials: Al
- Classification: **research**; role: trained_encoder
- Location: `/work/PERSO/vmorozov/analysis/liquid_predictability/rd-mace256-l3-z256-20260929`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.551 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "RD-MACE256-L3-Z256: raw rich-descriptor learning, 60 full epochs",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "trained_encoder",
  "classification": "research",
  "description": "Geometry-only MACE256, three interactions and 256-D patch/context states; same 3536 rich targets. All 183596 raw train contexts, immutable source splits. Large measured batch; 0.004 five-epoch warmup/cosine LR. No relaxation or physical distance supervision.",
  "provenance": "src/research/liquid_predictability/rich_encoder.py; sealed existing raw descriptor/geometry caches reused read-only.",
  "ancestry": "Existing fixed-Al64 melt-source roles; 88 eligible train sources, 15 selection, 15 calibration, 28 test. Added atoms are correlated within sources.",
  "evidence": [
    "configs/liquid_predictability/rich_mace256_20260929.json",
    "docs/rich_descriptor_encoder.md",
    "${storage:analysis}/liquid_predictability/rd-mace256-l3-z256-20260929/technical/launch.json"
  ],
  "limitations": [
    "Combined width/depth/state-size/batch/LR/duration intervention does not isolate their effects.",
    "One training seed; prior test-source reuse.",
    "Descriptor reconstruction is not itself evidence of crystal-distance information."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| radius | [8.0] |
| protocol | ["rich_descriptor_mace_full_epochs_v1"] |
| seed | [20260929] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/rd-mace256-rich-descriptors-20260929-cc418a6c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
