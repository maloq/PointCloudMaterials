# Robust-onset finite coordinate perturbation bank

[All datasets](../README.md) · [Browsable card](robust-onset-noise-20260924-c395b8c8.html) · [Full metadata](../records/robust-onset-noise-20260924-c395b8c8.json)

Three deterministic coordinate-noise views of fitting-only paired Al snapshots, using 0.2/0.5/1 percent expected 3D displacement RMS relative to local 12-neighbor spacing. Center fixed, neighbor edges rebuilt. Parent quantized positions and independent source ancestry preserved; no new simulation.

- ID: `robust-onset-noise-20260924`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/robust-onset/noise-20260924`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.120 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Robust-onset finite coordinate perturbation bank",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Three deterministic coordinate-noise views of fitting-only paired Al snapshots, using 0.2/0.5/1 percent expected 3D displacement RMS relative to local 12-neighbor spacing. Center fixed, neighbor edges rebuilt. Parent quantized positions and independent source ancestry preserved; no new simulation.",
  "evidence": [
    "configs/robust_onset/screen_20260924.json",
    "experiments/robust_onset_20260924/README.md",
    "${storage:cache}/robust-onset/noise-20260924/manifest.json"
  ],
  "limitations": [
    "Finite reused augmentation bank, not a new physical trajectory; full finite parent support retained.",
    "Onset labels used for encoder supervision in robust_onset_v1 only. The 15 development roots were used in prior research."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [20260924] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/robust-onset-noise-20260924-c395b8c8.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
