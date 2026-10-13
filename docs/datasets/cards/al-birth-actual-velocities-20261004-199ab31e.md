# Actual velocities for frozen Al nucleus preappearance neighborhoods

[All datasets](../README.md) · [Browsable card](al-birth-actual-velocities-20261004-199ab31e.html) · [Full metadata](../records/al-birth-actual-velocities-20261004-199ab31e.json)

Simultaneous saved MD velocities for exact original nearest80 birth observations; frozen original roles and labels, no new simulations. Coordinates are read from the unchanged birth cohort.

- ID: `al-birth-actual-velocities-20261004`
- Materials: Al
- Classification: **derived**; role: frozen_encoder_transfer_inputs
- Location: `/home/ids/vmorozov/training-cache/response-birth/cuda-meam-20261004-v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.011 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Actual velocities for frozen Al nucleus preappearance neighborhoods",
  "materials": [
    "Al"
  ],
  "role": "frozen_encoder_transfer_inputs",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Simultaneous saved MD velocities for exact original nearest80 birth observations; frozen original roles and labels, no new simulations. Coordinates are read from the unchanged birth cohort.",
  "provenance": "src/research/response_birth.py reconstructs all periodic neighborhoods exactly and reads velocities for the same atom identities and frames.",
  "lineage": "Identical original Al64 ancestors and birth rows; no additional independent sources.",
  "evidence": [
    "configs/analysis/response_birth_20261004.json",
    "docs/response_birth.md",
    "${dataset:al-birth-actual-velocities-20261004}/velocities.json"
  ],
  "limitations": [
    "Historical raw float16 velocity observations converted to float32 A/fs; no precision recovery.",
    "Retrospectively enriched isolated-establishment cases and liquid controls, not natural-risk calibration.",
    "Velocities available only in original unrelaxed domain for this evaluation."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-birth-actual-velocities-20261004-199ab31e.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
