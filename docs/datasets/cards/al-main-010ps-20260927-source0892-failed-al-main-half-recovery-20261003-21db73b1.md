# al-main-010ps-20260927-source0892-failed-al-main-half-recovery-20261003

[All datasets](../README.md) · [Browsable card](al-main-010ps-20260927-source0892-failed-al-main-half-recovery-20261003-21db73b1.html) · [Full metadata](../records/al-main-010ps-20260927-source0892-failed-al-main-half-recovery-20261003-21db73b1.json)

Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.

- ID: `al-main-010ps-20260927-source0892-failed-al-main-half-recovery-20261003`
- Materials: Al
- Classification: **incomplete_or_rejected**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/simulations/al-main-010ps-20260927-source0892-failed-al-main-half-recovery-20261003`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.047 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "raw_dynamics",
  "classification": "incomplete_or_rejected",
  "ancestry": "independent_melt_392346167",
  "split": "train",
  "source_id": 892,
  "sampling_ps": 0.1,
  "protocol": "al_dense_half_stop_010ps_v1",
  "description": "Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.",
  "evidence": [
    "/store/PERSO/vmorozov/simulations/al-main-010ps-20260927-source0892-failed-al-main-half-recovery-20261003/input_metadata.json"
  ],
  "limitations": [
    "2026-10-07: obsolete stopped trajectory.lammpstrj deleted at user request after verifying its completed native-restart successor al-main-010ps-20260927-source0892. Completed 0.10-ps positions/velocities and all native checkpoints are retained; this failed-attempt folder now contains provenance/checkpoints only."
  ],
  "storage_cleanup": {
    "date": "2026-10-07",
    "removed_files": [
      "trajectory.lammpstrj"
    ],
    "completed_successor": "al-main-010ps-20260927-source0892",
    "receipt": "docs/storage/store_cleanup_20261007/stopped-payload-removal.json"
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-main-010ps-20260927-source0892-failed-al-main-half-recovery-20261003-21db73b1.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
