# Ti: duplicate Slurm branch exports

[All datasets](../README.md) · [Browsable card](ti-meam-early-slurm-copies-6395c863.html) · [Full metadata](../records/ti-meam-early-slurm-copies-6395c863.json)

Six reported copies of the original Ti branches. Current manifest signatures expose exact duplicate groups; do not count copies as independent trajectories.

- ID: `ti-meam-early-slurm-copies`
- Materials: Ti
- Classification: **retired**; role: provenance_only
- Location: `/work/PERSO/vmorozov/simulations/ti_ta_crystallization_20260907/Ti_early_slurm`
- Present on this machine: False
- Potentials: Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Ti: duplicate Slurm branch exports",
  "materials": [
    "Ti"
  ],
  "role": "provenance_only",
  "classification": "retired",
  "description": "Six reported copies of the original Ti branches. Current manifest signatures expose exact duplicate groups; do not count copies as independent trajectories.",
  "potential_ids": [
    "ti-kavousi2019-meam"
  ],
  "evidence": [
    "docs/simulations/ti_ta_crystallization/README.md",
    "${dataset:ti-meam-early-slurm-copies}/config.json"
  ],
  "lineage": "Ti descendants share one root source; Ta snapshots are not independent melts. Retain source/parent grouping.",
  "storage_status": "Duplicate branch exports deleted at user request on 2026-10-04 after byte-checksum verification of all 36 arrays against retained Ti branches.",
  "retirement_record": "docs/storage/ti_duplicates_float32_20261004/deletion.json"
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ti-meam-early-slurm-copies-6395c863.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
