# Ta: five 10-million-atom branches

[All datasets](../README.md) · [Browsable card](ta-eam-five-24ps-0b8a1ae4.html) · [Full metadata](../records/ta-eam-five-24ps-0b8a1ae4.json)

Five 10,000,422-atom NPT branches at 1900 K, 24 ps, 0.1 ps frames, initialized from archived positions.

- ID: `ta-eam-five-24ps`
- Materials: Ta
- Classification: **retired**; role: provenance_only
- Location: `/work/PERSO/vmorozov/simulations/ti_ta_crystallization_20260907/Ta`
- Present on this machine: False
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Ta: five 10-million-atom branches",
  "materials": [
    "Ta"
  ],
  "role": "provenance_only",
  "classification": "retired",
  "description": "Five 10,000,422-atom NPT branches at 1900 K, 24 ps, 0.1 ps frames, initialized from archived positions.",
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "evidence": [
    "docs/simulations/ti_ta_crystallization/README.md",
    "${dataset:ta-eam-five-24ps}/config.json"
  ],
  "lineage": "Ti descendants share one root source; Ta snapshots are not independent melts. Retain source/parent grouping.",
  "retirement_record": "docs/storage/ta_simulation_removal_20261003/receipt.json",
  "storage_status": "Raw trajectories and native restarts deleted at user request on 2026-10-03. Reports/provenance saved under docs/storage/ta_simulation_removal_20261003/results. Original Ta snapshots and the earlier million-atom baseline retained."
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-eam-five-24ps-0b8a1ae4.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
