# Ta shooting parent model_1m: 4/4 shots complete

[All datasets](../README.md) · [Browsable card](ta-shooting-20260926-parent00-81ebb020.html) · [Full metadata](../records/ta-shooting-20260926-parent00-81ebb020.json)



- ID: `ta-shooting-20260926-parent00`
- Materials: Ta
- Classification: **retired**; role: provenance_only
- Location: `/store/PERSO/vmorozov/simulations/ta-shooting-20260926-parent00`
- Present on this machine: False
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "materials": [
    "Ta"
  ],
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "role": "provenance_only",
  "classification": "retired",
  "title": "Ta shooting parent model_1m: 4/4 shots complete",
  "ancestry": "archived-Ta-unknown-common-preparation; all parents and descendants conservatively grouped",
  "parent": {
    "name": "model_1m",
    "source": "/work/PERSO/vmorozov/datasets/Ta/initial_configurations/model_1m.pos",
    "source_sha256": "1038b1c87e081ce501ca2724d3c16d73d93531a14bbb3f9ea55649d822c36856",
    "source_step": 1480000,
    "atom_count": 1024000
  },
  "evidence": [
    "/store/PERSO/vmorozov/simulation-launches/ta-shooting-20260926/manifest.json",
    "/store/PERSO/vmorozov/simulations/ta-shooting-20260926-parent00/status.json"
  ],
  "limitations": [
    "Position-conditioned shots share an archived parent; original generating potential is unknown."
  ],
  "retirement_record": "docs/storage/ta_simulation_removal_20261003/receipt.json",
  "storage_status": "Raw trajectories and native restarts deleted at user request on 2026-10-03. Reports/provenance saved under docs/storage/ta_simulation_removal_20261003/results. Original Ta snapshots and the earlier million-atom baseline retained."
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-shooting-20260926-parent00-81ebb020.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
