# Ta shooting parent 2.8ns: 4/4 shots complete

[All datasets](../README.md) · [Browsable card](ta-shooting-20260926-parent02-c0d327f9.html) · [Full metadata](../records/ta-shooting-20260926-parent02-c0d327f9.json)



- ID: `ta-shooting-20260926-parent02`
- Materials: Ta
- Classification: **retired**; role: provenance_only
- Location: `/store/PERSO/vmorozov/simulations/ta-shooting-20260926-parent02`
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
  "title": "Ta shooting parent 2.8ns: 4/4 shots complete",
  "ancestry": "archived-Ta-unknown-common-preparation; all parents and descendants conservatively grouped",
  "parent": {
    "name": "2.8ns",
    "source": "/work/PERSO/vmorozov/datasets/Ta/initial_configurations/2.8ns.pos",
    "source_sha256": "7d20d67091eedf2f24a27978a5bc2e33c1755ed12f3ecd37ccfcfafff6b2376a",
    "source_step": 1400000,
    "atom_count": 10000422
  },
  "evidence": [
    "/store/PERSO/vmorozov/simulation-launches/ta-shooting-20260926/manifest.json",
    "/store/PERSO/vmorozov/simulations/ta-shooting-20260926-parent02/status.json"
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

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-shooting-20260926-parent02-c0d327f9.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
