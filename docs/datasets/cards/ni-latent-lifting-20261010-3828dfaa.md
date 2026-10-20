# Ni empirical decoder physical continuations

[All datasets](../README.md) · [Browsable card](ni-latent-lifting-20261010-3828dfaa.html) · [Full metadata](../records/ni-latent-lifting-20261010-3828dfaa.json)

Original and latent-nearest or random training full-cellQ,V continue by fixed-boxBAOAB at matching thermostatT.0.1/0.5ps, two independent shots. Ancestry stays with selected source; no new independent melts.

- ID: `ni-latent-lifting-20261010`
- Materials: Ni
- Classification: **research**; role: simulation_response_diagnostic
- Location: `/store/PERSO/vmorozov/simulations/ni-latent-lifting-20261010-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.006 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni empirical decoder physical continuations",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "simulation_response_diagnostic",
  "classification": "research",
  "description": "Original and latent-nearest or random training full-cellQ,V continue by fixed-boxBAOAB at matching thermostatT.0.1/0.5ps, two independent shots. Ancestry stays with selected source; no new independent melts.",
  "evidence": [
    "docs/ni_latent_sim.md"
  ],
  "limitations": [
    "Empirical training-state retrieval is not learned atomic generation.",
    "Small-ensemble distribution discrepancy is a diagnostic, not a nucleation rate."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["original full32000 Q,V; own fixed box; source thermostat T; 1fs BAOAB,1ps friction; independent two-shot 100/500fs continuations"] |

## Evidence

All 49 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-latent-lifting-20261010-3828dfaa.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
