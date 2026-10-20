# Ni small-velocity-perturbation NPT continuation pilot

[All datasets](../README.md) · [Browsable card](ni-ordering-branches-20261019-15d0d768.html) · [Full metadata](../records/ni-ordering-branches-20261019-15d0d768.json)

24present-liquid origins,16KE-preserving1%-RMSvelocity perturbations each plus24unperturbedcontrols; original1fsNPTsettings and source temperature,6ps horizon,0.1pslocal80measurements and full-cell phase/component observations.

- ID: `ni-ordering-branches-20261019`
- Materials: Ni
- Classification: **research**; role: physical_continuation_diagnostic
- Location: `/store/PERSO/vmorozov/simulations/ni-ordering-branches-20261019-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.012 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Ni small-velocity-perturbation NPT continuation pilot",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "physical_continuation_diagnostic",
  "classification": "research",
  "description": "24present-liquid origins,16KE-preserving1%-RMSvelocity perturbations each plus24unperturbedcontrols; original1fsNPTsettings and source temperature,6ps horizon,0.1pslocal80measurements and full-cell phase/component observations.",
  "ancestry": "24existingancestors:12train/12test;branches nested within origins,not independent source histories.",
  "sample_interval_ps": 0.1,
  "evidence": [
    "docs/simulations/ni_ordering_branches_20261019.md"
  ],
  "limitations": [
    "Initial Q/V/box are quantized saved observations; NPT extended states reinitialized, not native replay.",
    "A declared perturbation ensemble, not exact conditional calibration of unperturbed saved dynamics.",
    "Current completion receipts determine available branches."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-ordering-branches-20261019-15d0d768.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
