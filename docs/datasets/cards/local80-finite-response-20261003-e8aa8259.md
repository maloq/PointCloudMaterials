# Local80 finite-displacement MACE forward responses

[All datasets](../README.md) · [Browsable card](local80-finite-response-20261003-e8aa8259.html) · [Full metadata](../records/local80-finite-response-20261003-e8aa8259.json)

384 perturbed100fs MACE forwards and16 central AD trajectories, observed20/100fs; four train parents, two directions, both signs, two noise seeds,24/36A environments and collective displacement norms0.03–3A. Saves feature changes and scalar physical diagnostics; no trajectory-position arrays.

- ID: `local80-finite-response-20261003`
- Materials: Al
- Classification: **research**; role: finite_response_diagnostic
- Location: `/store/PERSO/vmorozov/simulations/local80-finite-response-20261003`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.018 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Local80 finite-displacement MACE forward responses",
  "materials": [
    "Al"
  ],
  "role": "finite_response_diagnostic",
  "classification": "research",
  "potential_ids": [
    "mace-mpa-0-medium",
    "al-lee2003-meam"
  ],
  "description": "384 perturbed100fs MACE forwards and16 central AD trajectories, observed20/100fs; four train parents, two directions, both signs, two noise seeds,24/36A environments and collective displacement norms0.03\u20133A. Saves feature changes and scalar physical diagnostics; no trajectory-position arrays.",
  "ancestry": "Parents0,1,2,45 from local80-response-20261002; fixed Al64 e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d. Training ancestors only; no new independent MD roots.",
  "provenance": "Original parent geometry, atom identities and original train-gate noise streams; MACE SHA25675428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638. Float32 cuEquivariance BAOAB1fs; original full-cell float64 RNG before gather.",
  "evidence": [
    "${storage:repo}/configs/simulation/local_response_followup_20261003.json",
    "${storage:repo}/docs/simulations/local_response_followup_20261003.md"
  ],
  "limitations": [
    "MACE future physics differs from original MEAM parent dynamics.",
    "Nested environments do not establish complete-periodic-cell equivalence.",
    "Large displacement stress assay, not an equilibrium perturbation distribution.",
    "Two noise samples and four training parents; no held-out scientific generalization claim."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/local80-finite-response-20261003-e8aa8259.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
