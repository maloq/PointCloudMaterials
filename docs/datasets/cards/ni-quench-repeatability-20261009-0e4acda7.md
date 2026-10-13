# Ni full-cell quench repeatability and input-sensitivity diagnostics

[All datasets](../README.md) · [Browsable card](ni-quench-repeatability-20261009-0e4acda7.html) · [Full metadata](../records/ni-quench-repeatability-20261009-0e4acda7.json)

Independent full-cell quenches of five pinned 32000-atom signed-MD endpoints and their matched perturbed branches. Original source ancestry and atom identity retained. No independent MD sources or new nucleus labels. Raw float32, canonical float32, tighter force tolerance and one float64-force reference.

- ID: `ni-quench-repeatability-20261009`
- Materials: Ni
- Classification: **research**; role: numerical_diagnostic
- Location: `/store/PERSO/vmorozov/simulations/ni-quench-repeatability-20261009-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 89; these are not independent-source counts.
- Stored frames: 89; duplicate-group records: 22
- Allocated storage, excluding registered nested datasets: 0.235 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni full-cell quench repeatability and input-sensitivity diagnostics",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "numerical_diagnostic",
  "classification": "research",
  "description": "Independent full-cell quenches of five pinned 32000-atom signed-MD endpoints and their matched perturbed branches. Original source ancestry and atom identity retained. No independent MD sources or new nucleus labels. Raw float32, canonical float32, tighter force tolerance and one float64-force reference.",
  "evidence": [
    "${storage:training_storage}/ni_relaxation/repeatability-20261009-v1/technical/plan.json",
    "docs/ni_relaxation_repeatability.md"
  ],
  "limitations": [
    "Five preselected diagnostic sources, not a representative cohort.",
    "Reproducibility does not establish basin robustness to input perturbations.",
    "Separate recipes; do not substitute these states into historical training banks."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [32000] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation."] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [1] |
| storage_dtype | ["float16"] |

## Evidence

All 445 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-quench-repeatability-20261009-0e4acda7.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
