# Paired observed/relaxed MACE pilot relaxed_cells

[All datasets](../README.md) · [Browsable card](relaxed-encoder-cells-20260920-d76dbdc7.html) · [Full metadata](../records/relaxed-encoder-cells-20260920-d76dbdc7.json)

Full periodic fixed-box Lee2003 FIRE quenches; tracked observed nearest80 identities; hot/hot, hot/cold, cold/cold. Original MD defines crystallization outcomes.

- ID: `relaxed-encoder-cells-20260920`
- Materials: Al
- Classification: **research**; role: relaxed_cells
- Location: `/store/PERSO/vmorozov/relaxed_encoder/pilot-20260920`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 230; these are not independent-source counts.
- Stored frames: 230; duplicate-group records: 48
- Allocated storage, excluding registered nested datasets: 0.721 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Paired observed/relaxed MACE pilot relaxed_cells",
  "materials": [
    "Al"
  ],
  "role": "relaxed_cells",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Full periodic fixed-box Lee2003 FIRE quenches; tracked observed nearest80 identities; hot/hot, hot/cold, cold/cold. Original MD defines crystallization outcomes.",
  "evidence": [
    "configs/analysis/relaxed_encoder_pilot.json",
    "output/relaxed_encoder/pilot-20260920/technical/plan.json"
  ],
  "limitations": [
    "Production in progress; consult completion receipts. One seed, two observed origins per source. Parent pretrained on historical development sources; calibration/test ancestry protected."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation."] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [1] |
| storage_dtype | ["float16"] |

## Evidence

All 690 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/relaxed-encoder-cells-20260920-d76dbdc7.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
