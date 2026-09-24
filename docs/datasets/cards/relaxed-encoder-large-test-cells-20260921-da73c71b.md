# Dense fixed-grid Al relaxed crystallization relaxed_cells

[All datasets](../README.md) · [Browsable card](relaxed-encoder-large-test-cells-20260921-da73c71b.html) · [Full metadata](../records/relaxed-encoder-large-test-cells-20260921-da73c71b.json)

150 existing independent 70,304-atom Al sources at 400-520 K, unchanged ancestry/splits. Readout training retains 15 origins; selection/calibration/test use 38 fixed origins spaced 12 ps. Full-cell Lee2003 MEAM fixed-box FIRE at force tolerance 0.01 eV/Angstrom. Reuse verified centered float32 clouds; archive new cells after verified float16 conversion. No encoder training or new MD.

- ID: `relaxed-encoder-large-test-cells-20260921`
- Materials: Al
- Classification: **research**; role: relaxed_cells
- Location: `/store/PERSO/vmorozov/relaxed_encoder/large-test-20260921`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 1440; these are not independent-source counts.
- Stored frames: 1440; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 3.680 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Dense fixed-grid Al relaxed crystallization relaxed_cells",
  "materials": [
    "Al"
  ],
  "role": "relaxed_cells",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "150 existing independent 70,304-atom Al sources at 400-520 K, unchanged ancestry/splits. Readout training retains 15 origins; selection/calibration/test use 38 fixed origins spaced 12 ps. Full-cell Lee2003 MEAM fixed-box FIRE at force tolerance 0.01 eV/Angstrom. Reuse verified centered float32 clouds; archive new cells after verified float16 conversion. No encoder training or new MD.",
  "evidence": [
    "configs/analysis/relaxed_encoder_large_test.json",
    "output/relaxed_encoder/large-test-20260921/technical/plan.json"
  ],
  "limitations": [
    "Planned 338 distinct local onsets on 30 test sources (27 event-bearing), not 338 independent nucleation events. Timeout exclusions shared across models. Historical source split. CPU/GPU FIRE minima may differ; backend and binary hashes retained."
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

All 4320 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/relaxed-encoder-large-test-cells-20260921-da73c71b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-24T10:36:51.012083+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
