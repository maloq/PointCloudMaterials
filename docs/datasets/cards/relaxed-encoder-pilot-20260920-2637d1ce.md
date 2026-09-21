# Paired observed/relaxed MACE pilot training_cache

[All datasets](../README.md) · [Browsable card](relaxed-encoder-pilot-20260920-2637d1ce.html) · [Full metadata](../records/relaxed-encoder-pilot-20260920-2637d1ce.json)

Full periodic fixed-box Lee2003 FIRE quenches; tracked observed nearest80 identities; hot/hot, hot/cold, cold/cold. Original MD defines crystallization outcomes.

- ID: `relaxed-encoder-pilot-20260920`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/relaxed_encoder/pilot-20260920`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.124 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Paired observed/relaxed MACE pilot training_cache",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
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
| lineage | ["independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_15458297", "independent_melt_197585375", "independent_melt_206158192", "independent_melt_210998254", "independent_melt_234122652", "independent_melt_248699045", "independent_melt_294342775", "independent_melt_340453041"] … (30 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al"] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| materials | [["Al"]] |

## Evidence

All 666 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/relaxed-encoder-pilot-20260920-2637d1ce.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T09:05:46.021728+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
