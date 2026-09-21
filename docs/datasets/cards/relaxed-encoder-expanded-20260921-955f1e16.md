# Expanded paired observed/relaxed Al training_cache

[All datasets](../README.md) · [Browsable card](relaxed-encoder-expanded-20260921-955f1e16.html) · [Full metadata](../records/relaxed-encoder-expanded-20260921-955f1e16.json)

90 training and 15 development sources; four encoder-training origins; 15 regular-grid assay origins across 150 sources. Full periodic fixed-box Lee2003 MEAM FIRE quenches at 0.01 eV/Angstrom. Tracked nearest80 candidates with radius8 support. Reuses verified pilot and older target clouds where query identities match. Validated CUDA/KOKKOS H100/A100 workers supplement CPU production; per-cell metadata records binary and backend. GPU rounding can select different FIRE minima despite matching initial forces and force tolerance.

- ID: `relaxed-encoder-expanded-20260921`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/relaxed_encoder/expanded-20260921`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 8.670 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Expanded paired observed/relaxed Al training_cache",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "90 training and 15 development sources; four encoder-training origins; 15 regular-grid assay origins across 150 sources. Full periodic fixed-box Lee2003 MEAM FIRE quenches at 0.01 eV/Angstrom. Tracked nearest80 candidates with radius8 support. Reuses verified pilot and older target clouds where query identities match. Validated CUDA/KOKKOS H100/A100 workers supplement CPU production; per-cell metadata records binary and backend. GPU rounding can select different FIRE minima despite matching initial forces and force tolerance.",
  "evidence": [
    "configs/analysis/relaxed_encoder_expanded.json",
    "output/relaxed_encoder/expanded-20260921/technical/plan.json",
    "output/relaxed_encoder/expanded-20260921/technical/accelerated/launches.json"
  ],
  "limitations": [
    "Preparing; consult per-cell completion receipts. One seed. Reused historical source split, no test-source encoder fitting. Full-cell quenching includes external context. Raw MD coordinates inherit float16 storage; relaxed local clouds saved float32 before full-cell conversion. Training-only target-domain normalization.",
    "Timeouts are excluded consistently across matched training pairs/readout windows; technical/skipped and training/assay release receipts record omissions. Other failures remain fatal."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation."] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297", "independent_melt_160380922", "independent_melt_176364202"] … (105 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al"] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| materials | [["Al"]] |

## Evidence

All 2384 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/relaxed-encoder-expanded-20260921-955f1e16.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
