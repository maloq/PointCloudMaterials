# CDV-MACE128 fixed Al64 spatial distance and direction

[All datasets](../README.md) · [Browsable card](crystal-vector-al64-covering-20260927-974862e2.html) · [Full metadata](../records/crystal-vector-al64-covering-20260927-974862e2.json)

Single-snapshot 25-patch distance-only covering, existing fixed Al64 samples plus uniform centers and original scans; past-confirmed crystal vectors, no history inputs. 177929 contexts over 150 source roles. Parent encoder was mixed-material.

- ID: `crystal-vector-al64-covering-20260927`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystal-vector/al64-covering-20260927`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 2.803 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "CDV-MACE128 fixed Al64 spatial distance and direction",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Single-snapshot 25-patch distance-only covering, existing fixed Al64 samples plus uniform centers and original scans; past-confirmed crystal vectors, no history inputs. 177929 contexts over 150 source roles. Parent encoder was mixed-material.",
  "evidence": [
    "${storage:cache}/crystal-vector/al64-covering-20260927/plan.json",
    "${storage:cache}/crystal-vector/al64-covering-20260927/manifest.json",
    "experiments/crystal_vector_20260927/README.md"
  ],
  "limitations": [
    "Correlated frames/centers; fixed source/ancestry roles inherited; directional targets undefined at zero/censored/equal-nearest distances.",
    "Changed context sampler compared with historical lab-fixed stencil; new arms matched to one another."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["crystal_vector_snapshot_al64_v1"] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 152 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystal-vector-al64-covering-20260927-974862e2.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
