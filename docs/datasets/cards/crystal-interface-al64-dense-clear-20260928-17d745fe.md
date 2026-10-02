# Dense interface-invisible Al contexts, unchanged fixed source roles

[All datasets](../README.md) · [Browsable card](crystal-interface-al64-dense-clear-20260928-17d745fe.html) · [Full metadata](../records/crystal-interface-al64-dense-clear-20260928-17d745fe.json)

Sealed expansion from 2,457,600 uniform candidate contexts: 256 centers on 64 evenly spaced frames from each of 150 existing Al trajectories. New geometry retained only when no interface is observed; all original rows preserved. Contains 1133196 cached contexts. Active liquid-distance training additionally excludes all observed established crystal and crystal-absent cells.

- ID: `crystal-interface-al64-dense-clear-20260928`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystal-interface/al64-dense-clear-20260928`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 17.630 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Dense interface-invisible Al contexts, unchanged fixed source roles",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Sealed expansion from 2,457,600 uniform candidate contexts: 256 centers on 64 evenly spaced frames from each of 150 existing Al trajectories. New geometry retained only when no interface is observed; all original rows preserved. Contains 1133196 cached contexts. Active liquid-distance training additionally excludes all observed established crystal and crystal-absent cells.",
  "provenance": "src/research/crystal_vector/expand.py; parent interface/PTM definitions unchanged; no new MD simulation.",
  "ancestry": "Inherited frozen al64-v1 independent-melt train/selection/calibration/test roles 90/15/15/30. Added frames and centers are correlated observations, not new independent sources.",
  "evidence": [
    "${storage:cache}/crystal-interface/al64-dense-clear-20260928/plan.json",
    "${storage:cache}/crystal-interface/al64-dense-clear-20260928/manifest.json",
    "configs/crystal_vector/interface_unseen_20260928.json",
    "experiments/crystal_interface_20260928/UNSEEN.md",
    "configs/crystal_vector/liquid_distance_20260928.json",
    "experiments/crystal_interface_20260928/LIQUID_DISTANCE.md"
  ],
  "limitations": [
    "Visibility is a label-side selection criterion, not an input or deployable absence oracle.",
    "Pre-rejection candidate denominators retained for conditional population weights.",
    "Geometry only; Al-only adaptation, with historically mixed-material supervised initialization.",
    "The active liquid-distance fit applies a stricter view: no established crystal anywhere in observed patches and a finite external-crystal target. Crystal-absent examples are evaluated separately."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["joint_snapshot_interface_unseen_v1"] |
| seed | [20260928] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 152 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystal-interface-al64-dense-clear-20260928-17d745fe.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
