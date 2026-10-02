# Ta known-parent shooting distance evaluation

[All datasets](../README.md) · [Browsable card](distance-encoder-ta-material-eval-20260927-b02db3b2.html) · [Full metadata](../records/distance-encoder-ta-material-eval-20260927-b02db3b2.json)

Four completed 1,024,000-atom branches; shot00 selection, shots01-03 test. 10,240 tracked centers, six 0.70-ps observations, six anchors per branch.

- ID: `distance-encoder-ta-material-eval-20260927`
- Materials: Ta
- Classification: **research**; role: evaluation_cache
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/ta-shooting-material-eval-20260927-v2`
- Present on this machine: True
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.997 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ta known-parent shooting distance evaluation",
  "materials": [
    "Ta"
  ],
  "role": "evaluation_cache",
  "classification": "research",
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "description": "Four completed 1,024,000-atom branches; shot00 selection, shots01-03 test. 10,240 tracked centers, six 0.70-ps observations, six anchors per branch.",
  "ancestry": "One archived Ta preparation; parent encoder saw a different trajectory from the same initial structure.",
  "limitations": [
    "Conditional new-velocity evaluation, not independent-preparation generalization.",
    "Parent was already trained on all six older Ta trajectories."
  ],
  "evidence": [
    "configs/distance_encoder/material_ta_20260927.json",
    "docs/distance_encoder_material_finetune.md"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cadence_ps | [0.5, 0.7] |
| protocol | ["external-crystallization-origin-v1"] |
| ptm_rmsd_cutoff | [0.1] |
| horizons_ps | [[3.0, 6.0]] |
| seed | [20260926, 20260927] |
| lineage | [{"interface_distance_A": 8.233252184899222, "local_birth_radius_A": 8.233252184899222, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.70496348320465, "thresholds": [{"name": "primary", "persistence_frames": 4, "persistence_ps": 1.5, "size": 64}, {"name": "size32", "persistence_frames": 4, "persistence_ps": 1.5, "size": 32}, {"name": "size128", "persistence_frames": 4, "persistence_ps": 1.5, "size": 128}, {"name": "persistent3ps", "persistence_frames": 7, "persistence_ps": 3.0, "size": 64}]}, {"interface_distance_Al_equivalent_A": 8.0, "local_birth_radius_Al_equivalent_A": 8.0, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_Al_equivalent_A": 3.6, "thresholds": [{"name": "primary", "persistence_ps": 1.5, "size": 64}, {"name": "size32", "persistence_ps": 1.5, "size": 32}, {"name": "size128", "persistence_ps": 1.5, "size": 128}, {"name": "persistent3ps", "persistence_ps": 3.0, "size": 64}]}] |
| material | ["Ta"] |
| atom_count | [1024000] |
| frame_count | [49] |
| parent_id | ["model_1m"] |
| velocity_seed | [186999831, 32816276, 608134340, 688081017] |

## Evidence

All 6 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-ta-material-eval-20260927-b02db3b2.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
