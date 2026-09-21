# Native Al 32768 order anchors and frozen random MACE reservoir

[All datasets](../README.md) · [Browsable card](neighborhood-jepa-native-al-32768-order-20260920-81a8ce82.html) · [Full metadata](../records/neighborhood-jepa-native-al-32768-order-20260920-81a8ce82.json)

Derived current/next local order targets and unlabeled fixed random geometric features on the existing 32768 training / 480 development anchor release; 90/15 independent Lee-MEAM lineages, 0.75 ps lag. No new simulations.

- ID: `neighborhood-jepa-native-al-32768-order-20260920`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/neighborhood_jepa/native-al-32768-order-20260920`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.020 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Native Al 32768 order anchors and frozen random MACE reservoir",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "description": "Derived current/next local order targets and unlabeled fixed random geometric features on the existing 32768 training / 480 development anchor release; 90/15 independent Lee-MEAM lineages, 0.75 ps lag. No new simulations.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "provenance": {
    "producer": "src/training_methods/neighborhood_jepa/regularization/data.py",
    "parent": "neighborhood-jepa-native-al-32768-v2-20260920",
    "order_definition": "src/analysis/liquid_structure.py:bond_order; q4/q6/w4/w6/qbar6/coherence/density/coordination",
    "normalization": "training-only current and next center targets, per-component std floor 1e-4",
    "reservoir": "Random width16 MACE, seed9173, orthogonal 128-to-64 projection, no labels or learned parent weights"
  },
  "limitations": [
    "Order neighbors restricted to encoder-observed local crop; not guaranteed identical to full-trajectory handcrafted baseline neighborhoods.",
    "One random reservoir initialization.",
    "No new independent test cohort."
  ],
  "evidence": [
    "${dataset:neighborhood-jepa-native-al-32768-order-20260920}/manifest.json",
    "${dataset:neighborhood-jepa-native-al-32768-order-20260920}/reservoir.json"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| split | ["selection", "train"] |
| materials | [["Al"]] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/neighborhood-jepa-native-al-32768-order-20260920-81a8ce82.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
