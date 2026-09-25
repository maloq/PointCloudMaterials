# portability-ti-smoke-20260913

[All datasets](../README.md) · [Browsable card](portability-ti-smoke-20260913-d332e0ec.html) · [Full metadata](../records/portability-ti-smoke-20260913-d332e0ec.json)

Registered reference or supporting artifact.

- ID: `portability-ti-smoke-20260913`
- Materials: Ti
- Classification: **fixture**; role: administrative
- Location: `/store/PERSO/vmorozov/simulations/portability-ti-smoke-20260913`
- Present on this machine: True
- Potentials: Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 1; these are not independent-source counts.
- Stored frames: 5; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "portability-ti-smoke-20260913",
  "materials": [],
  "role": "administrative",
  "classification": "fixture",
  "description": "Registered reference or supporting artifact.",
  "evidence": [
    "${dataset:portability-ti-smoke-20260913}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| material | ["Ti"] |
| atom_count | [128] |
| frame_count | [5] |
| timestep_ps | [0.001] |
| protocol | ["integration/storage validation only"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| storage_dtype | ["float16"] |

## Evidence

All 4 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/portability-ti-smoke-20260913-d332e0ec.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
