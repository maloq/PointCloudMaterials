# GeoFrame Al/Ta/Zr independent physical reference assay

[All datasets](../README.md) · [Browsable card](geoframe-physical-reference-20260923-6b117f18.html) · [Full metadata](../records/geoframe-physical-reference-20260923-6b117f18.json)

Nine original inherent snapshots, full-cell PTM/Al planar faults and 4096 uniform interior anchors plus their nearest neighbors per snapshot; fixed spatial holdout, q12/q14 and topology descriptors. Analysis arrays only; no new simulation.

- ID: `geoframe-physical-reference-20260923`
- Materials: Al, Ta, Zr
- Classification: **research**; role: analysis_cache
- Location: `/home/infres/vmorozov/PointCloudMaterials/output/geoframe_evolution/epoch34-reproduction-20260923/technical/reference`
- Present on this machine: False
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "GeoFrame Al/Ta/Zr independent physical reference assay",
  "materials": [
    "Al",
    "Ta",
    "Zr"
  ],
  "role": "analysis_cache",
  "classification": "research",
  "potential_ids": [],
  "description": "Nine original inherent snapshots, full-cell PTM/Al planar faults and 4096 uniform interior anchors plus their nearest neighbors per snapshot; fixed spatial holdout, q12/q14 and topology descriptors. Analysis arrays only; no new simulation.",
  "evidence": [
    "configs/geoframe_evolution/assay.json",
    "src/research/geoframe_evolution/reference.py",
    "output/geoframe_evolution/epoch34-reproduction-20260923/technical/reference/manifest.json"
  ],
  "limitations": [
    "Original static generating potentials and periodic cell metadata are unknown. Free boundaries with interior exclusion. Encoder training includes these frames; probe spatial holdout is transductive. Candidate order labels do not establish temporal persistence or nucleation."
  ],
  "ancestry": "Derived directly from registered Al, Ta and Zr original inherent configurations, retaining each input file hash and atom row index; no claim of independent snapshot ancestry."
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/geoframe-physical-reference-20260923-6b117f18.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
