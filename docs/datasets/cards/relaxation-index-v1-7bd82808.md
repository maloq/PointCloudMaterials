# Unified relaxation index and shared native restart store

[All datasets](../README.md) · [Browsable card](relaxation-index-v1-7bd82808.html) · [Full metadata](../records/relaxation-index-v1-7bd82808.json)

Recipe-bound pointers to converged periodic full cells and pre-quantization float32 neighborhoods; exact membership lookup and missing-work planning. Future minimizations preserve one shared native restart for arbitrary new neighborhoods.

- ID: `relaxation-index-v1`
- Materials: Al
- Classification: **operational**; role: administrative
- Location: `/store/PERSO/vmorozov/relaxation-index/v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 2; these are not independent-source counts.
- Stored frames: 3; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 37.741 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Unified relaxation index and shared native restart store",
  "materials": [
    "Al"
  ],
  "role": "administrative",
  "classification": "operational",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Recipe-bound pointers to converged periodic full cells and pre-quantization float32 neighborhoods; exact membership lookup and missing-work planning. Future minimizations preserve one shared native restart for arbitrary new neighborhoods.",
  "provenance": "Source manifests, archive checksums, potential hashes, engine evidence and exact original atom memberships; index entries retain their original archives.",
  "evidence": [
    "configs/relaxation_index.json",
    "docs/data_storage.md",
    "${dataset:relaxation-index-v1}/inventory.json",
    "${dataset:relaxation-index-v1}/scan.json"
  ],
  "limitations": [
    "An index is not an independent dataset or a train/test split. Original source ancestry and experiment roles remain unchanged.",
    "Historical float16 global coordinates cannot substitute for pre-quantization float32 neighborhoods. Unpinned historical engines remain visible but are not automatic recipe matches.",
    "Current backfill contains Al; the API uses the supplied monatomic periodic-cell potential rather than an Al-specific model."
  ],
  "relaxation_source_availability_note": "Some historical relaxed-cell/neighborhood inputs were deleted at user request on 2026-10-03; retained downstream artifacts are unchanged. See docs/storage/relaxation_unverified_removal_20261003/receipt.json.",
  "storage_status": "4,942 recipe-unverified entries retired at user request on 2026-10-03; provenance retained, payloads deleted. 13,743 recipe-bound entries remain available."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [32, 32000] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation.", "fixed_periodic_cell_norm_inf_v1"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [1, 2] |
| storage_dtype | ["float16"] |

## Evidence

All 18798 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/relaxation-index-v1-7bd82808.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
