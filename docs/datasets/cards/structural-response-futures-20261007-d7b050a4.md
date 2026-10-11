# Broad structural and exact future geometry targets for derivative motion

[All datasets](../README.md) · [Browsable card](structural-response-futures-20261007-d7b050a4.html) · [Full metadata](../records/structural-response-futures-20261007-d7b050a4.json)

1,056,768 reused packed spatial patches; 4096 selection diagnostic patches; 11200 aligned native MD future neighborhoods at0.1/0.5/3/12ps. Full source identities and hashes in complete.json and spatial.json.

- ID: `structural-response-futures-20261007`
- Materials: Al, Mg, Ta, Ti
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/structural-response/c3-l3-bf16-20261007-v1`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.065 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "Broad structural and exact future geometry targets for derivative motion",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta"
  ],
  "potential_ids": [],
  "role": "training_cache",
  "classification": "research",
  "description": "1,056,768 reused packed spatial patches; 4096 selection diagnostic patches; 11200 aligned native MD future neighborhoods at0.1/0.5/3/12ps. Full source identities and hashes in complete.json and spatial.json.",
  "provenance": {
    "recipe": "configs/structural_response/overnight_20261007.json",
    "workflow": "docs/structural_response.md",
    "state": "prepared"
  },
  "lineage": "Existing multimaterial training ancestors and88 Al views/84 original Al64 ancestors; no new independent simulation ancestry.",
  "limitations": [
    "Potential provenance remains per source in parent collections: Al motion uses verified Lee2003 MEAM; spatial mixed-material generating potentials are not homogeneous.",
    "Derived positions preserve original snapshot quantization; no interpolation or MD restart.",
    "Original roles retained; pooled held-out roles used only for motion evaluation."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/structural-response-futures-20261007-d7b050a4.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
