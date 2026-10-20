# Ni expanded100-fs physical teachers for density-rate conditioning

[All datasets](../README.md) · [Browsable card](ni-density-rates-response-20261013-cc129257.html) · [Full metadata](../records/ni-density-rates-response-20261013-cc129257.json)

Full-cell fixed-box FastNi BAOAB, actual Q/V, f64 integration/tangents/noise and f32 forces;1-fs physical steps and5-fs local80 recording. Only new anchors stored here; reuse prior bank by checksum.

- ID: `ni-density-rates-response-20261013`
- Materials: Ni
- Classification: **preparing**; role: physical_response
- Location: `/store/PERSO/vmorozov/simulations/ni-density-rates-20261013-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.330 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "classification": "preparing",
  "ancestry": "90 existing independent Ni melts;28 train and62 pooled test; original source roles preserved in plan",
  "source_plan_identity": "07b8b005cf8ce53c0f139d3ea653a6d9899cc7b25ad2fcdd5d46dccfbd3ab38f",
  "evidence": [
    "configs/ni_latent_sim/density_rates_20261013.json",
    "docs/measurement_sim.md",
    "/store/PERSO/vmorozov/experiments/measurement_sim/density-rates-20261013-v1/technical/plan.json"
  ],
  "title": "Ni expanded100-fs physical teachers for density-rate conditioning",
  "role": "physical_response",
  "planned_new_anchors": 451,
  "reused_anchors": 300,
  "description": "Full-cell fixed-box FastNi BAOAB, actual Q/V, f64 integration/tangents/noise and f32 forces;1-fs physical steps and5-fs local80 recording. Only new anchors stored here; reuse prior bank by checksum."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [101016760478919, 102428685568468, 103491679861642, 104366043549021, 104893036313879, 104904280387156, 105622981620685, 105810224460982, 106841773546521, 106944016204513, 10765385424984, 108177956450054] … (451 values; see JSON) |
| protocol | ["full-cell fixed-box BAOAB, actual saved Q,V, source T, no relaxation"] |

## Evidence

All 451 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-density-rates-response-20261013-cc129257.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
