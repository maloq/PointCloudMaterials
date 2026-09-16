# Implementations of recorded research protocols

`mace_encoder_diagnostics` tests the exact frozen forecast encoder's invariance,
geometry/membership sensitivity, storage precision, physical evolution and TDA
generalization. See the [protocol](../../experiments/mace_encoder_diagnostics_20260914/README.md).

These packages contain the training, preparation and analysis implementations
formerly kept in dated `experiments/` folders. Each method retains its scientific
objective and restart behavior. Configs and findings stay in the dated record;
source package names do not include run dates. Explicit historical config lookups
use dataset ID `research-records-20260913`; this names the verified STORE records.

Use `python -m src.research.METHOD.MODULE` from the repository root with `pointnet`,
using the arguments recorded in that experiment's README. Retired records link to
the STORE archive; use its original checkout for historical commands that rely on
archived inputs/configuration paths. Maintained current
trainers remain under `src/training_methods/`, analysis under `src/analysis/`, and
shared data producers under `src/data_utils/`. Do not create a new research module
for a dataset/seed/output-path change that an existing command can express.

| Package | Record |
| --- | --- |
| `mace_velocity` | [Local coordinate/velocity encoder](../../experiments/mace_velocity_20260915/README.md): smooth structure, time-even activity and time-odd motion, source-isolated holdouts and velocity interventions; no forecasting objective |
| `mace_local_state` | [Frozen local-group states](../../experiments/mace_local_state_20260915/README.md): short-time canonical coordinates, learned group-physics distance and uncertain density discovery; no forecasting objective |
| `mace_local_state/smooth*` | [Direct temporal training](../../experiments/mace_local_smooth_20260915/README.md): nonlinear physical-state maps of frozen velocity-checkpoint features, direct within-context slowness and matched information tests |
| `mace_tda_ridge_audit` | [mace_tda_ridge_audit_20260914](../../experiments/mace_tda_ridge_audit_20260914/README.md); fresh six-checkpoint inference, topology-loss interventions and independent ridge calculations in projector and encoder spaces; frozen MLIP/random initialization controls and direct comparisons without a readout |
| `forecast_spatial_mixture` | [forecast_spatial_mixture_20260913](../../experiments/forecast_spatial_mixture_20260913/README.md); matched point/distribution readouts, paired transitions, learned [individual-neighbor attention](../../experiments/forecast_spatial_attention_20260914/README.md), and measured structure / embedding future-path visualization |
| `forecast_crystallization` | [forecast_crystallization_20260913](../../experiments/forecast_crystallization_20260913/README.md); local PTM state, transition timing and frozen-forecast evaluation |
| `forecast_context` | [forecast_context_20260913](../../experiments/forecast_context_20260913/README.md); paired analysis of histories fitted with the maintained forecaster |
| `factor_vae` | [factor_vae_20260901](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/factor_vae_20260901/README.md) |
| `geoframe_continuity` | [geoframe_continuity_20260905](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/geoframe_continuity_20260905/README.md) |
| `liquid_sro_benchmark` | [liquid_sro_benchmark_20260905](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/liquid_sro_benchmark_20260905/README.md) |
| `mace_al_denoising` | [mace_al_denoising_20260910](../../experiments/mace_al_denoising_20260910/README.md) |
| `mace_bf16_training` | [mace_bf16_training_20260908](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_bf16_training_20260908/README.md) |
| `mace_diagnosis` | [mace_diagnosis_20260905](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_diagnosis_20260905/README.md) |
| `mace_original_vicreg` | [mace_original_vicreg_20260909](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_original_vicreg_20260909/README.md) |
| `mace_vicreg_audit` | [mace_vicreg_audit_20260909](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_vicreg_audit_20260909/README.md) |
| `mace_vicreg_relaxed` | [mace_vicreg_relaxed_20260910](../../experiments/mace_vicreg_relaxed_20260910/README.md) |
| `predictive_encoder_training` | [predictive_encoder_training_20260905](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/predictive_encoder_training_20260905/README.md) |
| `restart_audit` | [restart_audit_20260905](../../docs/simulations/restart_audit/README.md) |
| `smooth_temporal_encoder` | [smooth_temporal_encoder_20260905](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/smooth_temporal_encoder_20260905/README.md) |
| `spatiotemporal` | [spatiotemporal_20260905](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/spatiotemporal_20260905/README.md) |
| `temporal_hypotheses_12h` | [temporal_hypotheses_12h_20260906](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/temporal_hypotheses_12h_20260906/README.md) |

`mace_context/` owns the complete-context and center-node MACE pilot: original source/center reconstruction, frozen diagnostics, matched VICReg continuation and source-held-out topology readouts. Its `smoothness` stage compares temporal increments with two training-only scales and paired source intervals, separately from controlled membership crossings. Reusable graph computation is in `src/models/encoders/mace_context.py`.

The `recovery_*` modules in that package own the separate nonlinear-readout,
combined-center/inner and joint physical-supervision protocols. Their recipe is
`configs/analysis/mace_context_recovery.json`; see the
[recovery experiment](../../experiments/mace_context_recovery_20260914/README.md).

`mace_context/static.py` exports and verifies that joint checkpoint for the standard
Al/Zr analysis; `src/analysis/mace_context_adapter.py` reuses complete two-hop atom
features for its overlapping readouts. See the [static study](../../experiments/mace_context_static_20260915/README.md).

`mace_context/cluster_diagnosis.py` separates liquid spatial coherence from local
and collective physical information using saved-feature ablations and exact
center alignment with archived GeoFrame V2 results.
`mace_context/cluster_probe.py` follows with spatially separated frozen-feature
physical readouts to distinguish missing information from a poor cluster metric.

`mace_local_state/motion*` implements the distinct [consecutive local-motion protocol](../../experiments/mace_local_motion_20260916/README.md), with actual cadence, shared current-state directions, and within-condition physical retention.

`mace_velocity/data_amount*.py` implements the distinct native end-to-end encoder learning curve; use existing module stages `data-prepare`, `data-smoke`, `data-study`. See [protocol](../../experiments/mace_data_amount_20260916/README.md).
