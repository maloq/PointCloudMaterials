# Implementations of recorded research protocols

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
