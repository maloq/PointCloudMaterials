# Implementations of recorded research protocols

`trajectory_stability.audit` supplements existing current-screen and dense
trajectory exports with source-balanced spectral dimensions and physical-lag
stability. [Guide](../../docs/encoder_research/embedding_dynamics.md),
[definitions](../../docs/metrics/embedding_dynamics.md).
`trajectory_stability.native_dense` reuses pinned native producers to add
Geoformer and current MACE to the common 0.75 ps observed-trajectory table.
`trajectory_stability.noise` measures controlled coordinate-noise responses on
matched origins, including the numerical floor and ratio to natural 0.75 ps motion;
[guide](../../docs/encoder_research/input_noise.md).

`geoframe_evolution` retains every epoch of the original GeoFrameV2 recipe and
measures independent physical context, liquid order, continuity and future
prediction; [protocol](../../experiments/geoframe_evolution_20260923/README.md),
[commands](../../docs/geoframe_evolution.md).

`gatr_conditional_information.comparison` repeats the conditional-information and
trajectory-stability assays for pinned native MACE/GATr checkpoints, with exact
training-code extraction and symmetric radial-only controls. See the
[latest comparison](../../experiments/gatr_conditional_information_20260918/LOCAL_LAST622.md).

`backbone_tda/` prepares instantaneous nearest-80 persistence images and matches
frozen physical MACE/GATr readouts on the native source splits. It is separate
from active encoder source identities. See [workflow](../../docs/backbone_tda.md).

`local_predictability.backbone_v2` owns the fresh common physical/onset trainer
for accelerated native MACE and upstream axial GATr. It preserves v1 and uses
immutable packet joins, new fitting receipts and explicit workload profiling.
See the [v2 protocol](../../experiments/local_predictability_20260917/BACKBONE_V2.md).

`local_predictability.plan` validates/materializes the planned single-seed H100/H200
queue, including paired controls and confirmation padding. It does not implement
or launch the new training/assay adapters. See the
[protocol](../../experiments/local_predictability_20260917/README.md).

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
| `gatr_conditional_information` | [Information beyond radial structure](../../experiments/gatr_conditional_information_20260918/README.md): radial-only counterfactual, source-held-out bond/angular and prospective probes, redundant-input controls and dense matched environments |
| `mace_velocity` | [Local coordinate/velocity encoder](../../experiments/mace_velocity_20260915/README.md): smooth structure, time-even activity and time-odd motion, source-isolated holdouts and velocity interventions; no forecasting objective |
| `mace_local_state/physics.py` | Shared local-group physical observables used by native encoder training; frozen-map workflows [discarded](../../docs/discarded_frozen_encoder_maps.md) |
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

Pure motion calculations and verified trajectory IO used by native encoder
training live in `mace_velocity/motion.py` and `mace_velocity/sequence_data.py`.

The `mace_velocity` entry point also forwards `causal-prepare` and `causal-train`
to `src/training_methods/mace_causal/`. This distinct native tensor architecture
interleaves atom-level spatial/temporal messages before its one pooling stage.

`mace_velocity/data_amount*.py` implements the distinct native end-to-end encoder learning curve; use existing module stages `data-prepare`, `data-smoke`, `data-study`. See [protocol](../../experiments/mace_data_amount_20260916/README.md).

`mace_causal_comparison.py` collects completed native causal-state ablations and
frozen probes, verifies physical target pairing, and exports whole-source paired
intervals and plots. Recipe: `configs/mace_causal/comparison.json`.

`memory_report.py` freezes completed causal-state and partial-observation results
without pooling their distinct metrics. It verifies matched seed completeness,
preserves user-reported H200 evidence separately, and exports source hashes,
metric definitions, tables and figures. Recipe: `configs/analysis/memory_research_report.json`.

Causal runtime benchmarking uses the existing `mace_velocity causal-benchmark`
dispatch to `training_methods/mace_causal/benchmark.py`; packing is implemented in
`models/encoders/mace_causal_batch.py`, with residency and batching in `runtime.py`.

`local_predictability/observability.py` builds all-state packet recognition and
at-risk true-future sequence datasets, reusing the fixed descriptor trainer.
It saves predictions without scientific analysis. See the
[observability protocol](../../experiments/local_predictability_20260917/OBSERVABILITY.md).

The adjacent `native_queue.py` partitions existing onset stages across workers
without changing checkpoint identities. `raw_observability.py` trains a separate
all-state binary current-label model; `native_readouts.py` uses frozen onset states
for fresh linear/MLP readouts. Both retain predictions for deferred interpretation.

`crystallization_transfer` owns the expanded local MACE onset comparison: immutable graph/feature preparation, frozen/fine-tuned/scratch hazard fits, tensor-aware context and source-held-out event/spatial evaluation.

`crystallization_transfer.report` collects completed initial/scaling queues,
verifies paired test identities, and exports source-bootstrap comparisons and
scientific scaling plots without training new models.

The crystallization `adaptive`, `attention` and `refinement` modules implement
full-batch trainable normalization, controlled spatial/temporal context heads and
selection-only promotion to longer runs, through the existing transfer queue.

`crystallization_paths/` compares direct, autoregressive, mixture and diffusion local structural-state trajectories, reusing the crystallization-transfer source split and frozen MACE. [Protocol](../../experiments/crystallization_transfer_20260919/PATHS.md).

The path study's `diagnose`, `refined_model` and `refinement` modules implement selection-only failure diagnosis, revised trajectory models and physical-error-constrained promotion through the existing path queue.
# Fixed structural-state screen

`structural_state/` implements the distinct four-arm fixed-target study after BCR:
verified paired-cache reuse, resident graph geometry, native cuEquivariance MACE,
code-only heads, relaxed-target and relational objectives, source-held-out probes,
and detached local/independent Slurm queues. Current v2 repairs amplitude collapse
with fixed distance scales, fitting-only calibration and a direct pooled export;
it also audits actual training heads. See [protocol](../../experiments/structural_state_20260923/README.md)
and [operations](../../docs/structural_state.md).

`structural_state_onset_review.py` audits the completed v2 onset predictions and
bootstraps complete sources for paired AP and probability errors without fitting;
see [results](../../output/structural_state/repaired-review-20260923/README.md).
