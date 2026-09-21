# Current research

Only scientific questions, protocols, findings and reproduction configurations
belong here. Simulation campaigns and inventories are in [docs/simulations](../docs/simulations/README.md);
storage, portability and maintenance are in [docs](../docs/data_storage.md).

The [September 11–13 retention review](../docs/research_retention.md) lists retained
results and necessary checkpoints. [Older research and outputs are on STORE](../docs/archived_research.md).
The dated folder name is the start date; older experiments below support current research.

| Research | Current evidence / purpose |
| --- | --- |
| [Latest MACE/GATr conditional information](gatr_conditional_information_20260918/LOCAL_LAST622.md) | Completed update-622 comparison: MACE retains substantial angular information; GATr remains mostly radial and is 24% smoother; no useful added onset prediction in the tested readouts. |
| [Latest local MACE and GATr static Al](shared_pretraining_20260918/STATIC_AL_LOCAL.md) | Final step-622 local bond-order encoders; matched six snapshots and seven-cluster analysis, with native trained local support. |
| [GATr information beyond radial structure](gatr_conditional_information_20260918/README.md) | Frozen z128: source-held-out bond/angular and prospective readouts, radial-only and duplicate-input controls, 609 strictly matched spatial pairs; little extra angular information and no robust extra forecasting benefit. |
| [GATr internal directional embeddings](gatr_equivariant_20260918/README.md) | Frozen v6 geometric streams on A100/node07: directional turns, cage correction, phase-conditioned spatial order, rotation controls and readout interventions; interactive 3D explorer. |
| [Encoder and descriptor trajectory stability](trajectory_stability_20260918/README.md) | Latest selected MACE/GATr versus instantaneous TDA, SOAP and local structural descriptors; matched full Al trajectories, training-reference normalization and source bootstrap. |
| [Twelve-epoch shared and causal pretraining](shared_pretraining_20260918/README.md) | **Paused after optimization failure.** Batch-512/LR-0.02 fits developed saturated heads and collapse; [diagnosis and retained information](../output/shared_pretraining/diagnosis-20260918/RESULTS.md). |
| [Shared structural pretraining](structural_pretraining_20260917/README.md) | Three implemented five-metal fits: MACE/GATr neighbor VICReg and three-frame GATr temporal JEPA/SIGReg, with physical/instantaneous-TDA anchors. Fixed material cutoffs; 250,000 training records, one seed; launched detached on H100/RTX6000. |
| [Newest GATr v6 static Al](shared_pretraining_20260918/STATIC_AL.md) | Selected Al-only v6 checkpoint; same six snapshots, interior grid and seven-cluster static analysis. |
| [Active mixed GATr static Al](shared_pretraining_20260918/STATIC_AL_BACKTRACKING.md) | Frozen latest update 400 from the temporal-backtracking continuation; matched six-snapshot static analysis. |
| [Newest MACE v6 static Al](shared_pretraining_20260918/STATIC_AL_MACE.md) | Selected Al-only v6 MACE checkpoint on the same six snapshots, interior grid and analysis settings as GATr. |
| [GATr–VICReg static Al](structural_pretraining_20260917/STATIC_AL.md) | Selected RTX6000 checkpoint, native full-neighborhood z128, six standard Al snapshots and the full seven-cluster static workflow. |
| [Backbone TDA retention](local_predictability_20260917/TDA.md) | Frozen physical MACE/GATr snapshot states; matched linear/nonlinear instantaneous topology readouts, one seed |
| [Native backbone v2](local_predictability_20260917/BACKBONE_V2.md) | Fresh gated MACE/cuEquivariance versus axial GATr snapshot screen; all-state physical targets; one seed |
| [Local predictability](local_predictability_20260917/README.md) | Completed local descriptor, native onset and observability studies; [all available results](../output/local_predictability/research-summary-20260917/RESULTS.md), including physical/TDA screens; H200 training reported complete, validation nearly tied, test comparisons pending |
| [Predictive memory](predictive_memory_20260917/README.md) | Completed H100/H200 pilot; objective follow-up stopped after 10/16 fits. [Latest update](../output/predictive_memory/research-summary-20260917-stopped/RESULTS.md) and [consolidated results](../output/predictive_memory/research-summary-20260917/RESULTS.md) |
| [Causal native MACE](mace_causal_20260916/README.md) | Completed short/long-budget studies and H200 width comparison; small conditional history benefit. [Consolidated results](../output/predictive_memory/research-summary-20260917/RESULTS.md) |
| [Direct local-state smoothness](mace_local_smooth_20260915/README.md) | **Discarded** frozen-map approach; first sweep failed the retention gate; capacity fits retained without an established final evaluation |
| [Frozen local-group states](mace_local_state_20260915/README.md) | **Discarded** approach; [completed comparison](mace_local_state_20260915/RESULTS.md): learned physical distance improves smoothness/group information but loses instantaneous topology; tested density discovery does not resolve liquid states |
| [MACE liquid-cluster diagnosis](mace_context_clusters_20260915/README.md) | Matched GeoFrame V2 comparison, eight saved-feature ablations, physical readouts, and a [literature review](mace_context_clusters_20260915/LITERATURE_REVIEW.md) on coherent liquid representations |
| [Static Al/Zr joint MACE](mace_context_static_20260915/README.md) | Full standard static analysis with exact shared inner/center checkpoint weights and complete message context; Zr geometry transfer |
| [VICReg / TDA ridge audit](mace_tda_ridge_audit_20260914/README.md) | Fresh six-checkpoint inference, independent ridge reproduction, supervision checks, projector-versus-encoder comparison, frozen MLIP/random controls, and direct embedding geometry |
| [Forecast MACE diagnostics](mace_encoder_diagnostics_20260914/README.md) | Exact target encoder: invariance, geometry/membership changes, storage precision, physical time dependence, siblings and source-held-out TDA readouts |
| [Embedding forecast](embedding_forecast_20260911/README.md) | Direct versus autoregressive trajectories, completed pilot and enlarged fits; exact continuation and target-encoder provenance |
| [Observed history](forecast_context_20260913/README.md) | [24 completed fits](forecast_context_20260913/RESULTS.md), matched 0–24 ps history comparison |
| [Local crystallization](forecast_crystallization_20260913/README.md) | [Completed physical assay](forecast_crystallization_20260913/RESULTS.md), transition/timing readouts and [next questions](forecast_crystallization_20260913/NEXT_EXPERIMENTS.md) |
| [Spatial context and mixtures](forecast_spatial_mixture_20260913/README.md) | [Completed results](forecast_spatial_mixture_20260913/SPATIAL_CONTEXT_RESULTS.md): 12 ps history / 32 neighbors gives 56.86% transition F1; original, short-history and larger-context sweeps completed |
| [Learned spatial attention](forecast_spatial_attention_20260914/README.md) | Individual neighbor embeddings and relative geometry; matched 32-center attention versus mean pooling at 3/12 ps |
| [Relaxed MACE / VICReg](mace_vicreg_relaxed_20260910/README.md) | [September 11 comparison](mace_vicreg_relaxed_20260910/RESULTS_20260911.md); supplies the frozen target encoder for every forecast |
| [Full-data MACE / VICReg](mace_vicreg_full_20260910/README.md) | September 11 analysis and retained full-data reference checkpoint |
| [Al topology denoising](mace_al_denoising_20260910/README.md) | Older dependency: selected independent sources, relaxation targets, frozen MACE model and controlled denoising results |

The [ideas backlog](ideas.json) retains research questions. The file-backed
[run dashboard](../output/registry/index.html) indexes results; its root settings
live in [configs/experiment_registry.json](../configs/experiment_registry.json).
See [registry documentation](../docs/output_registry.md) and [result conventions](../docs/research_layout.md).
Scientific implementation is in [src/research](../src/research/README.md) and the
maintained [training/analysis commands](../scripts/README.md).

- [Complete context and center-node MACE pilot](mace_context_20260914/README.md): controlled membership crossings, held-out TDA readouts and matched VICReg continuation.
- [Recovering local structure in continuous MACE embeddings](mace_context_recovery_20260914/README.md): nonlinear readouts, combined center/inner features and matched training with physical supervision.
# Local coordinate/velocity state

[mace_velocity_20260915](mace_velocity_20260915/README.md) preserves the smooth
MACE structural representation and learns separate activity and directed-motion
channels from measured velocities, with an explicit coordinates-only ablation.
Both variants completed; structural stability remains near the original teacher.
The [smooth-manifold literature review](mace_velocity_20260915/LITERATURE_REVIEW_SMOOTH_MANIFOLD.md)
proposes direct temporal objectives, local motion constraints, and short observed
history while testing preservation of bond order and instantaneous topology.

[Consecutive local-state motion](mace_local_motion_20260916/README.md): **Discarded** frozen-map approach. All 44 fits and evaluation completed; none passed the information gate or joint 0.10 jump requirement. Results retained.

- [Native encoder training-data amount](mace_data_amount_20260916/README.md): matched-update independent-source learning curves with motion constraints.

- [Expanded MACE crystallization transfer](crystallization_transfer_20260919/README.md): frozen, fine-tuned, scratch and tensor-context onset queue on 362,400 candidate windows.

- [Recent local encoder and crystallization report](../output/crystallization_transfer/recent-report-20260919/README.md): pretraining recipe/provenance, context/data scaling, corrected encoder screens and completed trajectory refinements; [pending final update](crystallization_transfer_20260919/REPORT_UPDATE.md).

- [Equivariant neighborhood JEPA](neighborhood_jepa_20260920/README.md): independently encoded local MACE snapshots trained by structured neighbor/time prediction, SIGReg and fixed physical/TDA anchors.

- [Causal geometry-anchored neighborhood JEPA v2](neighborhood_jepa_v2_20260920/README.md)

- [Neighborhood JEPA regularizer/order comparison](neighborhood_jepa_regularization_20260920/README.md): SIGReg, VICReg and EpiJEPA-inspired geometric regularization.

- [Multi-horizon JEPA embedding prediction](neighborhood_jepa_multihorizon_20260920/README.md): tracked future local states at 3, 6 and 9 ps.

- [Frozen short-horizon information diagnostic](crystallization_information_20260920/README.md): feature add-backs and decoding, up to12ps.

- [Relaxed inputs and targets MACE pilot](relaxed_encoder_20260920/README.md): paired hot/hot, hot/cold, cold/cold and <=12ps onset readouts.

- [Overnight encoder information and trajectory context](context_night_20260921/README.md): local order/angular preservation, wider shells/history, and fixed-target trajectory forecasts.

- [Expanded relaxed-input/target and rank study](relaxed_encoder_expanded_20260921/README.md): larger source-held-out paired dataset, conditional SIGReg/VICReg, frozen onset probes.
- [Larger fixed-grid crystallization test](relaxed_encoder_large_test_20260921/README.md): 338 planned distinct local test onsets, historical encoder comparisons and whole-source uncertainty.

- [BCR-v1: bottleneck-conditioned reconstruction](bcr_v1_20260921/README.md): implementation, mechanism controls and correctness tests.

- [Independent-root BCR G1 pilot](bcr_g1_20260921/README.md): useful conditioning, matched controls and frozen liquid probes.

- [Symmetric structured MACE/GATr context](structured_context_20260921/README.md): 25 spatial query slots, alternating spatial/causal temporal attention, eight frozen-encoder forecast fits.
