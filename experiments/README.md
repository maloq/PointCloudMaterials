# Research experiment records

Start with [the run dashboard](../output/registry/index.html) for results, progress,
configs, checkpoints and plots across training and simulations. The versioned
[ideas backlog](ideas.json) records questions and next actions; [registry.json](registry.json)
lists external storage roots. See [the run/retention guide](../docs/output_registry.md)
before launching or reorganizing runs.

These directories retain configurations and findings for individual research
questions. Retired MACE implementations and their dedicated diagnostics were
removed on 2026-09-09; completed records identify the source snapshots needed
to reproduce their original protocols. Shared implementation
belongs in `src/`, and maintained commands are indexed in
[`scripts/README.md`](../scripts/README.md).

| Experiment | Question |
| --- | --- |
| [mace_al_denoising_20260910](mace_al_denoising_20260910/README.md) | Al TDA loss/fusion comparisons, completed mixed-potential pilot, [force/structure/topology potential comparison](mace_al_denoising_20260910/POTENTIAL_DIFFERENCES.md) and detached preparation of independent sources |
| [mace_temporal_transformer_20260909](mace_temporal_transformer_20260909/README.md) | Joint MACE and five-frame temporal attention for relaxed-anchor topology, followed by held-out temporal and spatial analysis |
| [mace_original_vicreg_tda_20260909](mace_original_vicreg_tda_20260909/README.md) | Queued matched original VICReg + TDA from the exported latent, TDA80 from epoch 1, then the standard static analysis |
| [mace_original_vicreg_20260909](mace_original_vicreg_20260909/README.md) | Pretrained small MACE with normalized 80-point geometry, no element inputs, the original Lightning VICReg/projector, and standard static analysis |
| [mace_vicreg_audit_20260909](mace_vicreg_audit_20260909/README.md) | Numerical VICReg/gradient audit against the original module, and TDA-free optimization controls for raw versus learned MACE coordinates |
| [mace_thermal80_tda1_24ep_20260909](mace_thermal80_tda1_24ep_20260909/README.md) | Delayed fresh variant C run on the same paired data: TDA from epoch 1, 24 epochs, frozen probes and static Al analysis |
| [mace_thermal80_20260909](mace_thermal80_20260909/README.md) | Variant C pilot: shared hot/relaxed 80-atom views, full-cell fixed-box minimization, relaxed TDA targets, 12 epochs and standard static Al analysis |
| [mace_plain80_20260909](mace_plain80_20260909/README.md) | Current simplified MACE: full 80-atom pooling, uniform epochs, fixed spatial/temporal VICReg, TDA80 beginning in epoch six; 12 epochs then standard static Al analysis |
| [mace_bf16_training_20260908](mace_bf16_training_20260908/README.md) | Continue update-500 MACE checkpoint with compensated BF16, preserving Adam/LR/sampling position and the original 4224-update budget; interim loss/gradient audit and metric glossary |
| [mace_balanced_representation_20260908](mace_balanced_representation_20260908/README.md) | Fresh MLIP MACE, 12 epochs of encoder-gradient-balanced spatial/temporal VICReg and TDA; no nuisance or forecasting; frozen probes and standard static-Al analysis |
| [mace_bf16_20260908](mace_bf16_20260908/README.md) | Selective and compensated BF16 MACE precision/throughput screening; user requested immediate BF16 continuation after the successful screen |
| [mace_throughput_20260908](mace_throughput_20260908/README.md) | Matched throughput measurements for larger chunks, GPU data, geometry reuse and compiled/faster matrix operations; optimized detached restart |
| [mace_target_complete_20260908](mace_target_complete_20260908/README.md) | Correct TDA support with an adaptive 80-atom window and pooled MACE context; restarted in the throughput experiment after user-authorized optimization |
| [mace_joint_properties_20260908](mace_joint_properties_20260908/README.md) | Four matched MACE continuations with all-property screening; [interim review](../output/mace_joint_properties_20260908/INTERIM_REVIEW_20260908.md) finds a mismatch between compact encoder support and TDA targets; remaining jobs superseded by the corrected-support run |
| [mace_topology_nuances_20260907](mace_topology_nuances_20260907/README.md) | Target stability, frozen decoders and topology-aware MACE; [completed review](../output/mace_topology_nuances_20260907/REVIEW_20260908.md): improved topology geometry, reduced smoothness |
| [pretrained_mace_ablations_lr5_20260907](pretrained_mace_ablations_lr5_20260907/README.md) | Recover the MACE ablations from checkpoint quota failure using node-local optimizer checkpoints and 5× peak learning rates |
| [ti_ta_crystallization_20260907](ti_ta_crystallization_20260907/README.md) | Generate one pure-Ti Kavousi MEAM crystallization source, branch from six stages, and add five archived Ta branches |
| [pretrained_mace_ablations_20260907](pretrained_mace_ablations_20260907/README.md) | Matched fresh-MLIP loss ablations queued after the current MACE static analysis; frozen probes and full-frame spatial comparison |
| [pretrained_mace_80_dt01_cosine_20260906](pretrained_mace_80_dt01_cosine_20260906/README.md) | Continue 80-atom MACE with verified 0.1 ps Al/Mg/Ta temporal pairs and per-optimizer-step warmup/cosine decay |
| [pretrained_mace_spatiotemporal_80_20260906](pretrained_mace_spatiotemporal_80_20260906/README.md) | Strict 80-atom MLIP-initialized MACE with a smooth compact context, four training objectives and online W&B |
| [pretrained_mace_spatiotemporal_20260906](pretrained_mace_spatiotemporal_20260906/README.md) | Fine-tune actual small MLIP MACE weights with spatial/temporal VICReg, TDA and future-latent prediction; run the standard encoder-only static analysis |
| [temporal_hypotheses_12h_20260906](temporal_hypotheses_12h_20260906/README.md) | Larger temporal datasets, predictive objectives and wider MACE support without TDA/SOAP supervision; [completed review](../docs/temporal_hypotheses_review_20260906.md), plus full static-Al predictive-density analysis |
| [predictive_encoder_training_20260905](predictive_encoder_training_20260905/README.md) | How do matched encoders compare after geometry and shooting-outcome supervision, validation tuning and convergence checks? |
| [liquid_sro_benchmark_20260905](liquid_sro_benchmark_20260905/README.md) | Can proper MACE and alternative encoders resolve subtle liquid order and predict independent shooting futures beyond coarse structural descriptors? |
| [mace_diagnosis_20260905](mace_diagnosis_20260905/README.md) | Does the smooth pilot's weak static-Al transfer come from the MACE block, training, numerical/domain shift, or its readout? |
| [smooth_temporal_encoder_20260905](smooth_temporal_encoder_20260905/README.md) | Do smooth local density/MACE representations and transported temporal memory improve continuity, structure retention, and embedding forecasts? |
| [geoframe_continuity_20260905](geoframe_continuity_20260905/README.md) | Do grouping/frame switches cause embedding jumps, and does motion-based frame transport remove them? |
| [restart_audit_20260905](restart_audit_20260905/README.md) | Why does same-rank CSLD restart continuation diverge at the first step? |
| [spatiotemporal_20260905](spatiotemporal_20260905/README.md) | How do matched Al/Mg/Ta temporal views and VICReg/VISReg objectives affect stability and structural clustering? |
| [independent_sources_20260903](independent_sources_20260903/README.md) | Produce independent 510/520 K Al sources for subsequent shooting selection |
| [ta_source_20260905](ta_source_20260905/README.md) | Produce and verify the initial Ta branch used by the spatiotemporal dataset |
| [factor_vae_20260901](factor_vae_20260901/README.md) | Compare the fixed GeoFrame factor-VAE sweep and its controls |

Run recipes from the repository root with `conda run -n pointnet python ...`.
Supply actual output/data paths in place of the example placeholders. Generated
results belong in the run directory, not here. These relocations do not rerun the
experiments or change their scientific findings.

For a new question, add `<topic>_<YYYYMMDD>/README.md` with the question, exact
command, config, output location and findings. Add `analysis.py` only for unique
analysis. Prefer an existing command plus a config for another run of a method.
Existing Hydra configs remain under `configs/` because their composition and
checkpoint references depend on that configuration root; link them rather than
copying them into a new experiment folder.
