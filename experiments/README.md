# Research experiment records

These directories retain code whose assumptions belong to a particular research
question. They are not an archive of abandoned scripts. Shared implementation
belongs in `src/`, and maintained commands are indexed in
[`scripts/README.md`](../scripts/README.md).

| Experiment | Question |
| --- | --- |
| [mace_topology_nuances_20260907](mace_topology_nuances_20260907/README.md) | Target stability and frozen-decoder diagnostics, then topology-aware attraction and distance preservation in 80-atom MACE |
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
