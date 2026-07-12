# GeoFrameTransformer

`GeoFrameTransformer` is the geometry-aware encoder used by
`configs/line_jepa_geo_frame.yaml`. It is a new registered architecture, not an
alias or in-place change to `RI_MAE_Invariant`. The old RI-MAE encoder and its
checkpoint keys remain unchanged for controlled ablations.

The name describes the inference architecture: a transformer over local geometric
frames. Normal encoding has no masked input and exports a compact 256-dimensional
environment embedding. Masking is used only by an optional training objective.

## Changes from RI-MAE-Invariant

1. **Pairwise patch geometry.** Every attention layer receives a learned bias from
   radial-basis distance features, displacement directions expressed in both local
   frames, relative frame orientation, and frame confidence.
2. **Multi-scale patches.** The default 12- and 24-neighbor patches share centers
   but have separate point encoders before learned scale fusion.
3. **Frame-confidence gating.** Eigenvalue gaps estimate whether a local triad is
   well determined. The canonical branch retains a configurable floor so symmetric
   patches do not lose their angular structure. Its fallback is a learned invariant
   encoder over point radii, within-patch distances, centroid contractions, and
   covariance energy rather than a small radial-statistics vector.
4. **Ray-conditioned patch attention.** Line-JEPA reuses cached patch tokens and
   attends to them with the endpoint ray expressed in each patch frame. This
   replaces the fixed eight-moment directional input in the new configuration.
5. **EMA token prediction.** A full-context frozen EMA token encoder supplies
   normalized targets for randomly hidden student patch tokens. The objective is
   real and contributes to training through `line_jepa_masked_token_coeff`.
6. **Compact attentive pooling.** Learned multi-query pooling replaces max-plus-mean
   pooling and projects directly to a 256-dimensional exported representation.

## Important configuration fields

| Field | Default in new config | Meaning |
|---|---:|---|
| `encoder.name` | `GeoFrameTransformer` | Select the new architecture. |
| `encoder.kwargs.patch_sizes` | `[12, 24]` | Neighbor counts for the two patch scales. |
| `encoder.kwargs.trans_dim` | `256` | Patch-token width. |
| `encoder.kwargs.latent_size` | `256` | Exported environment dimension. |
| `encoder.kwargs.ray_feature_dim` | `64` | Learned directional feature width. |
| `encoder.kwargs.mask_ratio` | `0.6` | Fraction of tokens hidden in the auxiliary objective. |
| `line_jepa_directional_feature_mode` | `encoder` | Use cached ray-conditioned patch attention. |
| `line_jepa_masked_token_coeff` | `0.5` | Weight of the EMA token objective. |
| `line_jepa_masked_token_samples` | `256` | Center environments used for that objective per batch. |

The reference H100 configuration uses `batch_size: 2400` and BF16. Encoder
compilation is disabled because TorchInductor's CUDA shape-padding benchmark in
the current environment mixes FP32 and BF16 operands before the first step. If
memory is fragmented or another process shares the GPU, reduce only `batch_size`
first; the masked-token subset has its own independent sample cap.

## VICReg pretraining

Use `configs/vicreg_geo_frame_multi.yaml` to pretrain the exported GeoFrame latent.
This configuration deliberately applies VICReg directly to the 256-dimensional
encoder output (`vicreg_projector_mode: identity`). A separate nonlinear projector
must not be used for this stage: it can satisfy VICReg while the representation
consumed by clustering and Line-JEPA collapses.

The pretraining configuration also uses deterministic FPS in train and evaluation,
same-environment jittered views, FP32 master weights with mixed-BF16 compute, and
checkpoints minimum direct validation loss. Silhouette remains a diagnostic because
a high silhouette can be produced by a degenerate one-dimensional source split.

Run it with:

```bash
conda run -n pointnet python src/training_methods/contrastive_learning/train_contrastive.py \
  --config-name line_jepa_geo_frame
```
