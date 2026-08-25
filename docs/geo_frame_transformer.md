# GeoFrameTransformer

`GeoFrameTransformer` is the geometry-aware encoder used by
`configs/vicreg_geo_frame_multi.yaml`. It is a registered architecture, not an
alias or in-place change to `RI_MAE_Invariant`. The old RI-MAE encoder and its
checkpoint keys remain unchanged for controlled ablations.

The name describes the inference architecture: a transformer over local geometric
frames. Normal encoding has no masked input and exports a compact 64-dimensional
environment embedding.

## Changes from RI-MAE-Invariant

1. **Pairwise patch geometry.** Every attention layer receives a learned bias from
   radial-basis distance features, displacement directions expressed in both local
   frames, relative frame orientation, and frame confidence.
2. **Multi-scale patches.** Optional multi-scale patches share centers
   but have separate point encoders before learned scale fusion.
3. **Frame-confidence gating.** Eigenvalue gaps estimate whether a local triad is
   well determined. The canonical branch retains a configurable floor so symmetric
   patches do not lose their angular structure. Its fallback is a learned invariant
   encoder over point radii, within-patch distances, centroid contractions, and
   covariance energy rather than a small radial-statistics vector.
4. **Compact attentive pooling.** Learned multi-query pooling replaces max-plus-mean
   pooling and projects directly to a compact exported representation.

## Important configuration fields

| Field | Default in new config | Meaning |
|---|---:|---|
| `encoder.name` | `GeoFrameTransformer` | Select the new architecture. |
| `encoder.kwargs.patch_sizes` | `[12]` | Neighbor counts for the configured patch scales. |
| `encoder.kwargs.trans_dim` | `64` | Patch-token width. |
| `encoder.kwargs.latent_size` | `64` | Exported environment dimension. |

The reference configuration uses `batch_size: 8192`, mixed BF16 precision, and
compiled encoder execution. If memory is fragmented or another process shares the
GPU, reduce `batch_size` first.

## VICReg pretraining

Use `configs/vicreg_geo_frame_multi.yaml` to pretrain the exported GeoFrame latent.
This configuration deliberately applies VICReg directly to the 64-dimensional
encoder output (`vicreg_projector_mode: identity`). A separate nonlinear projector
must not be used for this stage: it can satisfy VICReg while the representation
consumed by clustering collapses.

The pretraining configuration also uses deterministic FPS in train and evaluation,
same-environment jittered views, FP32 master weights with mixed-BF16 compute, and
checkpoints minimum direct validation loss. Silhouette remains a diagnostic because
a high silhouette can be produced by a degenerate one-dimensional source split.

Run it with:

```bash
conda run -n pointnet python src/training_methods/contrastive_learning/train_contrastive.py \
  --config-name vicreg_geo_frame_multi
```
