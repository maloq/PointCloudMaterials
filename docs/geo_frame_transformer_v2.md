# GeoFrameTransformerV2

`GeoFrameTransformerV2` is a separate registered VICReg encoder. It does not
change `GeoFrameTransformer` or reuse V1 checkpoints implicitly.

## Geometric representation

Each ordered group edge `i -> j` contains:

- a Gaussian radial basis of the normalized center distance;
- `c_j - c_i` in frame `i` and `c_i - c_j` in frame `j`;
- the full relative rotation `F_i.T @ F_j`;
- normalized covariance eigenvalues, both eigengaps, local scale, normalized
  density, and frame confidence at both endpoints;
- an optional signed radial-moment triple product that changes sign under
  reflection.

One shared edge embedding produces a query-specific per-head attention bias.
For the value path, incoming pair geometry is summarized at every destination
token and produces a per-layer value gate plus a low-rank additive geometric
value. Attention itself uses fused scaled-dot-product attention.

## Parity modes

`parity_mode: sensitive` is invariant to translations and proper rotations but
retains parity-odd information. This is the default experiment because it is the
richer representation and the configured mirrored VICReg views can learn the
reflection behavior required by the material data.

`parity_mode: invariant` provides exact O(3) invariance. Canonical patch tokens,
local-center position tokens, and oriented edge geometry are evaluated for both
local parity hypotheses with shared weights and averaged. Use
`configs/vicreg_geo_frame_transformer_v2_o3.yaml` for this ablation.

If chirality is a target signal, use `parity_mode: sensitive` and set
`vicreg_mirror_prob: 0.0`; otherwise the positive-pair objective explicitly
trains mirror-related environments toward the same representation.

## Configurations

- `configs/vicreg_geo_frame_transformer_v2.yaml`: parity-sensitive V2, 24
  groups, K=8/16, D=128, six layers, edge dimension 16, value rank 2.
- `configs/vicreg_geo_frame_transformer_v2_o3.yaml`: exact reflection-invariant
  override with otherwise identical settings.

Both configurations train from scratch with global VICReg. Local VICReg is a
separate future ablation and is intentionally not mixed into the V2 architecture
comparison.

## Matched H100 compute benchmark

At batch 8192 with two stochastic views, BF16 fullgraph compilation, VICReg
projector/loss, backward, gradient clipping, and AdamW update:

| Encoder | Step | Peak allocation |
|---|---:|---:|
| Optimized V1 | 122.13 ms | 25.13 GiB |
| V2 parity-sensitive | 114.09 ms | 22.85 GiB |
| V2 exact O(3) | 145.15 ms | 30.83 GiB |

The parity-sensitive configuration is 1.071x faster than optimized V1 and uses
2.28 GiB less peak memory. Exact O(3) symmetrization costs 1.19x step time over
V1. These are compute-step measurements; data loading, logging, validation,
checkpoint I/O, and full-training representation quality are outside their
scope.

On 256 real cached material environments, an arbitrary rotation plus translation
gave global relative feature drift `1.91e-3` for V1, `1.76e-3` for the default
V2, and `4.61e-4` for exact-O(3) V2. Mean cosine similarities were
`0.9999982`, `0.9999985`, and `0.9999999`, respectively. The remaining drift is
caused by floating-point tie changes in discrete FPS/k-NN/frame selection on
highly symmetric neighborhoods; the continuous V2 geometry tests are invariant
within their numerical tolerance.
