# MACE depth and measured parameter counts

The default for new native spatial MACE models is now **128 channels with a
128-dimensional export** (user instruction,25 September2026). Use
[`default128.json`](../../configs/supervised_onset/default128.json); the explicit
small/500k/1M/2M recipes remain capacity ablations.

| Default width128 component | Trainable parameters |
| --- | ---: |
| Spatial backbone including species/center embeddings |552,320|
| Projected residual export |82,176|
| **Exported encoder total** |**634,496**|
| Embedding-only onset head |17,157|
| **Single-view training model total** |**651,653**|

Both e3nn and cuEquivariance constructors give these counts. The backbone retains
two spatial blocks with128 channels for each of l=0,1,2. Each atom therefore has
up to128×(1+3+5)=1,152 scalar components in its equivariant hidden state. Pooling
concatenates the128 central scalar features and128 regional scalar features,
giving256 inputs to the128-D projected residual export. Equal channel and export
widths do not remove that projection or imply an automatic speed improvement.
No width128 performance benchmark or scientific training was run for this change.
Six existing encoder regression checks and a local cuEquivariance CUDA forward/
backward check passed; the latter used the constructor defaults and created no
W&B run. [Count and CUDA receipt](../../output/encoder_research/default-width128-20260925/technical/check.json).

## Historical comparisons

This is the measurement of the completed24 September architecture. The subsequent
[capacity experiment](../../experiments/supervised_capacity_20260925/README.md)
uses79,232 /503,552 /1,029,056 /1,981,184 encoder parameters and a revised export.
The historical counts below remain unchanged.

[Handbook](README.md) · [Prediction context](prediction_context.md) ·
[Checkpoint measurements](../../output/encoder_research/prediction-context-20260925/technical/mace-parameter-counts.json)

Measured25 September2026 by summing `numel()` over unique `nn.Parameter` objects.
Buffers, optimizer states and cached radial/angular features are excluded.
The historical supervised models were loaded strictly from their actual saved
weights. This comparison refers to our latest supervised GeometryEncoder, not
every earlier model in the repository named MACE.

| Model/component | Spatial message-passing layers | Parameters |
| --- | ---: | ---: |
| Local MACE-MP-0b2 **small** energy/force MLIP checkpoint |2|8,221,984|
| Our current single-view exported encoder |2|62,784|
| Our encoder plus historical temperature/age hazard head |2|80,837|
| Our encoder plus new embedding-only hazard head |2|79,941|
| Paired-view historical full model |2 per view, shared weights|179,461|
| Paired-view new full model |2 per view, shared weights|178,565|

The paired exported encoder includes98,624 fusion parameters in addition to the
62,784 shared geometry parameters:161,408 total for embedding inference. The
historical observed-input distilled model has97,349 parameters including its
training-only16,512-parameter teacher projection; the new version has96,453.
Exporting its embedding needs only the62,784-parameter geometry encoder.

| Architecture setting | Cached MACE-MP-0b2 small | Latest geometry encoder |
| --- | --- | --- |
| Channels |128|32|
| Hidden irreps in constructor |128x0e|32x0e +32x1o +32x2e|
| Angular expansion `max_ell` |3|2|
| Correlation order |3|2|
| Radial MLP widths |64,64,64|32|
| Atomic species supported |89|1: Al|
| Edge cutoff |5 A|5 A|
| Prediction/output |Atomic energy contributions; forces by differentiation|128-dimensional local state|

Correlation order3 combines up to three neighbor factors within a message
(four-body terms including the central atom); order2 combines up to two
(three-body terms). Stacking two interaction blocks expands spatial dependence;
it is distinct from this many-body order and from the radial MLP depth.

Our62,784 parameters decompose into39,296 interaction parameters,6,848 product
parameters,16,576 export-readout parameters,32 species-embedding parameters and
32 center-embedding parameters. Removing the export readout leaves46,208.
Although the first block uses vectors and rank-two tensors, this encoder's
final pooling uses scalar channels; the tensor-pooling screen was another variant.

Thus equal depth does not mean equal capacity: this MLIP has about131 times the
parameters of our exported single-view encoder. It also serves a different
objective and89 elements. It is not evidence that making our encoder131 times
larger would improve crystallization AP.

The checkpoint is `mace_mp_0b2_small.model`, SHA256
`d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9`;
the machine receipt stores its full local path. Counts vary by MACE release and
size. The general two-layer foundation-model architecture is described in the
[MACE-MP paper](https://arxiv.org/abs/2401.00096); the exact checkpoint belongs to
the [0b2 release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mp_0b2).
The numerical counts and5 A cutoff above come from the local checkpoint, not
from substituting specifications of another release.

For the serialized MLIP, the measurement can be reproduced in the project conda
environment using `torch.load(path, map_location='cpu', weights_only=False)`,
`len(model.interactions)` and `sum(p.numel() for p in model.parameters())`.
For our trained model instantiate its recorded `encoder_config`, strictly load
the matching historical source/checkpoint, and count `model.encoder.parameters()`.
