# Temporal MACE encoder

See the [encoder, TDA and relaxation problem review](encoder_tda_relaxation_problems_20260910.md)
for scientific limitations, earlier failures, completed corrections and current
evidence. The interfaces below specify implementation behavior, not demonstrated
physical accuracy.

The September 10 denoising version is `PretrainedMACEDenoising`, constructed with
`configs/mace_denoising_encoder.yaml`. It keeps MACE frozen and offers `residual`,
`atom_temporal`, and `atom_anchor` fusion. The first learns a correction to the
256-dimensional anchor features; the atom variants preserve identities until
after temporal attention and then learn spatial pooling. `atom_anchor` repeats
current atom features inside the same network as a matched capacity control.
All return `(B, 256)` and require `(B, T, 80, 3)` physical histories and material
indices. Atom permutations must be shared across frames so identity alignment
is preserved. The denoising backbone runs without coordinate gradients.

Its scalar atom features are quantized to float16 before normalization to match
the training cache; pooled anchor features remain float32. The fitted
`pooled_mean/std` and `node_mean/std` are encoder buffers. Encoder exports include
the registered name, constructor arguments, complete encoder state, topology
decoder and target scaling. The current experiment uses Al only and real 0.75 ps
sampling. See the [independent-source experiment](../experiments/mace_al_denoising_20260910/README.md)
for training, controls and held-source analysis. The original jointly trained
version is documented below and retains its own checkpoint format.

`PretrainedMACETemporalEncoder` in `src/models/encoders/mace_temporal.py` learns
one invariant structural embedding from several instantaneous neighborhoods.
A shared physical MACE produces 256 scalar features per frame. A projection and
learned encoding of relative time feed independently initialized transformer
blocks. The updated final-frame token is the output; there is no temporal mean
pooling. MACE still performs its existing spatial mean pooling over 80 atoms.

## Construction and input

In the `pointnet` environment, from the repository root:

```python
import torch
from omegaconf import OmegaConf
from src.models.encoders import build_encoder

cfg = OmegaConf.load("configs/mace_temporal_encoder.yaml")
encoder = build_encoder(cfg).cuda()
# histories: float32 CUDA tensor (B, 5, 80, 3), coordinates in Å.
# material: int64 CUDA tensor (B,), 0=Al, 1=Mg, 2=Ta.
z = encoder(histories, material)  # (B, 128)
tda_head = torch.nn.Sequential(
    torch.nn.Linear(encoder.invariant_dim, 256),
    torch.nn.SiLU(),
    torch.nn.Linear(256, 32),
).cuda()
loss = (tda_head(z) - relaxed_anchor_tda_pca).square().mean()
loss.backward()  # Trains both the transformer and the shared MACE.
```

The example assumes an allocated CUDA device and prepared histories/targets;
it does not prepare data or launch training. To construct on CPU, set
`cfg.encoder.kwargs.accelerated = False` and use CPU tensors/modules.
The configuration is an encoder specification, not a complete training recipe.
The registered constructor uses the physical `forward(points, material)` API,
like `PretrainedMACEEncoder`. The normalized single-cloud `EncoderAdapter`,
single-frame VICReg trainer and static analysis pipeline do not accept histories.

Time offsets are explicitly configured in ps, strictly increasing and ending at
zero. The default five frames span 0.4 ps at the ordinary continuation sources'
0.1 ps cadence. Change the offsets to change the physical window; `time_scale_ps`
only sets the units seen by the time MLP. Both are checkpoint buffers. A
one-frame `[0.0]` configuration is supported as an explicit ablation. All tokens
may attend to all observations in the supplied history; only the final anchor
representation is exported. This does not use future observations.

Use `frame_batch_size` to limit each MACE forward's number of clouds. This keeps
end-to-end gradients, so it does not bound total retained training activations.
The nested `mace.feature_mean` and `mace.feature_std` buffers initially preserve
raw features (zero/one); any fitted standardization must use training data only.
Save the full encoder state and configuration together; the original pretrained
MACE model is still required when reconstructing the constructor before loading.

## Data and training semantics

The current `thermal80` cache contains hot anchor/spatial/temporal/future views
and their relaxed partners. Its view dimension is **not** a history dimension.
Temporal training must assemble past frames of the same central atom from
`TemporalLAMMPSBinaryTrajectory`, in the configured time order, and decode stored
float16 coordinates to float32 before geometry calculations.

Select the anchor's hot 80 atom identities and follow those identities through
the history. Center each frame on the same central atom and handle periodic
images using that frame's box. Keep histories inside their assigned source/time
split. The relaxed label must describe those same anchor-selected identities,
as in `mace_relaxed.paired_clouds`. Do not feed relaxed partners or prediction
future frames into the encoder. The encoder cannot infer identity, cadence or
split membership from coordinates; the history producer owns these semantics.

Supervise only the combined embedding against the relaxed anchor's TDA, with
train-only target PCA/scaling, and apply any variance/covariance regularization
to that combined embedding. The encoder contains no prediction head or loss.
The existing per-view thermal/VICReg objective is separate from the maintained
`temporal80` training protocol in `src/training_methods/mace_temporal.py`.
That protocol builds identity-matched histories with `src/data_utils/mace_history.py`,
trains on relaxed-anchor TDA with variance/covariance regularization, and analyzes
real held-out histories. See the [training experiment](../experiments/mace_temporal_transformer_20260909/README.md)
for its config and the existing MACE family command. Selecting the encoder
constructor alone does not change a single-frame trainer's scientific protocol.

## Verification and file roles

`tests/test_mace_temporal.py` covers end-to-end gradients through actual MACE,
dependence on historical frames and physical time, rigid-motion/permutation
invariance, chunking equivalence and checkpoint restoration. Run:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  conda run -n pointnet python -m pytest tests/test_mace_temporal.py -q
```

The encoder, construction config, documentation and tests are maintained
components. This adds no experiment record, runner or generated dataset.
## Mixed-cadence available-data evaluation

The available Al comparison passes actual `(B, T)` frame offsets to temporal
attention, scaled by one common time unit. Frame and atom fusion support
per-history times; atom fusion repeats each history's times for its own tracked
atoms before spatial pooling. Exported mixed-cadence `PretrainedMACEDenoising`
encoders set `require_frame_offsets=True` and accept
`encoder(points, material, frame_offsets_ps)`. Fixed-cadence encoders retain
their configured default times. See the
[available-data experiment](../experiments/mace_al_denoising_20260910/README.md).
