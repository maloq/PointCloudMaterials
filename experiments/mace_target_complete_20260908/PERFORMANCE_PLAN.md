# Model and training performance proposal — 2026-09-08

This is a proposal based on the running corrected-context experiment. No running
configuration, process or queue was changed while preparing it. New performance
measurements should run after the current training and analysis finish.

## Observations from the implementation and live run

The H100 NVL has 95,830 MiB available. One inspection sampled 23,087 MiB used,
75% GPU utilization and 381.89 W power draw. These are instantaneous readings,
not a profile identifying the bottleneck. Between the first 22 logged updates,
the median time per update was 2.65 s, measured over ten-update intervals.
The effective batch is 1,536 quadruplets; graph microbatches contain 512 clouds.
Each ordinary update encodes four original views plus one perturbed anchor,
then recomputes the encoder while replaying cached representation gradients.

The implementation already uses cuEquivariance, fused AdamW, pinned transfers,
CPU prefetch and exact whole-batch VICReg gradient caching. Versions inspected:
PyTorch 2.11.0+cu128, mace-torch 0.3.16, cuequivariance-torch 0.10.0.

The checkpoint's Bessel basis, Agnesi transform and radial cutoff have no
trainable parameters. Distances, edge indices, radial features, spherical
harmonics and support weights can therefore be reused between the two encoder
passes for the same coordinates and augmentation. Learned node/message features
must still be recomputed as required by backpropagation.

The current PCA keeps 32 components and reports 99.9995% aggregate target
variance retained. This does not establish retention of every subtle physical
feature, but argues against treating a larger PCA dimension as the first fix.
The previous larger frozen decoder scored worse: 0.6336 versus 0.6234 MSE.

## Throughput changes, in order

| Priority | Change | Why it is concrete here | Verification |
|---|---|---|---|
| 1 | Benchmark graph chunks 512, 1,024 and 1,536 | Considerable memory headroom; larger chunks reduce Python iterations and kernel launches | Same checkpoint, batch indices, augmentation and effective batch; compare gradients and time |
| 2 | Pack the active data into GPU memory | Training coordinates at 80 atoms occupy 0.513 GiB in stored float16; projected float32 TDA targets add 0.137 GiB | Preserve sampling order, original target membership, FP32 input conversion and fixed train-only scalers |
| 3 | Reuse geometry between gradient-cache passes | The same geometry is currently rebuilt twice; its radial basis is fixed | Compare edge indices, descriptors, loss and backbone gradients, including perturbed views |
| 4 | Test faster dense matrix arithmetic | Current training requests full FP32 internal matrix precision | Compare numerical error with the small structural changes we want to resolve |
| 5 | Compile selected stable blocks after profiling | Dynamic graph construction and scalar synchronizations impede a single compiled graph | Inspect graph breaks and recompilations, and include compilation warmup in amortized timing |

Training and validation coordinates plus projected targets need roughly 0.7 GiB
before metadata, so the active dataset fits comfortably on this GPU. This avoids
repeated Python row gathering, target projection and host-to-device transfers.
It does not require another disk cache or loading all 512 stored neighbors.

The geometry cache should initially live for one optimizer update. Caching the
entire dataset's edge features would consume much more memory. A new coordinate
augmentation requires its own geometry. Never reuse learned encoder features
across optimizer updates when the backbone is being trained.

The current use of `nonzero` on CUDA causes host-device synchronization, as
documented by [PyTorch](https://docs.pytorch.org/docs/main/generated/torch.nonzero.html).
Geometry reuse also avoids repeating that synchronization. Profiling must tell
us its actual cost relative to message passing. The scalar conversion of every
loss component on every step is another smaller synchronization candidate;
retain finite-loss checks and transfer reporting scalars only when needed.

For precision, first test faster FP32 matrix multiplication in learned dense
blocks. Keep geometric decisions and VICReg covariance calculations at full
FP32 arithmetic. A blanket global TF32 setting can also affect matrix-based
distance calculations. BF16 is a subsequent experiment restricted to operators
supported by the installed MACE/cuEquivariance path, with sensitive reductions
kept in FP32. [PyTorch documents the precision/performance tradeoff](https://docs.pytorch.org/docs/main/generated/torch.set_float32_matmul_precision.html).

Compilation is a later step: isolate tensor blocks rather than expecting dynamic
edge extraction and Python orchestration to compile together. Check actual
recompilations using the [compiler diagnostics](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html).
cuEquivariance is already enabled, so its advertised acceleration is not an
additional available speedup for this run. [MACE acceleration documentation](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html).

The full effective batch contains 7,680 cloud encodings including augmentation.
Unused memory does not establish that all their training graphs fit at once.
Retain exact gradient caching until a measured alternative is better. Ordinary
accumulation of separate small-batch VICReg losses changes covariance statistics.
Switching orchestration to Lightning alone would not remove these compute costs.

## Representation and forecasting improvements

**Use a richer, smooth context readout.** The current pooled context is one
weighted average of first-layer atomic features. Add two or three overlapping
radial pools and weighted feature variance, then combine them with the central
features in the same exported embedding. This can retain where motifs occur
and whether the neighborhood is homogeneous or mixed. Multiplying every pool
by the existing outer taper preserves its boundary behavior. This is a proposed
small readout change using the pretrained backbone; improvement is unmeasured.

**Train smooth evolution that preserves structural changes.** Temporal VICReg
currently penalizes the distance between adjacent states. It can suppress both
nuisance motion and meaningful changes. Test a controlled replacement of some
temporal attraction with short-time predictive consistency or a weak temporal
curvature penalty on three frames. Keep variance/covariance regularization and
physical target supervision. Curvature regularization also needs event-timing
checks because real transitions can have high curvature.

At update 100, temporal/topology parameter-gradient cosine was -0.587; at update
200 it was -0.174. These are early batch diagnostics, not proof of the cause of
final validation behavior. Collect their distributions before changing weights.
If conflicts persist, compare gradient-norm balancing and periodic conflict-aware
updates. [CAGrad](https://arxiv.org/abs/2110.14048) is one principled candidate;
neither it nor PCGrad guarantees improvement of all held-out physical metrics.
Do not assume the previous aggressive topology-attraction experiment worked:
its improved topology metrics accompanied worse spatial/temporal behavior.

**Give the forecaster causal history.** Its current inputs are only the present
embedding, material, temperature and horizon. Local position embeddings omit
velocities and other microscopic degrees of freedom. Feed a small predictor
`z(t-0.2), z(t-0.1), z(t)` or use available velocity information, while retaining
the same single-frame encoder for static analysis. No post-origin frame may be
used as history when evaluating a forecast from time t. Test multiple horizons
and repeated rollouts, with physical TDA forecasts alongside latent error.
An uncertainty head can represent unresolved futures; evaluate calibration and
likelihood as well as mean prediction error. It cannot restore missing information.

Learning coordinates through their transition behavior has precedent in
[VAMPnets](https://www.nature.com/articles/s41467-017-02388-1). A later kinetics
objective would require appropriate conditioning across temperatures and
nonstationary crystallization trajectories; a pooled global transition model
would be an uncontrolled assumption here.

**Improve the target's stability and selectivity.** The existing audit found
neighbor membership changes in 6% of patches at a 0.005 Å perturbation. Whitened
TDA change MSE rose from 0.00401 with fixed members to 0.00758 after reselection.
The target also abruptly excludes persistence pairs with death radius above
3.5 Å. Test a smooth taper for that cutoff, and audit membership sensitivity
separately from persistence-image smoothing. Persistence-image stability theory
does not automatically remove discontinuities introduced by our preprocessing.
[Persistence images paper](https://www.jmlr.org/papers/v18/16-337.html).

Evaluate stable H1/H2 details separately, including same-material,
similar-density liquid environments. Report gain over material/density baselines
so a low global error cannot hide conditional-mean prediction. Preserve useful
short-lived features above the measured perturbation tolerance; removing all
short persistence indiscriminately could discard the desired liquid nuances.

## Data, evaluation and experiment order

The current balanced sampler draws 360,448 quadruplets per material per epoch.
Mg has 32,768 available training quadruplets, or 11 passes per epoch. Ta has
8,192, or 44 passes per epoch and 264 over six epochs, all from one trajectory.
More independent Ta trajectories, and additional Mg/Al source diversity, are a
priority. Select new liquid, precursor, interface and defect examples using
continuous descriptors and outcomes; PTM labels alone are not sufficient.
Hold out whole sources. More atom samples from one trajectory do not create
independent-source validation.

Use a broad representative validation set plus a fixed challenge set for subtle
liquid structure and genuine transitions. Several independent validation sources
and multiple training seeds are needed to assess small gains. The same exported
embedding must be evaluated for topology, spatial coherence, temporal behavior,
physical forecast skill and retained variation. A low aggregate training loss
must not compensate for a meaningful regression in one property.

Proposed sequence after the current analysis:

1. Profile an isolated representative run; benchmark chunk size, GPU-resident
   data and geometry reuse individually, then combine demonstrated improvements.
   Keep the effective batch, updates, sampling and learning-rate schedule fixed.
2. Test precision separately and reject numerical changes that obscure the
   perturbation-scale signals the experiment is intended to learn.
3. Compare the current readout with smooth radial pools plus feature variance,
   using equal data and update budgets. Evaluate all properties of the same z.
4. Test causal-history forecasting and then a replacement for part of temporal
   attraction, keeping these scientific hypotheses distinguishable.
5. Repeat promising changes with more independent sources and multiple seeds.

No numerical speedup or scientific gain is claimed before those measurements.
This file is an experiment record, not a maintained tool or a new queued run.
