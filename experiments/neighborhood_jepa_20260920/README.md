# Snapshot MACE from equivariant space–time neighborhood prediction

Question: can prediction of neighboring local environments teach a more useful
invariant snapshot representation than physical/TDA anchors plus VICReg alone?
Can previous-center context improve that training without changing deployment?
This is a LeJEPA-inspired predictive objective, not a claim that image-domain
LeJEPA guarantees apply unchanged to equivariant molecular dynamics.

## Data and inputs

Use 65,536 dynamic training anchors sampled from the registered expanded
Al/Mg/Ti/Ta release, plus its unchanged 480 selection anchors from 15 native-Al
sources. The six training strata contribute Al shooting 27,173, native Al 8,995,
other Al 4,357, Mg 7,678, Ta 7,678 and Ti 9,655 anchors. The retained ancestral
lineage counts are 16 shooting, 36 native Al and one per other stratum. These are
correlated local observations, not 65,536 independent trajectories. No downstream
calibration/test data, static data or new simulations enter this training.

Select six actual neighbors at the current frame inside normalized radius 4.25.
Randomize the first neighbor, then maximize angular coverage; symmetry ties are
randomized. Four-neighbor variants use the nested first four. No unique cubic
orientation or lab-axis frame is imposed. Six selected neighbors are not claimed
to be an exact symmetric shell. Track the same atom IDs at previous/current/next
saved times. Each of the resulting 21 local views is a complete radius-8
snapshot, with the established material normalization and smooth 6–8 support.
There is no crop-around-neighbor approximation from the center's smaller patch.

The source timestamps provide actual negative/positive time offsets, including
irregular gaps. Only **current-frame** relative neighbor positions enter queries.
Future positions, future encodings and target physical observables are never
predictor inputs. They appear only on the target side of training. The previous
center's time offset accompanies the optional previous embedding context.

## Encoder and predictor

MACE is shared and independently encodes each snapshot. Its existing two
interaction blocks and cuEquivariance backend remain; the new angular pathway
forms weighted spherical harmonics l=1,2,4,6 from atom-level scalar features and
relative positions **before pooling**. Four channels per order give 120
components. This avoids trying to recover cubic orientation from a pooled l=2
feature that vanishes by symmetry. A 128-dimensional invariant readout combines
the original invariant state and channel Gram contractions of the tensors.

A shared equivariant query predictor sees the center state, query radius,
physical lag, contractions with query-direction harmonics, and optionally the
previous center state/time. It predicts both invariant and equivariant target
states. It is invariant to query-list permutation and covariant under joint
rotation of observations and query positions. It does not encode deployment
history: exported `NeighborhoodEncoder.export` returns `invariant` and
`equivariant` from one local snapshot.

Both target and input embeddings receive gradients through the same encoder.
There is no teacher, stop-gradient or moving target cache. SIGReg acts on the
current-center invariant projection within material/potential groups, not on
flattened equivariant components. Physical85, instantaneous TDA144 and tensor
q4m/q6m anchors supervise current/next centers in every arm. The past frame is
context/latent supervision, not a fake zero physical target. Temporal arms also
decode fixed future physical/TDA targets from the predicted center embedding.
No relaxed TDA or velocities are used.

Physical and TDA are decoded from the actual exported invariant state. Bond
orientation is decoded from the equivariant channels. Grouped head normalization
uses differentiable training moments, with fixed training-only recalibration for
validation. The physical/TDA heads receive the known material/potential group
so removal of domain offsets does not make raw-unit decoding ill-posed. SIGReg
receives no group embedding. The snapshot encoder itself is batch independent. See the exact
[loss and metric definitions](../../docs/metrics/neighborhood_jepa.md).

## Queue

One seed (20260920), 17 five-epoch-equivalent screens:

| Family | Variations |
| --- | --- |
| Controls | Anchors + SIGReg; anchors + temporal VICReg; anchors + spatial VICReg |
| Spatial | Four / six neighbor queries at the current frame |
| Space–time | Four / six neighbors at all three times, plus future center |
| Context | With / without previous-center equivariant/invariant state |
| Predictive weight | 0.03 / 0.1 / 0.3; extra 0.03 history variant |
| SIGReg weight | 0.003 / 0.01 / 0.03 |
| Learning rate | Head/encoder: .002/.0002, .001/.0001, .0005/.00005 |
| Fixed future target | Default weight 0.25 versus zero |

VICReg controls retain the previous normalized-VICReg coefficient 0.1; SIGReg
has its separately calibrated scale.

The default unnormalized SIGReg term was much larger than its physical anchor
in preflight. Weight 0.01 is therefore the default, with a declared sweep, rather
than letting distribution regularization dominate by accident. Tensor prediction
averages per-order component errors with a fixed 0.1 amplitude scale. The physical
and TDA anchors retain coefficients 1 and 0.25; tensor bond supervision is 0.1.
There is no direct backtracking/slowness penalty in this new comparison.

Train from scratch with global batch 256, MACE microbatch 256, minimum 32 anchor
pairs per material/potential group. Each update uses 15 or 21 views per anchor;
all arms retain current/next physical anchors. AdamW uses weight decay 1e-4,
10% warmup and cosine to 1% of peak; encoder/head gradients are clipped separately
at 1/5. BF16 applies selectively in the established MACE pathway; geometry,
higher-order tensors, predictor and objectives remain FP32. Full-batch derivatives
are replayed through encoder microbatches, including target derivatives.

One epoch equivalent is 256 updates at this data/batch size. It describes sampled
anchor draws, not a complete shuffled pass. Five-epoch screens have 1,280 updates.
Choose one learned control and two learned predictive settings by **fixed present
physical + 0.25 TDA selection error**, requiring improvement over the training-mean
baseline. Do not rank by moving latent loss. Restart those settings for 12 epochs
(3,072 updates), giving 20 fits in total. This selection is Al-only and does not
establish cross-material generalization. Next-center physical prediction and its
persistence baseline are reported separately; good downstream crystallization
performance remains to be tested after the encoder comparison.

Scientific config: [mace_20260920.json](../../configs/neighborhood_jepa/mace_20260920.json).
[Detached execution](../../docs/neighborhood_jepa_20260920.md) records the launcher
and exact frozen-code/provenance layout. No test cohort is used to tune the queue.
