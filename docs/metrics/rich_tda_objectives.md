# Packed rich-descriptor objective comparison (30 September 2026)

This is a new label-free encoder study, not a continuation of RH2. Historical
artifacts retain their frozen source and metric definitions. The user authorized
seven 20-epoch pilots on 1/10 of the existing fitting subset, followed by a
validation-selected full-data fit. No crystallization labels, AP selectors,
temperature, simulation age, time, species or material features enter fitting.

## Population and inputs

The full fitting population is exactly RH2's 1,056,768 raw patches and original
pool-row IDs, with its original training-only mean/SD/active-target transform.
Packing copies float32 normalized coordinates and raw 442 descriptors without
recomputing them. A fixed nested uniform draw of 105,677 rows (seed+97) supplies
every pilot. Pilot target moments are fitted only on those rows. All seven pilots
share sample IDs, moments, initialization seed and epoch permutations. Independent
epoch permutations use NumPy SeedSequence([seed, epoch]) over every fitting row;
each row occurs exactly once. The final partial batch is retained (7,373 rows
in pilots). No phase/material balancing changes the original sampling measure.

Encoder input is one nearest-80 current patch in fixed material-normalized units,
radius 8 and cutoff 5, no halo/history/motion/context. MACE width/export 256,
three interactions, angular/correlation orders 3, constant atom channel and
cuEquivariance are retained. Only the exported scalar state enters the descriptor
head. Vector channels enter regularization only. Regularization projectors are
training-only and never replace the exported state. All heads predict the same
99 geometry, 40 bond-order, 45 CNA and 258 alpha-complex TDA coordinates.

## Controlled treatments and losses

The first four arms are a 2x2 comparison: VCReg on exported scalar/vector channels
or separate projectors, crossed with covariance denominators D(D-1) and D.
Scalar projector: 256 -> 512, LayerNorm, SiLU -> 256. Vector projector: bias-free
32 -> 32 channel mixing, preserving Cartesian equivariance. The projectors are
initialized identically in every arm; unused projectors do not receive gradients.
Scalar and vector covariances retain the original population-moment denominator
N (and division by 3 for Cartesian components), with differentiable global moments.
Variance coefficient .05, std floor 1, epsilon 1e-4, covariance coefficient .01.
Thus per-channel normalization strengthens covariance by 255 for scalars and 31
for vectors; it is not a renaming of the same penalty. All arms also retain
0.01*mean_channels(mean_batch(exported_z)^2), and a five-epoch regularizer ramp.

Three incremental arms extend projector/per-channel VCReg:

1. **tda-heads:** the residual decoder trunk is shared; 21 semantic blocks receive
   separate width 64 nonlinear heads plus the direct linear state readout. Three
   blocks are geometry/bond-order/CNA; the 18 TDA blocks are neighborhood 32/80 x
   H0/H1/H2 x statistics/spectrum/smooth-Betti. Each family has 1/4 task mass;
   the TDA family assigns 1/18 to each block, then averages active coordinates.
   This is a joint head/semantic-weighting treatment, not an isolated capacity effect.
2. **tda-blocks:** retain these heads and block weights; replace coordinate-wise
   inverse-variance weighting with a common scale within each block. In canonical
   standardized coordinates, weight_j is proportional to fitting SD_j^2 within
   a block. Equivalently, its error is summed raw-coordinate squared error divided
   by summed fitting variances. Means, active mask and exported output units stay
   unchanged. This applies to all descriptor blocks, including the three other families.
3. **tda-distance:** add weight .01 topological distance contrastive loss, ramped
   over five epochs, on the actual exported scalar state. A deterministic uniform
   panel of up to 512 positions in the randomly permuted batch supplies pairs.
   Teacher distances are squared Euclidean distances over the six H0 death-RBF /
   H1,H2 image-like spectral blocks; each has equal mass and block variance scaling.
   Targets are fixed, detached descriptors. Cosine similarity logits use contrastive
   temperature .1 (an algorithm parameter, not a physical model input). For every
   ordered non-self pair i,j, the denominator sums exp(sim(i,k)/.1) over all k!=i
   with teacher distance(i,k)>=distance(i,j). All ties are included. Average the
   negative log probabilities over pairs. Sorting and cumulative log-sum-exp avoid
   a cubic tensor. No crystallization/event ranking is involved.

The descriptor training objective is .5*weighted_squared_error+.5*log(2*pi).
VCReg and topology penalties are separate. Every model exports canonical
coordinate-standardized predictions; changed training weights do not silently
change evaluation definitions.

## Selection, diagnostics and exports

All 192,960 original Al selection rows are evaluated every epoch. Checkpoint and
pilot-method selection use the same RH2 family-balanced coordinate-standardized
Gaussian NLL: equal mass per family, equal mass per active coordinate inside it.
This selector excludes all regularizers, uses each pilot's common fitting transform,
and is unchanged across objective variants. Ties use the declared arm index.
The best pilot recipe is initialized from scratch on the full original subset;
no pilot optimizer/weights are reused. Full-data moments revert to RH2's original
moments. Pilot and full normalized scores have different fitting transforms and
are not interchangeable. Pilot selection performance is selection-biased; the
final test is the independent check. No pilot test evaluation is performed.

Mean-relative errors, individual feature MSE/RMSE/R2 and final export definitions
are those of `rich_multimaterial_encoder.md`. The full fit alone exports calibration
and test, retaining all fixed Al64 IDs. No held-out external-material claim is made.
Additional H0/H1/H2 rows average active coordinates of that homology dimension
across both neighborhood sizes, with the same mean-baseline denominator convention.
These are diagnostic subgroups and do not change the overall selector.

Validation covariance diagnostics use centered scalar vectors, uniform mass for
all selection rows, population covariance. Eigenvalues are clipped at zero for
roundoff. Participation rank=(sum eigenvalues)^2/sum eigenvalues^2; d95 is the
smallest leading-eigenvalue count reaching 95% of variance. Compute both for raw
exported states and regularized scalar states. Minimum SD is the minimum square
root diagonal covariance. These are linear effective dimensions, not nonlinear
manifold dimension estimates; uniform rows are not source-balanced weights.

Training logs scalar/vector variance and covariance components and the RMS
derivatives of task loss and scalar/mean regularization with respect to exported
z, multiplied by the per-rank batch count to remove the mean-loss scaling. Their
RMS ratio measures the scalar regularization/task balance at z, not total parameter
gradient competition (vector gradients are excluded). No fixed values or duplicate
wall-time history is sent to W&B. Scientific fits are online; local numerical checks
and selection tables do not create runs.

All runs use AdamW, global batch 8192, max LR .01, five-epoch warmup/cosine to 1e-5,
gradient clip 5 and 20 epochs. Pilots have 13 updates/epoch (260 total); full fit has
129 (2,580 total). Checkpointed patch chunks512 bound activation memory; global
VCReg remains whole-batch. No GPU-hour budget changes data or the scheduler.

## Literature and scope

[RipsNet](https://proceedings.mlr.press/v196/surrel22a/surrel22a.pdf) learns fixed
persistence vectorizations with squared error. [TopologyNet code](https://github.com/ZJUCAGD/TopologyNet/blob/main/utils/train_TopologyNet.py)
uses separate H1/H2 MSEs. Our structured heads/blocks are adaptations to the existing
heterogeneous descriptor bank; its Gaussian samples are not canonical integrated PI pixels.
[Luo et al., NeurIPS 2023, section3.2](https://papers.neurips.cc/paper_files/paper/2023/file/6b555e8552240d6dfe0767146c9ebf36-Paper-Conference.pdf)
motivates ordered topological-distance contrastive supervision. Their molecular
filtrations differ from our geometry-only alpha-complex descriptors. Projector
placement and per-channel covariance normalization are motivated by
[VICReg](https://arxiv.org/abs/2105.04906); this remains single-view VCReg with
descriptor supervision, not a claim to reproduce the full two-view VICReg method.
