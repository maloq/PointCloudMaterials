# Predictive encoders that retain useful structure and tolerate small perturbations

Question: can an encoder improve **sustained local crystallization onset AP** by
learning event-relevant structural information, while retaining present/future
physical information and avoiding the noise sensitivity of our current Geoformer?
This is a prospective one-seed screen, not a claim that any proposed method wins.

## Literature and the decisions it supports

- [Zaidi et al., Pre-training via Denoising for Molecular Property Prediction,
  ICLR 2023](https://arxiv.org/abs/2206.00133) show that coordinate denoising can
  provide useful molecular pretraining. Their equilibrium/force-field argument
  does not establish a force-learning interpretation for our off-equilibrium Al
  observations. Here we test noisy-input **fixed physical-target reconstruction**
  and modest latent consistency. This is not a reproduction of their vector
  denoising loss and does not estimate physical forces.
- [Brown et al., Smooth-AP, ECCV 2020](https://arxiv.org/abs/2007.12163) replace
  discontinuous ranks with sigmoid comparisons. We adapt that construction to
  source-weighted binary onset AP. The full fitting risk set is small enough to
  avoid sampled ranking estimates: all 827 windows and 30 positives participate.
  Exact self weight is retained. Tests compare the low-temperature limit with
  weighted empirical AP and compare cached/replayed gradients with ordinary
  backpropagation. Retrieval results are motivation, not evidence of Al onset skill.
- [Qi et al., Stochastic Optimization of Areas Under Precision-Recall Curves,
  NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/0dd1bc593a91620daecf7723d2235624-Paper.pdf)
  provide a distinct AP-optimization approach with stochastic compositional
  machinery. We choose the simpler full-cohort sigmoid-rank experiment for this
  small dataset. We do **not** call our implementation SOAP, claim its convergence
  guarantees, or substitute AUROC ranking for AP.
- [Pezzicoli, Charpiat and Landes, Rotation-equivariant Graph Neural Networks for
  Learning Glassy Liquids Representations, SciPost Physics 2024](https://arxiv.org/abs/2211.03226)
  support retaining equivariant geometric information when learning structural
  predictors of dynamics. Our architectural hypothesis is more specific: scalar
  pooling may discard orientation alignment. We retain invariant contractions of
  vector and rank-2 channels at three radii, including both mean squared local
  magnitude and squared magnitude of the pooled tensor. Their glass benchmarks
  do not validate this crystallization readout.
- [Bapst et al., Unveiling the predictive power of static structure in glassy
  systems, Nature Physics 2020](https://www.nature.com/articles/s41567-020-0842-8)
  motivate learning structural predictors of future dynamics and examining spatial
  extent. Our first architectural comparison holds the available 8 Å support fixed;
  three pooling scales do not add observations outside that support.

Our relaxed-structure teacher is an experiment motivated principally by our own
relaxed-descriptor results. It uses descriptors of the paired **present** relaxed
configuration as privileged training targets; the observed-input student does not
require relaxation at deployment. Full-cell relaxation can carry information from
outside the local observation. It is a useful target, not an exactly recoverable
function of the observed local patch.

## Predeclared arms

All arms use the native two-block, width-32, cuEquivariance MACE backbone, export
128 channels, and share one seed, fitting/tuning/development sources and physical
labels. Width is held at the existing native geometry setting; it is not a claim
about the earlier width-16/32 temporal backbone comparison.

| Arm | Change | Matched comparison |
|---|---|---|
| A-physical | Present geometry/order and fixed 3/9/12 ps order-increment targets; hazard head sees detached z | Informative encoder control |
| B-onset | Hazard likelihood also trains the encoder | B vs A: onset information in z |
| C-rank | Add full-fitting-cohort Smooth-AP gradient every 16 updates | C vs B: ranking objective |
| D-relaxed-teacher | Also reconstruct present relaxed geometry from observed inputs | D vs C: privileged structure |
| E-noise-teacher | Add noisy-input clean-target reconstruction and latent consistency | E vs D: robustness training |
| F-tensor-noise-teacher | Replace scalar pooling/readout with multiscale tensor contractions | F vs E: architecture in the robust setting |
| G-tensor-rank | Tensor readout with C's tasks | G vs C: architecture without teacher/noise |
| H-relaxed-input | C's objectives with relaxed inputs | H vs C: deployment input domain |

The tensor readout has more parameters; it is a practical architecture comparison,
not a capacity-matched proof that tensors explain any improvement. E/F add noisy
forward passes; C–H add full-risk ranking passes. Updates are matched, FLOPs are not;
logs and the preflight record report measured cost and parameter counts.

The scalar readout exports normalized center/pooled channels plus learned channels.
The tensor readout compresses center scalars and three smooth summaries at 4/6/8 Å
to 128 channels. Each scale includes scalar means, vector/tensor local strength,
vector/tensor alignment and a smooth count. Fixed density normalization and smooth
radial support avoid hard neighbor-count denominators.

## Training and selection

Reuse 2,880 paired snapshots from 45 independent Al MEAM roots: 25 fitting, 5 tuning,
15 development. Only 30/8/18 positive 12 ps windows occur in these splits, among
827/231/643 eligible windows. These development roots have been reused; they are
**not an untouched test set**. No new simulation or relaxation is requested.

The shared physical loss is equal-block radial/l2/l4 geometry MSE, plus 0.25 current
order MSE and 0.25 multi-horizon order-increment MSE. Standardization is fit-only.
The teacher adds 0.5 relaxed-geometry MSE. Noise arms add 0.5 of the physical loss
on a noisy view and 0.1 latent MSE divided by fixed initial fitting variance; clean
latent targets are detached. There is no variational information bottleneck,
whitening, slowness, bending, or prescribed low-dimensional motion subspace.

Each update samples 128 source-balanced physical observations and 128 separately
source-balanced eligible risk observations, preserving natural event prevalence.
Hazard likelihood has five intervals ending at 0.75/3/6/9/12 ps. Every 16th update,
AP arms add one full-cohort weighted Smooth-AP loss (coefficient 1, sigmoid
temperature 0.01 on 12 ps cumulative risk). Gradient caching preserves that full
loss while replaying encoder gradients through microbatches of 32. No stochastic
encoder layers or parameter updates intervene between caching and replay.

Use 2,048 updates, encoder/head LR 1e-4/3e-4, 128-update warmup and cosine decay.
Geometry/head initialization uses fitting observations only; the event head starts
at fitting conditional hazards. Primary selection is maximum tuning AP among
128-update checkpoints retaining geometry, current-order and future-increment
errors within 5% of their calibrated initialization. The initial checkpoint is
eligible. Report selected step: an untrained winner is not an improvement.
Final checkpoints are retained even if they fail this gate.

## Evaluation and limitations

- Source-weighted AP at each event horizon; primary AP12. Also Brier, hazard NLL,
  recall and achieved development FPR at a tuning-derived 5% FPR threshold.
- Separate frozen linear/MLP probes of z plus temperature, selected by tuning NLL.
  Their results are separate from the jointly trained AP-selected head.
  Matched observed-geometry, relaxed-geometry and temperature-only descriptor
  controls use the same probes and cohort (six additional inexpensive fits).
- Physical decoding, full paired-corpus spectral dimension, and exact **0.75 ps**
  jump/movement spectra on the previous 32,040-state dense evaluation chart with
  420 reference observations. That chart can overlap fitting roots: it is a
  descriptive transfer diagnostic, never used for selection. H's observed-coordinate
  dense evaluation is explicitly a domain-transfer check.
- New independent development perturbations at 0.1%, 0.5%, 1%, 3% expected 3D RMS
  displacement relative to mean center-to-12-nearest-neighbor distance. Report
  realized percentages, Å and latent response separately. Center fixed, edges and
  geometric features rebuilt, original finite candidate support retained. The
  training bank has three fixed views at 0.2%, 0.5%, 1%; it is not fresh Gaussian
  sampling every update. Noise tests probe added perturbations on quantized data,
  not recovery of the lost coordinate precision.
- Paired temperature-stratified root-bootstrap AP differences relative to B;
  intervals describe source sampling only, not training-seed uncertainty.

This screen can identify useful objective/readout directions, but 30 fitting events
cannot establish the best encoder generally. A larger, ancestry-controlled onset
cohort should confirm finalists before further seed or hyperparameter sweeps.

Configuration: [screen_20260924.json](../../configs/robust_onset/screen_20260924.json).
Implementation: `src/research/robust_onset/`. Reproduction and queued-job details:
[operations](../../docs/robust_onset.md). Results:
`output/encoder_research/robust-onset-20260924/RESULTS.md`.
