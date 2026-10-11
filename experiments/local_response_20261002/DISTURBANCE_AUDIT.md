# Disturbance and hyperparameter audit, 3 October 2026

This audit reads the frozen135-query bank and twelve existing fits. It recomputes
separate parameter gradients on the90 training queries, without an optimizer
step, new checkpoint selection or new W&B run. Every original bound scientific
source hash and the label-bank hash matched. New diagnostic tables have their
own [metric contract](../../docs/metrics/local_response_audit.md).

## What is perturbed

The input is the center and79 nearest neighbors from a70304-atom Al MD source.
The MACE teacher simulates a moving open24Å environment of roughly3300 atoms.
Two random Gaussian initial displacement directions are projected to zero summed
displacement over the79 neighbors, orthonormalized and padded with zeros at the
center and outside the local80 atoms. All simulator atoms subsequently move;
the center is fixed only in the initial perturbation.

Each direction has unit Euclidean norm over all Cartesian coordinates. Therefore
epsilon is the length of the complete displacement vector. It is not the
displacement of each atom. Measurements across90 train parents and two directions:

| Collective epsilon, Å | Per-mobile-atom vector RMS, Å | Largest atom displacement, Å |
| ---: | ---: | ---: |
|0.003 |0.000337526 |0.000949286 |
|0.010 |0.001125088 |0.003164287 |
|0.030 |0.003375264 |0.009492861 |

MACE training uses analytic infinitesimal directional derivatives at epsilon0;
the0.01/0.03Å finite shifts are derivative checks. UNEP/MEAM instead estimate
derivatives by the paired central difference at finite epsilon. Both signed
paths use identical initial momenta and complete thermostat-noise streams. Each
path's initial feature anchor is perturbed too. Decoded source positions are
continuous float32 inputs during MACE differentiation, without re-quantization.

## What responds

The target is a256-dimensional fixed Fourier feature vector of future changes
in eight smooth radial densities and four angular powers around the tracked
central atom.128 components describe20fs changes;128 describe joint20+100fs
changes. The Fourier amplitude is0.125 per component and path coordinates are
divided by0.1 before the fixed random projection. This is an approximate
distribution embedding through its branch-average features, not an explicit
trajectory-density model or a crystalline-fraction prediction.

Train parents supply8 response shots each. Their mean response RMS is:

| Feature prefix | Raw mean response, feature/Å | Raw feature change at epsilon0.01Å, linear estimate | Change in training feature-standard-deviation units |
| --- | ---: | ---: | ---: |
|20fs |0.0120974 |0.000120974 |0.00409170 |
|Joint20+100fs |0.0228848 |0.000228848 |0.00454599 |
|All256 |0.0183039 |0.000183039 |0.00432482 |

These are RMS across90 parents, feature coordinates and two directions after
averaging8 shots. Plus-to-minus changes are twice the displayed central-to-plus
changes. They measure feature sensitivity, not atomic trajectory displacements.
The simulation uses450K,1fs BAOAB steps and100fs friction. The horizons are only
0.02/0.10ps; original3/6ps crystallization has not been evaluated by this assay.

## Loss and optimizer

Each Fourier coordinate is centered/scaled by first-eight-shot training
statistics. Feature standard deviations range0.001967–0.082991; none hits the
1e-4 floor. The scalar response scale is0.4324816 perÅ, above its0.001 floor.
For the predictor's standardized feature outputs, the objective is value
half-MSE plus lambda times response half-MSE after division by that response
scale. Lambda was1 throughout. Constant terms produce the recorded Gaussian
feature-mean NLL; selection uses only validation value NLL.

The zero-output target energies are0.21346 for the value half-MSE and0.5 for
the response term. Thus equal coefficients do not yield equal optimization
influence. Direct gradient measurements at initialization show response/value
gradient-norm ratios2.225,2.456 and2.339 for the three seeds. The initial gradients
are aligned (cosines0.856–0.870), so this initially accelerates a shared direction.

The geometry-only MACE encoder has634496 trainable parameters and the nonlinear
head49536, total684032. There are90 distinct train environments,15 fixed
validation environments and30 held-out environments. Native random initialization
is used; the pretrained simulator potential is a separate teacher.

| Setting | Recorded value |
| --- | --- |
|Encoder/export width |128/128 |
|Predictor head |128→128 SiLU→256 |
|Graph cutoff/interactions |5Å /2 |
|Optimizer |AdamW |
|Encoder/head learning rates |1e-4 /5e-4, constant |
|Weight decay |1e-5 |
|Response weight |1, applied from update1 |
|Actual optimizer batch |All90 train parents |
|Microbatch |8, gradients accumulated |
|Maximum ordinary epochs/updates |200 |
|Early-stopping patience |40 updates |
|Global gradient clipping |10; never activated |

## Why the present optimization is questionable

One epoch is one optimizer update. Response seeds20261002/4 selected epoch12
and stopped at52. Seed20261003 avoided that stop, ran200 updates and selected
its final checkpoint. It was also the seed that improved held-out prediction.
The failed seeds' late validation curves have not been measured, so the claim
that they would recover with more updates remains a hypothesis.

At their final epoch52 checkpoints, the first/third response seeds have nearly
equal value and response gradient norms, but cosines-0.926 and-0.935 across all
parameters (-0.944/-0.951 within the encoder). The objectives compete strongly
there. This is expected to be possible in a multi-objective fit; it does not
prove a gradient-sign bug or make lambda1 mathematically invalid. It does give
direct evidence for testing a different weight/schedule.

The maximum recorded combined gradient norm is1.3505, far below clipping10.
At the checked MACE gate, AD/finite-difference relative error is0.0315% for
epsilon0.03Å and0.1109% for0.01Å. The student's JVP check is also below0.1%.
There is no evidence in these checks of numerical differentiation failure or
exploding gradients. Training-shot noise contributes an estimated5.58% of
mean-response squared magnitude, versus19.18% for standardized mean values.
This pooled estimate does not make every individual response precise.

The response task also differs conceptually from value prediction under partial
observation. Teacher derivatives hold the initially unobserved exterior fixed
while moving local atoms. The predictor sees only80 atoms, and therefore learns
an average over unobserved environments. Those two derivative quantities need
not agree. This remains a scientific hypothesis, separate from the measured
loss competition and stopping behavior.

## Proposed follow-up, not launched

Reuse the existing MACE bank, fixed source roles and validation likelihood.
Keep the old results intact. Use a fixed400-update budget for all seeds so the
52-update stopping difference cannot dominate. Compare lambda0,0.1,0.3,1,
plus a lambda0.3 treatment with100 value-only updates and a50-update ramp.
These are candidate settings, not validated optima. Select checkpoints by the
same validation value NLL and report seed dispersion as well as source intervals.
A separate lower-learning-rate control can follow if validation/gradient curves
still show instability. Increasing simulator shots or changing potential is not
needed for this first optimization check. More independent parents and more
than two displacement directions are subsequent scientific controls.

Reproduction:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m src.research.local_response_audit --config configs/analysis/local_response_audit_20261003.json
```

Outputs: [audit tables and figures](../../output/response_atlas/local80-training-20261002/analyses/disturbance-audit-v1/).
