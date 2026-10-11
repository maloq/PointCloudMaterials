# Does continuous motion help between sparse observations?

The saved0.1ps observation spacing hides fast motion. An autonomous latent
stochastic model can be queried at arbitrary physical durations, but that
alone does not identify the missing dynamics. This experiment combines real
velocities, actual long-horizon outcomes and short simulator sensitivities.
It does not infer that an interpolated path is physically correct merely
because it is continuous.

## Representation and comparisons

For a local80 state Q,V, the structural embedding is s=E(Q), and the motion
embedding is m=0.1ps DE(Q)[V]. In internal time tau=t/(0.1ps), the learned model
has ds=m d tau and dm=a(s,m)d tau+B(s,m)dW. Neural functions are autonomous;
duration is an integration interval, not a predictor feature. The initial
acceleration has a bounded learned residual and unit latent damping. This
initial0.1ps damping is an architectural regularizer, not the physical
thermostat damping. Diffusion acts on latent motion. No coordinate decoder
or physical reconstruction loss is trained.

| Treatment | Latent integration | Short teacher supervision |
| --- | --- | --- |
| discrete | fixed0.1ps kinetic step | conditional mean latent endpoint |
| continuous |0.05ps maximum step, shortened to query times | same endpoint |
| derivative | same continuous rule | same endpoint plus analytic input JVPs |

All arms have identical parameter counts, inputs, targets, data splits,
optimizer budget and three seeds. The first comparison isolates time-step
resolution in the same parameterization; it is not a comparison against a
separately optimized RNN. The second isolates simulator derivative supervision.

The active budget is1600 optimizer updates per fit, batch/microbatch256:
409600 sampled neighborhoods per model (about67 dataset equivalents over6080
fitting rows, with replacement and ancestor-balanced sampling). All nine fits
run the complete budget; inner-validation likelihood chooses the exported
checkpoint. The original400-update plan was superseded after teacher collection
and before fitting. Extending duration introduces no new data or simulator shots.

The long objective is likelihood of24 radial/angular future statistics at
0.1,0.5,3,12ps. These statistics anchor the representation to real future
structure and prevent an unconstrained latent teacher/student system from
learning arbitrary constant endpoints. A small variance/covariance penalty
also discourages collapse. The short target encoder uses a stopped-gradient
EMA. No crystal labels, crystallinity losses, AP optimization or explicit
conditions are used.

The whole cell is simulated, while the student sees80 atoms. Simulator force
Hessian-vector products propagate local Q and V perturbations through the
100fs continuation. The learned prediction is differentiated with respect to
the same initial inputs. Independent physical noise shots estimate the
conditional mean and its derivative; latent SDE particles have their own
noise and are not incorrectly equated to physical Brownian trajectories.

## Falsifiable checks and limits

Held-out ancestor future NLL, normalized feature MSE, persistence/unconditional
controls and90% marginal coverage measure predictive information and calibration.
Actual10/20/50fs held-out MEAM observations check sub-cadence behavior. Native
0.01ps histories supply a separate withheld-time diagnostic with their original
training ancestry retained. Drift-only step refinement checks whether smaller
integration steps materially change the answer. All64 and legacy16 are reported.

An improvement requires reproducible gains across seeds and ancestors against
the matched no-derivative arm, not just a smaller training tangent loss. A
continuous model can still be wrong at unobserved times. The local80 environment
is open and may require a halo or history for approximate Markov closure; this
first comparison holds its spatial support fixed. Descriptor likelihood is not
the full future atomic distribution, and this first run does not measure
nucleus-birth AP or establish improved nucleation prediction.12ps reaches beyond
previous0.5ps targets but is shorter than many observed embryo establishment
times. Two physical shots per anchor limit conditional-law accuracy.

## Literature basis

- [Neural ODEs](https://arxiv.org/abs/1806.07366): continuous learned dynamics;
  solver queryability is distinct from identifiability between observations.
- [Latent SDEs](https://proceedings.mlr.press/v108/li20i.html): stochastic latent
  evolution for partially observed dynamics. This pilot uses direct predictive
  mixture likelihood, not that paper's variational path objective.
- [Discovery of coordinates and dynamics](https://arxiv.org/abs/1904.02107):
  encoder chain-rule derivatives motivate m=DE(Q)V. Its reconstruction treatment
  is deliberately omitted under this project's current research policy.
- [Generator EDMD](https://arxiv.org/abs/1909.10638): local generator information
  includes stochastic derivative terms. Here analytic finite-time tangent
  targets are used; there is no claim that JVP alone is a complete Ito generator.
- [Equivariant Graph Neural Operator](https://arxiv.org/abs/2401.11037): geometry
  and velocities in molecular trajectory models. Its small-molecule benchmarks
  do not establish bulk-Al transfer. The present kinetic latent SDE is not a
  Fourier neural operator or a mesh-independent PDE operator.

Recipe and reproducible commands: [workflow](../../docs/continuous_motion.md).
Calculations and uncertainty: [metric contract](../../docs/metrics/continuous_motion.md).

## Nucleus birth at longer lead times

Evaluate whether the frozen discrete, continuous and derivative-trained codes
contain information about root-nucleus birth1/3/6ps later. Compare structure,
structure+motion and predicted future-law spaces, with nonlinear and linear
readouts and descriptor/prior controls. Use the common exact0.1ps labelled
velocity cohort, original-training-only NLL readout selection, and the requested
pooled validation/calibration/test. Neighborhood eligibility,1:4 controls and
whole-ancestor uncertainty follow the [metric contract](../../docs/metrics/continuous_motion_birth.md).
The [recipe](../../configs/analysis/continuous_motion_birth_20261006.json) writes
`continuous_motion/all010-20261006-v3/analyses/birth-136-v1` under training storage.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.birth launch --config configs/analysis/continuous_motion_birth_20261006.json
```
