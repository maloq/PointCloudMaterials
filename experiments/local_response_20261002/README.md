# Can simulator responses improve representations of local Al structure?

The previous experiment supervised complete periodic256-atom perturbed FCC cells.
This experiment trains a geometry-only128-channel MACE encoder with128-dimensional
export on real nearest80-atom neighborhoods extracted from larger MD states.
The learned nonlinear head predicts256 smooth future Fourier features. The
question is whether responses add predictive information useful under local
partial observation, beyond more ordinary shots or longer optimization.

Four treatments share inputs, target definitions, source roles, initialization
seeds and validation likelihood selection:

| Treatment | Training values | Additional supervision | Optimization |
| --- | --- | --- | --- |
| values8 |8 branches |None |200 epochs, patience40 |
| responses8 |8 branches |Two local directional derivatives per branch |Same epoch/patience limits |
| values32 |32 branches |None |Same epoch/patience limits |
| values8_time |8 branches |None |Same-seed response fit's measured optimization time;6000-epoch safety cap |

Three seeds give twelve fits. Each source contributes one predetermined
outcome-independent center/frame from its fixed64 identity contract:90 training,
15 validation and30 held-out sources. Calibration sources are unused. All models
are evaluated on every identical query. This is a new response-query assay,
not the all64 crystallization-window benchmark. No downstream crystallization
superiority is inferred from short20/100fs future-feature scores.

The teacher changes the parent physics from Lee2003 MEAM to fixed MACE-MPA-0.
All simulated atoms in the surrounding open environment move. Only the initial
local80 displacement is differentiated; the center stays fixed initially while
subsequent motion and exterior tangents evolve. Teacher450K and horizon metadata
are never encoder or predictor inputs. There is no history, observed velocity,
species variation, relaxation or explicit time/temperature conditioning.

Environment choice is a prerequisite: four training queries, two common-noise
branches, candidate radii18/24/30A versus all larger radii through36A. Each future
prefix must pass absolute value RMS1e-4 and relative response5% convergence.
This checks sensitivity to enlarging moving open environments; it does not prove
equivalence to the complete periodic70304-atom system. Failed gates halt training.

Primary comparisons are held-out noise-corrected value MSE and response MSE,
with proper feature-mean Gaussian NLL and paired source-bootstrap uncertainty.
The complete trained nonlinear predictor is evaluated; a fresh linear readout
is not used to judge whether nonlinear training preserved information.
Optimization and oracle acquisition costs remain separate. The time control
allows more checkpoint selections, so it is not matched selection multiplicity.

An advantage over values8 but not values8_time would suggest an optimization
explanation. An advantage over both value-only budgets would support useful
response supervision in this short-horizon setting. Failure to transfer from
complete cells to local observations would be scientifically informative:
fixed-exterior interventional responses need not equal gradients of the future
mean conditioned only on an80-atom patch.

Definitions: [metric contract](../../docs/metrics/local_response.md).
Execution: [workflow](../../docs/simulations/local_response_20261002.md).

## Completed results, October3

All135 queries and twelve fits completed; final evaluation finished05:50 Paris.
The24A moving environment passed the declared training-source convergence gate.
All twelve saved checkpoint/prediction hashes, full135-row prediction coverage,
finite predictions/responses and final online W&B summary receipts were verified.

Across30 held-out sources and three training seeds, full256-coordinate
noise-corrected MSE was:

| Treatment | Future-feature MSE | Response MSE |
| --- | ---: | ---: |
| Constant prior |0.50789849 |0.77681604 |
| values8 |0.27437001 |0.39338080 |
| responses8 |0.32415905 |0.43542142 |
| values32 |0.27039761 |0.38981217 |
| values8_time |0.27436061 |0.39338069 |

Response supervision increased average future-feature error18.15% and response
error10.69% relative to values8. Paired source-bootstrap95% intervals for these
absolute differences were respectively[0.02522438,0.07971184] and
[0.01983959,0.06640868]. These intervals condition on the three fitted seeds;
they do not include training-seed or training-source resampling uncertainty.
At20fs the response-error difference was inconclusive; the joint20+100fs
features showed the clearer deterioration.

Seed dependence is substantial. Response fits selected epochs12,200,12 for
seeds20261002/3/4. The middle seed improved future-feature error23.70% and
response error28.72% versus its matched values8 fit; the other two were worse.
Thus the result is an unstable treatment with worse average performance in this
pilot, not evidence that local response learning can never work. The complete
trained nonlinear predictor was evaluated throughout. Extra value-only training
time did not improve its validation-selected result; all three time controls
reached their measured budgets.

The next informative control would reuse the existing label bank to examine
response-loss weight and delayed response supervision, selecting only on the
declared validation likelihood. This is a proposal, not an additional launched
experiment. The present results remain short-time MACE-teacher feature prediction,
not an evaluation of original MEAM crystallization at3/6ps.

The completed [disturbance and hyperparameter audit](DISTURBANCE_AUDIT.md)
quantifies physical perturbations, feature sensitivities, label noise and
parameter-gradient competition at the frozen checkpoints. It launches no new fits.
