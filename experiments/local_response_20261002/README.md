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
