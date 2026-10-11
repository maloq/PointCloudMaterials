# Response weight, optimization duration and finite perturbations

The previous local80 response fit was worse on average, but one seed improved
both value and response error. Two seeds selected update12 and stopped at52;
their late value/response gradients strongly opposed each other. The third ran
to200 updates. This motivates separating response weight from premature
stopping without claiming either explanation is established.

The new comparison fixes400 updates for all15 fits: three seeds each for weights
0,0.1,0.3,1, plus0.3 after100 value-only updates and a50-update linear ramp.
Architecture, initialization seeds, optimizer, original mean8 labels, training
normalization, source roles and input geometry remain matched. The original
validation feature Gaussian NLL remains the selector. The final400-update
checkpoint is also evaluated as a declared diagnostic, never selected by test.
There is no AP-based loss, model selection or promotion.

The finite-response assay asks when a local derivative ceases to approximate an
actual future change. Whole-displacement norms0.03,0.1,0.3,1,2,3Å correspond to
per-atom RMS0.00338,0.01125,0.03375,0.11251,0.22502,0.33753Å. Four training
parents, two directions, both signs, two common-noise shots and24/36Å environments
give384 perturbed forwards to100fs, observed at20/100fs. Central trajectories
also supply raw physical-observable derivatives. Outputs compare simulated and
linearized Fourier/physical changes, atomic divergence, energy disturbance,
minimum distances, and halo convergence. The larger amplitudes are a stress
assay; no assumption that they are typical thermal fluctuations is made.

The student sees only the original center-relative80-atom geometry and center
indicator. Encoder support8Å, edge cutoff5Å,128 channels/exported features,
no history, motion, temperature, age, species or material channels. Its predictor
sees only the128-dimensional embedding. Teacher-only halo motion and450K
thermostat conditions remain explicitly separated from model inputs. Nominal
batch256 retains the recorded microbatch8 higher-derivative memory exception;
the actual accumulated optimizer batch is all90 training parents.

All30 old held-out sources are retained. They are reused exploratory evaluation,
not a fresh confirmation cohort. Source-bootstrap intervals condition on the
three fits. These short-time MACE targets do not establish crystallization
prediction under the original MEAM dynamics.

Recipe: [local_response_followup_20261003.json](../../configs/simulation/local_response_followup_20261003.json).
[Metric definitions](../../docs/metrics/local_response_followup.md) specify the
calculation, and [execution workflow](../../docs/simulations/local_response_followup_20261003.md)
contains reproduction commands and run locations.
