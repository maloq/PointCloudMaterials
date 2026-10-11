# Position/velocity-conditioned MEAM responses

This is a new unrelaxed local80 protocol, not a continuation of frozen MACE
fits. The original135 query identities and fixed Al64 release
e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d retain
90/15/30 train/selection/test ancestors. Four sampled velocity fields per
geometry do not increase independent ancestry. Temperature, age and time are
not network inputs. Velocity is explicitly authorized on4 October2026.

Teacher: the validated exact Lee2003 pure-Al MEAM PyTorch implementation,
float64 positions/forces/tangents,1fs BAOAB,450K,100fs friction. Parent MD has
the same potential but a different integrator/thermostat/ensemble protocol.
Initial velocities are independent full-cell Maxwell draws, with full-cell COM
removed once, saved in A/fs and held fixed across stochastic replicas. They
are sampled input states, not measured parent MD velocities. All environment
atoms move. Future noise is generated in original atom order before gathering
the nested environment; it is independent across velocity states and samples.

Targets have128 fixed Fourier features per marginal horizon100/500/1000fs.
Each block describes24 absolute future observables and24 changes from its own
initial anchor. Observables are the existing12 radial and12 angular densities,
with5.5A smooth support. The observable mean/SD comes from the recorded720
train patches of the previous campaign, all within the same90 train ancestors.
One fixed Gaussian48x128 map, frequencies divided by sqrt48, and uniform phase
is reused across horizons. This is384 coordinates, not a joint-path law or a
full atomic trajectory likelihood. Feature mean/SD for fitting uses only the
new training shots, SD floor1e-4.

Two inherited orthonormal coordinate displacement fields have unit collective
norm, zero central displacement and zero summed neighbor displacement. Two
velocity directions reuse these fields times0.004A/fs. Thus their derivatives
are per dimensionless collective velocity multiplier, whereas positional
derivatives are per collective Angstrom. Initial momentum tangent is mass
times velocity tangent divided by ASE's fs unit. The anchor and trajectory
are both differentiated. A velocity perturbation does not perturb the initial
coordinate anchor. Full-path AD is checked by common-noise centered finite
differences at multipliers0.001/0.0001 for one direction of each kind, at every
horizon; the better relative error must be<=.02. Value-only reexecution agrees
with derivative-execution values to maximum absolute1e-9 on one independent
execution audit. No differentiable fit through teacher parameters is needed.

Periodic revision:24/36A and36/48A open-environment checks failed at longer
horizons. The original numerical queries remain frozen. The periodic implementation
uses GPU spatial bins and minimum-image displacements for pairs, screening triples
and future observables. It retains the exact energy algebra and pinned potential.
Every one of four train parents is checked against full-periodic LAMMPS energy
(error/atom<=1e-8eV), force RMS<=1e-8eV/A, and force-direction finite differences
(relative<=.002). New periodic complete20fs positional/velocity derivatives are
checked to relative<=.02, supplementing the already passed1ps open-integrator
checks. An independent value-only execution audit uses max absolute1e-9.
Four full-cell1ps references then assess the original48A results, at every
horizon and both derivative kinds. Only if ALL value RMS<=1e-4 and response
relative<=.05,48A is used as validated acceleration; otherwise the full native
periodic cell is used. No shorter horizon is substituted. The periodic force
and boundary tables record all individual errors, not only aggregate pass flags.

Training has4 noise replicas per velocity state; validation16 values; test16
values and8 derivatives on disjoint streams. Training values reuse response
paths after the independent execution audit. Test correction removes the
estimated variance of the finite-shot target mean using unbiased sample
variance/B. Velocity fields for different source roles remain disjoint.

The Q and QV models have identical parameter counts,128 MACE channels and128
exported coordinates. Geometry has8A support,5A edges, nearest80 atoms and one
constant atom channel plus center indicator. QV supplies velocities through
five rotation/reflection-invariant edge features: squared center-relative
speeds of sender/receiver, their dot product, and their projections along
the directed edge. A5->32->6 MLP adds a smoothly tapered contribution to the
radial features. This is an invariant velocity-conditioned graph, not a full
vector-channel extension. Q sets velocity features to zero; its equally sized
MLP still participates. No explicit metadata, halo, history or relaxation enters
either encoder. The head sees z128 only.

value_nll = .5 mean((prediction-standardized_mean_target)^2)+.5 log(2pi),
equal feature/horizon/state weighting. Q inputs repeat for the four velocity
fields: minimizing this conditional-target loss estimates their marginal
mean. response_half_mse = .5 mean(((predicted_directional_derivative-target)
/response_scale)^2), with feature-normalized target and separate training RMS
for positional/velocity derivative groups, floor.001. Q response arms use only
the two positional directions; QV response arms use all four, equal group weight.

All arms have distance weight.1. QV distance targets use raw Fourier shot-bank
mean differences squared, minus each independent bank's trace(sample_cov)/4.
Q distances use the four conditional velocity-field means as independent
replicates, subtracting trace(sample_cov_of_field_means)/4; this estimates
velocity-marginal future distance and includes within-field sampling noise.
Negative distance estimates are retained. Distances are divided by the mean
over distinct training inputs (must be positive). Predicted distance is mean
coordinate squared z difference divided by twice initial train embedding
variance, frozen per fit. Repeated input identities are excluded: phase-state
IDs for QV, geometry IDs for Q. distance_half_mse uses distinct pairs within
microbatch8. Distance semantics follow each arm's available input information.

Four arms times three seeds,400 updates, batch256 sampled without replacement
from360 train velocity states, explicit microbatch8 for higher derivative memory.
All scientific fits use online W&B stable resumable IDs. Selection every5
updates plus initialization is validation feature NLL, never clustering/AP.
gradient_norm is pre-clipping norm; clipping10. Missing unscheduled history
fields are blank. Selected update and selection NLL identify the same checkpoint.

Predictive rows are per horizon. value_corrected_mse subtracts value mean
noise variance. position_response_corrected_mse and (QV only)
velocity_response_corrected_mse use feature-normalized units, without extra
response-RMS scaling. The force-free control evolves the same velocities and
noise with zero forces. Its squared mean-feature difference is corrected by
variance of the PAIRED feature differences/B, not by target variance alone.
shuffled_velocity_corrected_mse permutes the79 neighbor velocity vectors in
each held-out input, preserving the central velocity and local speed multiset;
it measures reliance on spatial assignment of motion. These diagnostics are
not used for selection. Each ancestor contributes four equal-weight states.
Corrected scores may be negative. No nuclei-birth model is fitted here.

Clustering uses unchanged all64 uniform unrelaxed observations:74880 training
rows for fitting clusters and24960 held-out rows. QV has separate sampled-velocity
and mean-of-four-velocity-embedding views, with exactly the same fixed450K
draws across models. These are static transfer diagnostics with synthetic
velocities, not native MD phase-space evaluations. Q uses its geometry embedding.
K=7 MiniBatchKMeans (batch4096,n_init3,max_iter200,reassignment0), three recorded
seeds, no clustering-based selection. Many-to-one agreement=sum_k max_g n_kg/N;
one-to-one uses Hungarian assignment. ARI/AMI, occupied clusters, largest-cluster
fraction, exp(entropy) effective clusters, and ARI against the first clustering
seed accompany accuracy. Velocity views never multiply independent row counts.
All metric CSVs preserve these definitions and implementation hashes.

## Explicit three-hour pilot, 4 October 2026

The user stopped the large response-training campaign and restricted this chat's
experiments to 2–3 hours. The separate `three-hour-v1` recipe selects 12/4/4
train/selection/test ancestors by the recorded deterministic parent-index list,
preserving their original roles. There are two initial velocity fields per
ancestor, two training noise shots, two selection value shots, and two test value
shots plus two test derivative shots on disjoint streams. The resulting 24/8/8
phase states are not independent ancestors. Normalization of the physical
observables still uses the frozen earlier training-only reference; feature
normalization and distance/derivative scales are fitted to the pilot training bank.

All physical settings, 100/500/1000 fs horizons, full periodic cells, local80
inputs, four derivative directions and metric formulas above are retained.
Replica-variance corrections divide by the actual shot count (two here).
Q marginal distances use the two velocity-field means per geometry. Four matched
arms use one training seed, 200 updates, batch24/microbatch8: explicit deviations
required by the smaller bank and wall-time budget. This is a feasibility pilot,
not a conclusive clustering comparison or an estimate of seed-to-seed stability.

The all64 clustering rows and three clustering seeds remain unchanged. QV's
marginal view now averages two velocity embeddings; the sampled-velocity view
and all draws remain matched between arms. Initial-state sampling and noise seed
formulas are unchanged, allowing checksum-recorded reuse of existing full-cell
two-shot batches. Reused values retain their original producer identity.

No additional MEAM checks are run, as explicitly requested by the user.
The recorded earlier student symmetry check and prior execution authorization
are reused. A shared three-hour deadline includes collection, fitting and
evaluation; workers stop without automatic requeue. Partial receipts do not
establish completion.

## Compiled CUDA MEAM, horizons up to 0.5 ps

The `cuda-meam-500fs-20261004-v1` campaign is separately authorized to exceed
three hours. It restores the full original 90/15/30 ancestors, four fixed initial
velocity fields, 4 training shots, 16 selection value shots, and 16 test value
shots plus 8 independent test derivative shots. Four arms times three training
seeds use 400 updates, batch256/microbatch8. The all64 clustering population and
four-draw velocity-averaged view retain their original definitions.

Targets use only 100 and 500 fs: two marginal blocks of 128 fixed Fourier
features, hence 256 predicted coordinates. The architecture still exports z128.
There are two positional and two velocity directions, with the same units and
training/evaluation formulas. The fixed initial velocities and BAOAB/noise
ordering remain those of this velocity protocol; the separate `FastOracle`
recipe that resamples initial momenta is not used.

The teacher now uses the established `meam_gpu.FastAl.force_tangent` interface,
compiled analytic force/Hessian-vector products and CUDA neighbor/screening
construction. Value-only shots use `FastAl.force`. Force arithmetic is float32;
positions, momenta, tangents, feature evaluation and energy accumulation are
float64. Full native periodic cells are retained. Prior float64 response banks
are not mixed into this new bank; only their prepared input cells are reused.

Existing completed CUDA MEAM benchmark receipts and source hashes are pinned.
No additional MEAM validation is run, following the user's explicit instruction.
The existing small student derivative/symmetry check is retained for the
256-output prediction head. Supported Blackwell GPUs use the recorded CUDA
toolchain. Two independent single-GPU jobs and the current-allocation helper
share query/fit locks; each queued job has 24h and at most two automatic resumptions.
The earlier three-hour pilot remains a separate frozen campaign.

## Float32 state revision, 4 October 2026

At the user's explicit request, `cuda-meam-500fs-float32-20261004-v2`
replaces the mixed-precision v1 campaign. The v1 workers were stopped and
their partial banks retained separately. The new configuration declares
`state_dtype=float32` and `force_dtype=float32`. Initial positions/velocities,
momenta, coordinate/velocity tangents, periodic boxes, thermostat noise,
observable normalization and Fourier projection all use float32 on the GPU.
Saved local starting states, values, force-free controls and responses also use
float32. CPU-prepared parent data and initial random-map generation preserve
their original precision and are cast before GPU use. The unchanged MEAM
backend retains its scalar total-energy reduction in float64; forces and
force tangents returned to the integrator have the state dtype.

Fourier maps keep the same seed and CPU draws, rounded to float32. Thermostat
draws use the declared state dtype, so matching integer seeds do not establish
identical sample paths across precision revisions. All four model arms share
the new float32 bank. No mixed-precision targets are reused, and no additional
MEAM validation is added. Ancestors, horizons, shots, seeds, model capacity,
objectives and evaluation definitions remain those of the full 0.5 ps campaign.

## Native-velocity, source-temperature, float64 integration revision, 6 October 2026

`native500-f64-20261006-v1` retains all original 135 geometries and 90/15/30
source roles; uses exactly one actual simultaneous full-cell MD velocity field
per geometry, read in source atom order then gathered into the existing parent
order. Verify source manifest, frame, atom IDs, original Q hash and timeline.
Source velocities stored in A/ps are divided by 1000 in float64 to A/fs.
No velocity redraw, temperature rescaling, COM removal, relaxation or source
resplit is applied. Source Q,V were quantized float16; higher integration
precision cannot recover those missing digits. Metadata records both precisions.

Use fixed full periodic 70,304-atom cells, unchanged FastAl Lee2003 Al MEAM,
BAOAB step 1 fs and damping time 1000 fs. Physical thermostat temperature is
each source's original setpoint (400/450/500/510/520 K), taken from frozen source
audit metadata and never concatenated to encoder/predictor input tensors.
The user's October 6 precision instruction supersedes the October 4 default
for this new run: states Q,V,p, tangents, noise, observables and saved response
targets use float64; force/energy algebra and neighbor topology use float32,
and scalar energy accumulation remains float64. Force/HVP outputs return in
the float64 state dtype after differentiation through the float32 energy cast.
Neural model inputs, parameters and optimization remain float32. No new MEAM
validation is introduced; existing force/HVP implementation receipts are pinned.

Targets remain 128 Fourier coordinates at 100 fs and 128 at the PRIMARY 500 fs
horizon, with the same fixed CPU-generated map and earlier train-only physical
observable normalizer. Independent thermostat noise is the only shooting
randomness; native V is held fixed. Training: four shared value/response shots;
validation: sixteen value shots; test: sixteen value and eight disjoint response
shots. A distinct noise-seed namespace prevents reuse of old trajectory banks.
The native bank has 135 phase states, not 540; it does not add ancestry.

The four Q/QV × value/response arms and three fitting seeds retain the same
architecture, objectives, derivative scales, validation NLL selector and 400
updates. Batch256 is sampled WITH replacement from the 90 training states,
explicitly declared; microbatch8 is retained for higher-derivative memory.
All repeated input indices are excluded from distance pairs. For BOTH Q and
QV, distance targets use the four stochastic shots conditional on the saved
Q,V and source thermostat. A velocity-marginal distribution is not identifiable
from one native V per geometry and is not claimed. Q observes only coordinates;
its available information is intentionally restricted relative to the teacher.
No synthetic copies of the native velocity are treated as independent fields.

All64 clustering retains exactly 74,880 training and 24,960 held-out uniform
observations, the fixed descriptor labels and all existing metric formulas.
For each patch, recover the original nearest80 atom identities (reuse released
neighbor IDs where present, otherwise the original periodic nearest-neighbor
query), require exact equality to the released coordinates, and gather actual
same-frame velocities. Store these inputs in a checksum-bound cache; shards
retain source provenance. QV reports `native_velocity`; Q reports `geometry`.
There is no sampled or velocity-averaged view for this protocol. K=7 and all
three seeds, many-to-one/one-to-one/ARI/AMI/collapse diagnostics stay unchanged.
Clustering is never a training/checkpoint selector. Historical sampled-velocity
scores are not relabeled as native-velocity evaluations. Frozen old artifacts
retain their original metrics, hashes, precision and settings.

No crystal-specific training loss, temperature/time input, reconstruction branch
or AP selector. Scientific fits use online W&B with stable resumable IDs;
frozen evaluations stay local and update only their recorded training run.
A separate nine-hour campaign deadline includes queue wait and all stages,
with no automatic extension/requeue. Incomplete collection, fitting and all64
evaluation remain explicit. This named same-cohort follow-up does not cover
dense/shooting/million-atom/relaxed Al populations. It changes damping, initial
velocities, thermostat temperature and precision together; a gain alone cannot
attribute effects to any one change or establish native NPT equivalence.
