# Causal MACE physical-state protocol v1

This is a new native scalar/vector/rank-two MACE architecture initialized from
scratch. It does not inherit the old scalar MLIP weights or their normalization.
All variants use identical source splits, anchors, physical targets, parameter
shapes, initialization seed and update budget. A: snapshot/present only;
B: snapshot/present+future; C: B with relative velocities; D: C with causal
atom history; repeated_anchor: D trained on repeated current frames; E: D
continued with constrained slowness. Disabled branches remain allocated to keep
parameter shapes and initializations matched, but their active capacity differs.

Attention logits are bounded to [-2,2] to prevent extreme learned odds defeating
the vanishing age envelope near its boundary. Atom features are divided by
sqrt(1 + mean(h²)) before and after temporal updates. The Euclidean irrep norm
is rotation invariant; the additive one retains amplitude information while
bounding high-order product growth. This is an internal feature normalization,
not population whitening or an embedding variance objective.
The sequence adapter verifies the retained position/velocity arrays and label
hashes. Inputs contain observed frames only. Source windows match the requested
physical times exactly. All atoms retain actual producer IDs. Exact backwards
space-time ancestor closure supplies computational context; smooth pooling
supports [0,3], [5,7], [7,9] A in the pilot. Context radius and tensor width are
separate config ablations. The wider summary is an atom-level pooled context,
not a separate coarse graph. The oldest frame at exactly 2.25 ps has zero age
weight, including inside temporal normalization. Retained labels span only 6 ps;
the pilot explicitly appends 6.75 ps of native-cadence follow-up using the same
physical label producer. Original labels/inputs remain immutable; extended arrays,
selected frame indices, source and target hashes are retained. The pilot uses
0.75, 3 and 9 ps futures, the latter two exceeding observed history.

Fixed physical targets reuse `mace_velocity.data.labels`: 16 group geometry
observables, 16 H0 bins, 64 H1 bins, 64 H2 bins, 6 even motion and 3 signed motion
values (169 total). Target definitions, including hard nearest-80 topology
membership and persistence filtering, are unchanged. This workflow does not
claim those targets are smooth. qbar6 is column 4 of the producer's GROUP_NAMES.

Normalization uses training sources only. Each source contributes equally to
the first and second moments of its current and future endpoint target rows.
TDA scales use each homology block's RMS standard deviation; other columns use
their own standard deviation. All scales have a declared 1e-6 floor and constant
columns are recorded. Endpoint residuals are standardized; physical deltas use
the same scale with zero mean. Path targets are the mean and population standard
deviation of qbar6 on every retained frame strictly after the anchor through the
future endpoint. Path mean uses the fitted qbar6 mean/scale; path SD uses its scale.

Present and future endpoint losses each average the six blocks equally after
averaging channels within each block; future losses average horizons equally.
The future term also includes separately predicted physical deltas (weight .25)
and path summaries (weight .1). All heads read the single exported embedding.
Future/hazard heads additionally receive temperature/1000 and actual lag in ps.
The optional diagonal Gaussian head reports positive SD = .02 + softplus(raw),
Gaussian NLL including 0.5 log(2 pi), and one-SD empirical coverage (reference
68.27%). It does not model joint temporal covariance or multiple modes.

Local events refer to the first confirmed sustained episode within the retained
sequence, not an unobserved lifetime history. They reuse the nearest-80 tracked-center PTM assay, with FCC/HCP/BCC
crystalline at RMSD cutoff .1. A sustained event requires the configured number
of consecutive retained frames; onset is the first frame, not confirmation time.
The risk set requires a noncrystalline anchor and no already-confirmed sustained
episode in the observed prefix. Future frames define labels only. Hazard bins
are (left,right]; an event contributes -log h_k and -sum_{j<k} log(1-h_j).
A censored sample contributes only fully observed bins whose endpoints have
enough retained follow-up for confirmation. No transition oversampling is used:
training samples a source uniformly, then one of its center/anchor windows
uniformly. Hazard NLL averages eligible batch examples; no-risk batches contribute
zero hazard loss. Brier scores include only known event/censor outcomes at each
bin. They are finite-horizon local transition risks, not committors.

Alarm thresholds are fitted on validation only, separately per cumulative horizon.
Choose the smallest score threshold whose source-weighted false-alarm rate is at
most .05 in the pilot. Include a threshold strictly above the maximum score as
the no-alarm candidate. Without both validation classes, that operating point is
undefined. Event tables weight each contributing source equally, then each known
example equally within source. They include average precision, AUROC, Brier score,
mean risk, observed event frequency, precision, recall and false-alarm rate. Raw
detected/missed counts are retained. Timing uses the first bin endpoint crossing
the selected horizon's threshold and measures distance to the observed onset bin
interval; it does not invent continuous sub-cadence timing. Timed recall includes
misses in its denominator and accepts interval distance <= first-bin duration.

CSV source rows average examples within each source. Summary rows give an equal
mean over contributing sources; 95% intervals resample whole sources with
replacement (500 draws in the recipe). A single source has undefined intervals.
`low_order` means current group mean qbar6 < .30; it is not a phase assignment.
Persistence predicts each future physical endpoint equal to the measured current
physical target. It uses no model-fitted current reconstruction. Present, future
and delta errors are standardized per-block MSE. NLL/coverage average all 169
endpoint channels; report them separately from equal-block MSE.

Temporal pairs match source, tracked atom ID and the exact configured lag.
The low-order population requires both endpoints low-order. For each source and
population, J = sqrt(mean ||z(t+lag)-z(t)||^2 / (2 sum Var(z))), using population
variance across that source's selected centers/anchors. Zero variance is undefined,
never reported as a good zero jump. Quantiles refer to raw Euclidean increments.
Summary J is the equal-source mean, not a single global variance ratio. Shared
history can reduce adjacent jumps; jumps alone are not a success criterion.
The 0.10 criterion applies to the equal-source mean J at 0.75 ps in the
low-order population (both endpoints qbar6 < .30). Each split exports a separate
criterion record; missing pairs or variance make it undefined. It never overrides
physical information constraints or certifies a model as scientifically successful.

A-D selection uses present block-mean MSE plus weighted future endpoint, delta,
path MSE and hazard NLL with the training coefficients (A uses present only).
E starts from a matched D checkpoint, uses a fixed equal-source mean within-source
training-reference covariance trace in its slowness loss, and minimizes
validation J only among models satisfying every present, endpoint and delta
per-block validation MSE (both all and low-order populations), path MSE, and
defined hazard NLL/Brier <= reference*(1+relative_tolerance)+absolute_tolerance.
The pilot tolerances .05 and .005 are engineering choices, not demonstrated
acceptable scientific information loss. Test sources never select checkpoints.
No bending, direction-rank, whitening, frozen-teacher or latent-prediction loss
is used. Checkpoints include strict state, optimizer/RNG, config, normalization,
cache identity and source hashes. Saved prediction arrays pair physical targets,
source IDs, tracked atom IDs and anchor times with the exported state.

Frozen probes fit a separate linear decoder for each horizon and a nonlinear
decoder with configured hidden width, all on training sources with validation
selection. The state-sufficiency pair adds an independently trained raw-history
encoder to frozen z. `state_history` receives the original observed atoms;
`state_constant` receives one fixed training history for every example. Both
have identical parameter count, initialization, batches and update budget. Only
frozen z varies in the constant-history control. Their physical score difference
tests accessible information omitted by z; absence of a gain is not a proof of
sufficiency. Diagnostic branch embeddings are not exported representations.


Table export: 2026-09-16T18:43:28.999173+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
