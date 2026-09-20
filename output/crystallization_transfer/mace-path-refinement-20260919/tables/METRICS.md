# Structural-path refinement metrics and selection, version 2

All event, physical trajectory, source-weighting, calibration, timing, CRPS and
spatial metric calculations are inherited unchanged from `crystallization_paths.md`.
The same 150 sources and frozen parent MACE are used. The future-center cache is
reused only after checking its pinned plan hash, source inventory, checkpoint,
scale, time grid and all per-source completion/checksum records.

Thirty targeted screens share one seed and a 12-epoch cap. Selection now uses 128
fixed windows per selection source (up to 1,920 total); the earlier screens used
64. The two selection-score populations must not be compared as paired observations.
Training is exact shuffled source-weighted passes. Checkpoint selection is the
lowest dense event Brier score, evaluated using free rollouts after each epoch.
Stop after four unimproved epochs (six for diffusion), after at least four epochs.
Saved actual updates, epochs and samples account for early stopping. Each method's
best eligible screen is retrained with a 36-epoch cap and eight-epoch patience.
Eligibility requires selection physical MSE within 10% of the best in that family;
physical MSE is the equal mean of packet, bond-order and crystallinity standardized
MSE, then equal-source weighted. Select lowest Brier among eligible screens.
This gate uses no test metrics. Every trained fit retains its selected checkpoint,
raw calibration/test predictions and the unchanged held-out metrics.

Direct/deterministic AR controls retain the original block-balanced structural MSE
plus exact dense event NLL. Ablations change one declared setting: LR1e-4,
dropout0.1 together with weight decay1e-3, structural loss coefficient0.25, current
state anchoring, 48ps history, or observed motion. Anchoring retains the actually
observed current MACE embedding, decodes the other present physical channels,
and sets current crystallinity to zero from the existing liquid-origin eligibility.
The forecast is an offset from that current state (not accumulated increments);
an auxiliary present-reconstruction loss has weight0.25. No future state is input.

The motion ablations additionally observe 43 current-frame packet channels:
speed and signed radial-velocity summaries (packet80:112), plus velocity-related
moments (packet117:128). These use existing relative velocities. They exclude PTM
and future labels. Scaling uses the already training-only physical target moments.
Their comparison changes available information, not just predictor architecture.

Deterministic AR controls use a fixed six-epoch teacher-forcing decay, independent
of total budget; alternatives use two epochs or no teacher forcing. Gaussian AR
uses teacher forcing throughout **training** to optimize conditional likelihood,
and samples its own states at validation/test. Diagonal and rank8/rank16 covariance
variants use `(diag(exp(2 logstd)) + factor @ factor.T)` within each future state;
innovations across states propagate through the recurrent dynamics. Gaussian
structural NLL averages across the 32 times and divides by265 channels; event NLL
is added without that division. The diagonal objective omits its constant. This
likelihood protocol intentionally differs from the original block-weighted scheduled
sampling surrogate; differences in its training loss are not quality metrics.

The free mixture control retains the original four-component weighted objective.
An ablation triples the structural weight. Event-stratified mixtures use training
outcome categories: onset by9ps, 9–48ps, 48–96ps, or survival beyond96ps; the two-mode
control separates onset/survival. A context-only gate predicts category probability.
Each component's 129-way event distribution is masked to its own support; their
mixture defines the event CDF. Training uses gate cross-entropy plus conditional
event NLL plus that labeled component's structural Gaussian loss. Future categories
are training labels only. Inference samples predicted gates, never true categories.

Diffusion refinements use an exact zero-terminal-SNR schedule. Velocity prediction
uses `v = sqrt(alpha)*noise - sqrt(1-alpha)*clean`; reconstruction is
`clean = sqrt(alpha)*noisy - sqrt(1-alpha)*v`, avoiding inverse-small-alpha
amplification. A full-dimensional noisy skip removes the old rank bottleneck.
The clean-prediction alternative directly predicts the clean path. Twenty percent
of training examples use the pure-noise terminal level, with other levels sampled
uniformly. Structural/event diffusion losses retain the block averages, with an
event-weight3 ablation. Width128/384, 16/32 DDIM steps and EMA0.999 are tested.
EMA averages parameters each optimizer update, copies fixed buffers, and is used
for checkpoint selection and inference when enabled. Model, EMA, optimizer, stale
validation count, stop decision and RNG states are checkpointed for resume.

Generated event curves are projected onto their nearest valid absorbing -1/+1
step curve (including survival), using all128 times. Minimize squared error;
equal-energy ties prefer the latest onset/survival. This replaces the old first
positive crossing, which could turn one noisy excursion into an early event.
Finite-ensemble training-prior smoothing is unchanged and explicit.

These are exploratory follow-ups on a previously inspected test cohort. No test
metric is used to select new settings, checkpoints or promotions, but these
results are not a fresh untouched confirmatory test. One training seed, no claim
about training-seed uncertainty. Historical exported definitions remain frozen.


Table export: 2026-09-19T15:20:27.561683+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
