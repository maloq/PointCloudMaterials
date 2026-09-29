# Continuous spatial distance

Target is the current periodic distance in Å from a probe atom to the nearest
atom in a reference crystal component of at least 64 atoms whose lineage has
already been confirmed at the observation frame. No future confirmation enters
the input or reference mask. No reference crystal gives infinite distance.

Each model returns a zero-inflated lognormal distribution. Context models return
a learned mixture of 25 shared patch distributions. A zero distance uses point
mass likelihood; positive distances below 64 Å use lognormal density per Å;
distance ≥64 Å (including infinity) uses survival likelihood at 64 Å. Censoring
does not turn infinite distances into exact observations at 64 Å. The primary
proper score is negative log likelihood. Its density units mean its numerical
value must not be compared to the historical six-category distance NLL.

Reported mean and median refer to min(distance,64 Å). Capped-mean RMSE and
capped-median MAE compare those predictions against the capped observed target.
They do not assert an exact geometric distance for empty/far-reference cells.
CDF Brier scores at 4/8/12/20/32 Å are weighted mean squared probability errors.
`distance.csv` gives each source equal weight within each recorded population.
Fixed at-risk test centers and controlled scan positions are reported separately.
Scan positions are repeated, non-independent observations of controlled paths.

Training and checkpoint selection use a predeclared equal mixture of the original
fixed at-risk population and uniformly sampled atoms at the same observation
frames. Within each half every source has equal mass and its observations equal
weight. Uniform sampling uses no crystal labels. There is no transition/distance
oversampling. Normalizers are fitted only to the original training sources and
are shared between treatments. The minimum selection NLL after epoch 12 selects
among 16 epochs. Test and calibration sources are never augmented for fitting.
Existing evaluation rows and paths are preserved. Model seed is 20260926.

Local MACE, vector, harmonic and symmetric predictors consume observed geometry
features from the same frozen encoder. `visibility_only` is a trained label-side
control given only two flags: reference crystal in the local and full context
patches. It is not deployable without reference segmentation and is not a matched
raw-input competitor. Similar performance would support visibility as sufficient
for that score, not establish a causal explanation for every learned feature.
No time, temperature, velocities, material IDs or relaxed geometry enter inputs.
Confidence/visibility tables use the separately frozen `spatial_confidence`
definitions. All fits use online W&B; no AP-based objectives or selection.
