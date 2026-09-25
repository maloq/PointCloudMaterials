# Spatial hierarchy v1 metrics

Physical, hazard, source-weighted AP, frozen-probe, retention selection and spectral
metrics follow the frozen definitions in `robust_onset.md`, captured alongside
this family in the implementation contract. All target labels still concern the
same inner 8 Å region. The primary reference is L-local, not the prior B-onset arm.

Input features use current observed coordinates only. Three nested regions have
radii (4,6,8), (8,10,12), or (8,12,16) Å. Each uses six Gaussian radial channels,
cross-radial l2/l4 harmonic Gram contractions (21 each), and smooth count. A quintic
taper is one below 0.8 R and goes smoothly to zero at R. Moments/counts use fixed
normalization 128*(R/8)^3. These are geometry summaries, not crystallinity labels.
Context feature standardization uses source-balanced fitting observations only.

`outer_context_swap_rms`: select eight development observations per root with a
fixed seed. Within each temperature, select a different development root's patch
as donor. Keep every coordinate inside 8 Å exactly and replace only the 8–16 Å
atoms. Rebuild context and local inference, then compute
sqrt(sum_i w_i ||z_swap_i-z_clean_i||²/(2 V_fit)), with root-balanced weights and
V_fit the trace of the selected encoder's clean fitting covariance. The donor
mapping and observation indices are exported. This is a diagnostic intervention,
not physical evolution or a selection criterion. A large response need not be good.
The local-only arm should have zero response up to floating point computation.

`clean_input_replay_rms` uses the same formula comparing clean inference rebuilt
from the wider cache against cached clean fitting-pipeline features. This catches
input/normalization drift; it is not a perturbation sensitivity metric.

Noise perturbs noncentral coordinates in the common 16 Å candidate patch and
rebuilds both regional summaries and the inner graph. RMS percentages use each
clean patch's mean center-to-twelve-nearest-neighbor distance, as in robust_onset.
The outer candidate set is fixed, with a smooth support boundary. Temporal analysis
uses exact 0.75 ps pairs from the existing dense chart with newly retained 16 Å
inputs; the previous 10 Å native inference cache is insufficient for this study.

This is a 45-root reused screen with 18 positive development windows and one
training seed. Paired bootstrap intervals measure root uncertainty only.

## Inherited definitions, captured with this protocol

# Robust onset v1 metrics

Primary: source-weighted average precision for sustained local first onset by
the explicitly configured `primary_horizon_ps` (3 ps for new experiments; the
original frozen study used12 ps), among causally eligible noncrystalline anchors. Three observed noncrystalline
frames establish eligibility; three future crystalline frames confirm an event.
The original independently produced labels are reused. Event bins end at
0.75, 3, 6, 9, 12 ps; bin 5 means survival through 12 ps. All examples have the
required follow-up. This is finite-horizon local risk, not a committor.

`source_weights` gives every independent root equal total weight. No event
oversampling is used for fitting probabilities or evaluating precision. AP is
sklearn weighted average precision, not trapezoidal PR area. Weighted Brier is
mean squared probability error. Hazard NLL is the first-event/survival likelihood.
The 5% FPR threshold is chosen on tuning negatives with complete ties; report
the actual resulting development FPR, recall and precision.

Primary checkpoint: tuning AP at `primary_horizon_ps` maximum among checkpoints whose tuning geometry,
current-order and future-increment MSEs each remain <=1.05 times their calibrated
initial value. Selection never reads development outcomes. Initial step zero is
eligible and explicitly reported. Separate frozen linear/MLP probes use the
existing tuning-NLL criterion; they must not be mixed with the joint head.
Observed/relaxed descriptor controls are zero-padded from 89 to 128 dimensions;
the temperature-only control supplies 128 zero features plus temperature.
All three use identical linear/MLP probe recipes and source splits.

Physical targets are fit-standardized. Training geometry loss averages the radial17,
l2 Gram36 and l4 Gram36 blocks equally. `physical` JSON diagnostics report ordinary
dimension-mean standardized MSE (89 dimensions for geometry, 8 current order,
24 concatenated 3/9/12 ps increments), source-balanced. Thus training geometry
loss and exported dimension-mean MSE have deliberately distinct weights.

Define local spacing d_i = mean of the twelve smallest positive distances from
the tracked center to other atoms in the clean patch. Perturb each noncentral atom
by independent N(0, (f*d_i/sqrt(3))² I), retaining the center exactly. The expected
3D displacement RMS is f*d_i. Let q_i be mean squared realized 3D displacement
over noncentral atoms. Report:

- `input_rms_A = sqrt(sum_i w_i q_i)`;
- `input_rms_percent_of_spacing = 100 sqrt(sum_i w_i q_i/d_i²)`;
- `mean_spacing_A = sum_i w_i d_i`;
- `embedding_noise_rms = sqrt(sum_i w_i ||z'_i-z_i||²/(2 V_fit))`, where V_fit is
  trace of source-weighted clean fitting covariance of this selected encoder.

The percentage is RMS of local relative displacements, not ratio of two means.
Always distinguish it from normalized **embedding** Noise RMS. Primary fraction
is 0.005 (0.5%); all four perturbation scales remain in JSON. One deterministic
independent noise draw per observation, eight observations per development root.
Each model receives the same random fields; relaxed inputs have their own spacings.
Rebuild edges/angular/radial quantities, preserve the finite original candidate
set, and do not requantize perturbed inputs. Noisy inference cannot fetch an atom
absent from the original candidate set; finite-support limitation is explicit.

Dense trajectory metrics use only exact 0.75 ps differences, no interpolation.
They reuse `trajectory_stability.spectrum.analyze`: source-balanced jump RMS and
quantiles normalized by the historical clean reference covariance; uncentered
increment/velocity spectra include drift, centered fluctuation spectra remove it.
Participation rank is (sum eigenvalues)²/sum eigenvalues²; d95 is the smallest
number of principal directions retaining 95% of spectral energy. Paired-corpus
dataset rank and dense-chart dataset rank are different populations, both named.
They are linear dimensions, not estimates of nonlinear intrinsic dimension.

Comparisons export AP3, AP6 and AP12, with primary-horizon Brier, prevalence,
recall/FPR, selection and paired intervals explicitly labeled. Alarm thresholds
are chosen separately on tuning negatives at each horizon. Changing the primary
horizon does not change the full five-bin hazard NLL or physical targets.

Paired AP intervals at the configured primary horizon resample complete development roots within temperature,
including multiplicity, with 2,000 shared bootstrap draws. Draws with no positive
weight are excluded and the valid count reported. These intervals do not quantify
training-seed uncertainty. The 45-root paired development assay is reused, and the
dense chart may overlap fitting roots; it is a descriptive diagnostic only.

Smooth-AP is a *training surrogate*: weighted positive-average precision with
sigmoid score comparisons, full self mass, full fitting risk set and temperature
0.01. It is evaluated on primary-horizon cumulative probabilities every 16 updates,
with positives defined by the corresponding onset bin, and is
not exported as empirical AP. It does not inherit the SOAP optimizer's guarantees.
