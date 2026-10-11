# Frozen response representations for nucleus preappearance

This diagnostic retains the original fixed-Al64 isolated-establishment birth
cohort, identity7b3046cdebf539d81a9f50b4abb5629f1ddb1b27edd886e0888b7658f9f96dd9.
It uses the derived event-centered track, not the historical all64/legacy16
center population. Source roles remain90/15/15/30; only eligible original
histories contribute. The released1475 rows cover59/8/13/15 train/selection/
calibration/test events. Test has200 rows,40 positives,15 events and11 sources.
Centers and repeated observations are not independent events or sources.

Cases precede isolated establishments from the full-cell PTM lineage audit.
All eight original input frames have zero FCC/HCP/BCC PTM atoms in the whole
moving8A sphere, conservatively including atoms beyond the consumed nearest80.
The first bounded-window local appearance includes eventual birth-core atoms.
Four same-source/same-endpoint controls per case remain liquid throughout the
declared history and follow-up, at least6ps and through case confirmation.
Original retrospective site selection, label/exclusion definitions, inverse
event row weights and1:4 prevalence are unchanged. This is not prospective
natural nucleation risk, critical-nucleus identification or arbitrary atomwise
crystallization. Previously inspected held-out sources remain exploratory.

## Inputs and frozen representations

All twelve likelihood-selected models from the float32 FastAl100/500fs campaign
are used: Q-values, Q-response, QV-values and QV-response, three pretraining
seeds each. Encoder pretraining source roles are checked against the original
birth cohort. No encoder fine-tuning, new MD, relaxation or label-based
checkpoint selection occurs. New diagnostic probes remain local.

Every released patch is mapped to its actual source/frame/central atom. Rebuild
the periodic nearest80 neighborhood with the original producer's query rule and
require exact coordinate equality to the frozen birth bank. Read simultaneous
MD velocities for those exact atom identities, convert angstrom/ps to
angstrom/fs by dividing by1000, and retain float32 values. Historical raw
archives contain float16 positions/velocities; conversion does not recover lost
precision. Source manifests and extracted velocity-bank hashes are retained.
No random replacement velocities or finite-difference estimates are used.

Inference imports the exact campaign source, loads its full model strictly and
replays eight saved test-query predictions within2e-4 absolute error. Both the
128D embedding and the256D existing nonlinear prediction-head outputs are
exported in float32. The head remains in its recorded training-standardized
feature units; readout standardization is separately fitted on training data.
Q ignores velocities inside its frozen producer. QV consumes the saved velocities
relative to the central atom with the original fixed0.004A/fs normalization.
All models consume centered unrelaxed nearest80 coordinates,8A support,5A
edges and no extra halo. No temperature, age, explicit time, source identity,
future geometry, event metadata, species or material channel enters a model.

Two observations end0.75ps before appearance: current uses frame7; history8
uses frames0..7 at exact0.75ps cadence, spanning5.25ps. History packets concatenate
chronological features, mean, population standard deviation and last-minus-first
change. Encoders always see one frame; history enters only the diagnostic readout.
The same1475 rows are retained for every treatment. No model-specific row drops.

## Readouts and controls

Each model/space/observation gets CatBoost and logistic readouts:96 fits total.
Reuse the established birth recipe: CatBoost depth4, up to1000 iterations,
learning rate0.04, L2=10, selection Logloss early stopping with patience100;
logistic C in[.001,.01,.1,1,10], chosen by selection-source binary NLL. Logistic
mean and scale use training event weights, scale floor1e-4. Training sample
weights are normalized to mean one. Readout seed20260930 is fixed across arms;
it is distinct from the three original encoder-training seeds. GPU CatBoost
uses the established settings; four CPU fit threads per lane are an explicit
execution change from the earlier eight-thread setting.

Calibration fits a monotone affine map of clipped logits using separate
calibration-source binary NLL plus1e-4 quadratic regularization, slope[.01,20]
and intercept[-20,20]. The false-alarm threshold is the event-weighted95th
percentile among calibration controls; alarm means probability strictly above
threshold. The existing fit.scores and fit.calibrate functions define these
calculations. AP never selects models or parameters.

Structural controls reuse saved fixed-test current/history8 readouts for442
rich descriptors, rich/TDA MACE and VICReg MACE, plus the constant prior, from
temporal-site-20261001. Saved row IDs, roles, labels, event IDs and weights must
match exactly; hashes bind the retained predictions and calibration records.
These controls retain one original fitted seed. Their frozen historical
definitions are preserved. New tables apply the declared common metric
calculation to retained predictions without refitting or changing them.

## Scores, paired comparisons and uncertainty

All scores use original inverse-event weights normalized within each role.
NLL and Brier are primary proper predictive scores; weighted AUROC/AP are
diagnostics. Probabilities are clipped to[1e-7,1-1e-7] for scoring.
Calibration-locked recall and achieved FPR are reported on test, never tuned
using test. Raw and calibrated probabilities are both retained.

Within-pair AUC is the fraction of four controls below the case, counting ties
as half. Within-pair top1 is one if the case has maximum probability, divided
by the number tied at the maximum. Assign a pair's value to its five rows and
apply the original event weights, giving equal total contribution per event.
Chance values are0.5 and0.2. Top1 is case localization among five retrospectively
sampled candidates, not a prospective search across a full MD cell.

Uncertainty uses2000 whole-test-source bootstrap draws with seed20260930 and
shared draws across every treatment. Point estimates for an arm are means of
the three per-model metrics, not metrics of ensemble probabilities. Bootstrap
replicates likewise average the three metrics for each common source draw.
95% intervals use2.5/97.5 percentiles; seed_sd is the sample SD across fitted
encoder seeds. Neither is uncertainty from retraining or multiple-testing
correction; the seed draws do not increase the independent-source count.

Paired contrasts are response-minus-values for Q and QV, QV-minus-Q for
values and response, and each response-campaign arm minus each structural
control with the same readout/observation. Differences use the same source draw and average matched
seed metrics before interval construction. Negative NLL/Brier differences
favor the left arm; positive discrimination differences favor the left arm.
The bootstrap_scores implementation is shared with the established birth assay.

Final per-model metrics update the original twelve recorded W&B training-run
summaries without restarting runs or creating new online diagnostic runs.

Generated per-model feature caches use the shared six-entry leased cache;
checkpoints, fitted probes, predictions and metrics remain permanent. Exports
freeze this definition and implementation hashes. Shooting, native0.1ps,
relaxed and unfinished full-cadence nucleus searches remain separate protocols,
not silently pooled independent samples. The current dense release has only
train/selection ancestors and cannot supply a fixed-test birth comparison.
