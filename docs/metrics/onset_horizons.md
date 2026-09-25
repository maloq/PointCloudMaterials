# Saved-prediction horizon review

Primary metric is source-weighted average precision for sustained local onset
within **3 ps**. Secondary horizons are6 and 12 ps. Event bins end at 0.75,3,6,9,12 ps;
an event by 3 ps has bin <=1, by 6 ps <=2, and by 12 ps <=4. Bin5 is survival
through 12 ps. The same eligibility and three-frame sustained-event confirmation
apply at all horizons. Confirmation can extend beyond the endpoint; those frames
define labels and never enter the encoder. A3 ps horizon is not a 3 ps minimum
warning time: events closer to the observation origin also count.

Replay reads the exact saved cumulative probabilities, corpus indices, source
identities and labels. Native integer source IDs must match the corpus producer;
they are then mapped through records to ancestry roots. Joint and native rows
must exactly equal the existing causally eligible development grid. Full-source
weighting gives every root equal total weight. AP is sklearn weighted average
precision; Brier is the corresponding weighted squared probability error.
No-positive AP is undefined, never zero. Replayed AP, Brier and prevalence are
checked against each original saved metric, with absolute tolerance1e-12 and
relative tolerance1e-10. Complete input files and config are fingerprinted.

This does not refit encoders/readouts or select new checkpoints. Historical
AP12-selected joint models remain labeled as such. Frozen probes retain their
original tuning-NLL selection; Epi checkpoints retain fixed pass counts.
Descriptor controls already contain these metrics but have no retained per-row
predictions; those values are explicitly labeled saved-metric reuse.

Paired AP3 intervals use 2000 shared temperature-stratified bootstrap draws of
whole development roots. Root multiplicity scales the original root-balanced
weights. Draws containing no positive windows are excluded, with their count
reported. Absolute AP and paired AP-difference percentile95% intervals condition
on the fitted model and this very sparse cohort; they do not cover seed or
model-search uncertainty. Report positive windows and contributing roots for
fit, tuning and development at each horizon. Final Epi rows average the two
predeclared seeds, not the best checkpoint or best development result.

All existing0.75 ps trajectory, noise and spectral-dimension definitions remain
unchanged. A shorter prediction horizon must not be confused with trajectory lag.
