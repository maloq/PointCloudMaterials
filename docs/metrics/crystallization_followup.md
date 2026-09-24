# Literature-guided frozen-MACE follow-up

Same 150 independent Al sources and eligible archived origins as the completed
relaxed reuse study; original MD first sustained local onset remains the label.
Both backbones are frozen MACE exports. Original versus relaxed changes the
checkpoint, input domain and historical local support together. It does not
isolate the causal effect of relaxation alone. One fitted seed throughout.

Checkpoint selection is **source-balanced integrated Brier over 0.75–12 ps**,
the mean of squared CDF errors at 16 real 0.75 ps steps. This differs from the
historical 96 ps checkpoint-selection objective. Every new control is retrained
under the same 12 ps selection rule. Forecasts still cover 96 ps.

The historical runtime keys `selection_brier` and `best_selection_brier` refer
to this 12 ps selection score for this family. Exported `dense_integrated_brier`
and `brier96` retain their historical 96 ps definition. `integrated_brier12` is
the new 16-bin mean, and `classification.12.0.brier` is the single endpoint score.

`short_weight=1` adds right-censored event NLL through 12 ps plus block-balanced
physical/embedding MSE at 3,6,9,12 ps to the unchanged full 96 ps training objective.
Events after 12 ps contribute survival through the horizon, not a false event at
12 ps. All input normalization is training-only and source-balanced. Future state
normalization and source weighting retain the parent producer definitions.

The runtime also exports the original path, classification, timing and spatial
metrics described in `structured_context.md` and `crystallization_paths.md`.
Short AP uses the existing five-bin hazard conversion (with clipping at 1e-7),
at .75,3,6,9,12 ps. Recall uses calibration-source thresholds targeting 5% FPR;
actual test FPR is reported. Detected-event timing is conditional on detection;
missed and positive window counts must accompany it. Overlapping windows are
not independent events. Irregular origins do not define continuous-time exposure.

The new `restricted_time_mae12` is source-balanced absolute error between
E[min(T,12)] and observed min(T,12), including non-events. For the discrete
upper-endpoint convention, E[min(T,12)] = .75*(1 + sum_{j=1}^{15}(1-CDF(.75*j))).
This is an error in restricted expected time, not exact timing for censored events,
and may be dominated by survivors. It supplements Brier and missed-event metrics.

Pairwise Brier gain is control minus candidate on identical test row IDs and
event labels. Average each source's window gains, then average sources equally.
Intervals resample whole sources 1,000 times. They quantify source uncertainty
conditional on the fitted seed, not training-seed uncertainty. Multiple exploratory
contrasts are not familywise significance tests. These historical test sources
have informed earlier hypotheses; confirm winners on untouched sources later.

No PTM labels, future states, future relaxation, interpolated observations or
future-confirmed events are used as features. Absolute simulation time is an
observed condition in the reference and explicitly removed in one diagnostic.
