# Fixed-confidence spatial warning

Probability always means P(distance to the nearest atom of an already confirmed,
at least 64-atom reference crystal component ≤ R), for every R=4/8/12/20/32 Å.
It is not the probability of future crystallization or of any crystal anywhere.
Thresholds are strict >0.5, >0.75, >0.95, without fitting or test-set selection.
Both instantaneous alarms and two-consecutive-observation alarms are exported;
the latter is the primary rule inherited from the spatial-approach study.

`alarms.csv` reports the distance at the first alarm. Conditional median/p10/p90
exclude missed paths; detection counts and misses are mandatory alongside them.
Recall at D divides detections at distance ≥D by ALL test approach paths. False
alarm rate divides alarmed test away paths by ALL away paths. These pooled rates
are not source-bootstrap estimates. Away paths have distance >32 Å throughout;
this does not measure false alarms on near-miss or tangential paths. Path sampling
is controlled, nearest-atom snapped, and endpoints may lie inside crystal.

Visibility at alarm is true if ANY contributing observation contains reference
crystal, or respectively any FCC/HCP/BCC PTM-labelled atom, in the actual local
or 25-patch input. Early-clear counts require distance >8 Å and every contributing
observation clear. `visibility-only.csv` applies the same alarm rule directly to
label-side presence flags. This is an oracle diagnostic using full-cell reference
labels, not a trained deployable baseline. PTM labels themselves may use neighbors
outside a patch. Visibility association does not establish causal sufficiency.

`confidence-reliability.csv` reports held-out reliability separately for original
at-risk centers and controlled scan positions. Each source has equal mass before
subsetting; precision and mean confidence normalize the retained mass. Coverage
is retained mass divided by subset mass. Thus a score >.95 is not claimed to be
95% accurate. Clear-context subsets are shared across models, including local
models. Scan-position prevalence is artificial and repeated positions are not
independent samples. Empty subsets/exceedances produce blank, never zero.

All five models and every radius/threshold are exported. No best threshold is
selected. Inputs are checksum-verified and row/path alignment is checked. These
are spatial diagnostics in frozen snapshots, not temporal lead times or evidence
for predicting unseen nucleus birth. Existing producer artifacts are preserved.
