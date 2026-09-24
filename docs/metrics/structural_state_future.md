# Two-seed physical/future encoder comparison

All per-seed calculations follow [structural_state.md](structural_state.md),
including its explicit v3 extension. Tables physical.csv, neighbors.csv,
training_heads.csv, comparisons.csv and embedding_geometry.csv concatenate those
exports with an encoder seed column; no extra normalization or pooling occurs.
The campaign uses two encoder initialization/sampling seeds and one fixed probe
seed. Initial encoder state (except the two fitting-normalization buffers) must
match bitwise across arms. Fitting-normalization buffers use rtol1e-6/atol1e-8;
encoder configuration, target-construction receipts and head ridge penalties
must agree. Feature comparison uses relative Frobenius error<=1e-5 and maximum
absolute error<=1e-4. Initial head predictions must have RMS error<=1e-4 and
maximum error<=1e-3 in standardized target units. All measured differences and
initial checkpoint hashes are recorded in technical/status.json.

This audit replaced a failed pointwise rtol2e-4/atol3e-5 check on23September.
The original failure affected one of368640 feature values. Four repeated GPU
forwards of the exact same initial checkpoint showed maxima4.37e-5–5.64e-5 and
relative RMS2.78e-6–2.80e-6; cross-arm errors were of that same size. Direct
checkpoint comparisons found identical encoder weights, while fitting-only
normalization/calibrated heads differed by floating-point reduction/solve noise.
The repeat receipt is technical/cuda-initial-repeat-audit.json. This is a
post-training numerical provenance correction, not a change to any scientific
metric, success threshold, encoder, prediction or checkpoint. Original exported
definitions are preserved in technical/initial-report-version.
Cache/record, completed4096 checkpoint and feature hashes remain verified before
any completed seed contributes. Pending seeds are explicit.

onset.csv replays 12ps predictions on exactly the same development indices,
source IDs and event bins across all arms, representations and both seeds.
Each source has equal weight; rows within a source share its weight uniformly.
Average precision uses all ties and sklearn's noninterpolated convention, verified
against sklearn at the point estimate. Brier uses raw cumulative risk; binary
log loss clips risk to[1e-7,1−1e-7]. Event NLL uses five conditional logits: each
survived bin contributes softplus(logit), an observed event contributes
softplus(−logit), and a non-event survives all five bins. Replayed point metrics
must reproduce original producer metrics within rtol1e-6/atol1e-8.

Source uncertainty uses2000 paired bootstrap draws, seed20260923. Whole independent
sources are sampled with replacement within temperature, retaining stratum sizes.
A selected source contributes all its eligible rows with its original within-source
weights. The same draws are used for all models and both encoder seeds.
Percentile95% intervals omit undefined no-positive AP draws and retain valid_draws.
paired_onset.csv is candidate minus reference in original metric units: positive
is better for AP, negative for Brier/log loss/NLL. mean_seed_effects.csv averages
these paired differences across completed seeds in each draw. It conditions on
these particular trained seeds and does not estimate seed-population uncertainty.
No correction is made for multiple exploratory comparisons.

mechanism_effects.csv contains each predefined contrast and seed. All errors are
noncrystalline development errors. worst_current_error_change_percent is the
maximum of100*(candidate/reference−1) over native final radial/l2/l4-head MSEs
and frozen ridge relaxed radial/angular/l6/current-order MSEs. Retention passes
at<=2%. withheld_neighbor_error_change_percent averages the relative changes
of relaxed angular and l6 k5 pair discrepancies, with equal target-family weight.
future12_error_change_percent is the relative change in frozen ridge12ps
future-order MSE. The neighbor/future rule passes only with retention and a change
<=−1% in the respective metric. Support requires the same rule in both seeds.
These are predeclared practical screens, not significance tests. onset_ap_delta
and onset_brier_delta use the same contrast's12ps MLP point estimates and make
agreement or disagreement with physical metrics visible; they do not estimate
causal mediation. A larger rank is never a success criterion.

plots/onset-factorial.png shows the final exported MLP12ps AP and Brier for four
matched arms per encoder seed. It is a descriptive repeated-development figure.
Only18positive windows among643 development windows from15 sources are available
(10sources contain positive windows). Report source intervals, both seeds and
constant/descriptor controls; no result is an independent final-test claim.
