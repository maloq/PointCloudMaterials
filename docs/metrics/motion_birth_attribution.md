# Motion birth attribution diagnostics

These diagnostics preserve all 4,905 rows, original roles, 1:4 case-control sets,
physical-event weights and exact 1/3/6 ps inputs from the pinned unfiltered birth
cohort. They never remove observations based on crystallinity or neighborhood.
The positive fraction is imposed at 20%; these probabilities and AP do not describe
a prospective scan of arbitrary liquid sites. Previously inspected held-out data
make this an exploratory explanation study.

Integrity checks reconstruct every Q,V input from its raw trajectory and atom ID,
check exact lead times, first-member identity, same-source/frame pairing, disjoint
ancestor roles, exact coordinate overlaps and structural-pretraining ancestry.
Absence of exact overlap is not proof against all possible near-duplicate leakage.

Physical features use the existing 442 geometry/bond-order/CNA/TDA descriptors of
the exact nearest-80, radius-8A input. Nine instantaneous velocity summaries and
four fractions of relative Cartesian entries lying on 1/64,1/32,1/16,1/8 A grids
audit motion and quantization cues. No explicit temperature, age, time, source ID
or future label field is a predictor input. Existing full-cell PTM component
labels (minimum four atoms) measure current component membership/interference;
they are audit metadata, not neural inputs or physical-control predictor inputs.
A 64-atom component is not necessarily a persistent established nucleus.

Frozen physical diagnostic readouts reuse the same four training ancestors for
inner selection and refit on all original training ancestors. Linear C and tree
iterations are selected by binary NLL. AP, AUROC, NLL and Brier use event weights
and 500 whole-ancestor bootstrap draws. The existing sklearn-equivalent weighted
AP/tie calculation is unchanged. Feature group selection is declared in the
config, not selected by held-out AP.

Input/readout interventions and surrogate explanations at 3ps are conditional
sensitivity diagnostics, not causal physical simulations. They use frozen model
checkpoints and frozen readouts. Unchanged inputs must reproduce stored model
predictions before explanations are accepted. Feature groups can be correlated;
permutation and SHAP attribution need not uniquely identify a mechanism. A
surrogate is interpretable only to the degree of its held-out fidelity.

The declared 3ps interventions zero/reverse the motion code, permute structural
or motion codes within matched five-center sets, normalize each first-shell mean
radius to the training median, or add independent 0.01A Gaussian position noise.
Latent-future predictions use the original 32-row ordering, eight particles and
identical noise seeds for every intervention. Zero/reversed motion codes equal
the centered student's zero/reversed velocity encoding. Code permutations are
representation ablations, not physical MD continuations. Rotation of Q,V on 64
inputs checks invariance. Twenty-five within-pair permutations of frozen scores
provide an evaluation sanity control. No test labels select an intervention.

Physical surrogate regressors predict the frozen structural+motion CatBoost
readout's logit at 3ps, using all 455 present-state descriptors. Their iterations
are selected by RMSE on the same inner training ancestors, followed by refit on
all training ancestors. Weighted held-out logit R2/RMSE and probability
correlation report fidelity. Weighted mean absolute TreeSHAP values are grouped
by geometry, bond order, CNA, TDA, kinematics and quantization; the 20 largest
individual values are also exported. These are surrogate associations with the
neural readout and should not be represented as unique causal neural mechanisms.

The separate frozen-interventions-v2 export permits measured numerical replay
variation. A 256-row repeated-execution audit found mean future-law probability
differences about 0.00041 between identical runs, comparable to differences
against the saved predictions, with relative code differences about 0.000019.
The v2 checks bound mean/99th-percentile probability discrepancies at
0.0001/0.005 for structural and structural+motion readouts, and 0.002/0.02 for
future-law readouts; every actual maximum is also recorded. This accounts for
float32 reduction differences amplified by student subtraction and BF16 latent
integration. Tiny intervention differences must not be interpreted as biological
or physical effects exceeding this numerical resolution. The CPU controls and
surrogate tables retain their original v1 implementation snapshots.

The separate embedding-interventions-v3 export limits attribution to structural
and structural+motion readouts for all eight models. V2 completed both VICReg
joint models but halted on the adapter future-law replay (mean probability
difference 0.002007, 99th percentile 0.01738). Its partial results and producer
are preserved; no further tolerance relaxation is used to claim future-law
attribution. V3 evaluates the two reliably replayed embedding spaces and does
not compute stochastic latent futures. The underlying cohort and checkpoints
are unchanged.
