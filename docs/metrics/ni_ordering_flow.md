# Ni ordering history/context measurements

Frozen final-EMA comparison. Training CSV records full mixed objective, flow clean
prediction MSE, decoded grouped MSE, auxiliary smooth value MSE and directional
response MSE normalized by train teacher response power (floor1e-6). Auxiliary
value weight0.1; derivative additional factor0.25 every fourth update.

Trajectory scores reuse ni_kinetic_latent.assessment.scores without changes:
normalized whole-path energy with unbiased ensemble-pair term; increment energy;
lag1/5 increment-product energy; endpoint mean MSE; temporal, ensemble and actual
MD power; empirical5th/95th-quantile coverage from32draws. Coverage is a finite
ensemble diagnostic, not exact population90% calibration. Four measurement
families have equal weight using the inherited train variation mask.

Each local entry is completion of three consecutive crystalline central-atom
future frames. PTM is truth; generated/oracle phases use the same frozen train
logistic measurement readout. For k entries among P independent neural samples,
raw Brier=(k/P-y)^2. Monte-Carlo-corrected Brier=
k(k-1)/(P(P-1))-2*y*k/P+y, an unbiased estimator of (p-y)^2 conditional on input
and event label; individual estimates can be negative. This corrects sampling
noise, not model bias. The rare panel has256samples at a fixed outcome-independent
two windows/source/regime. Negative-control score=y; actual-future readout Brier
uses actual future measurements, never a predictive input.

High-order liquid is noncrystalline and above original train-liquid80th-percentile
qbar6/coherence. Occupancy is time fraction; occupancy error squared difference;
longest-dwell error is squared duration difference inps². Present-liquid reporting
is a descriptive evaluation stratum, not a birth-input filter. The outcome-enriched
window bank does not estimate natural nucleation rates. Undefined raw metrics
remain undefined and all source identities remain in the exported comparison.

Values average rows within ancestor then ancestors equally;1000whole-source
bootstrap draws. Paired comparisons average fitted seeds then bootstrap sources;
fitted seeds are fixed. All trained seeds are reported, without test selection.

Actual-nucleus readouts preserve the frozen32/64-atom1/3/6ps assay, source weights,
role ancestry and prediction contracts from ni_phase_sim.births. CatBoost uses
600iterations/depth4/logloss and no eval_set. AP,AUROC,NLL,Brier and fixed reliability
bins use source bootstrap. Linear logistic readouts use train-only mean/std,
C=1,max_iter1000 and the same row weights; exported as descriptive point estimates.
The generated-future feature vector concatenates244-dimensional endpoint means
and standard deviations at1/3/6ps. Observed-history and128D learned-history features
are separate probes. The trajectory model never receives birth labels.

Response diagnostics: heldout smooth JVP squared error/train direction-power,
zero-JVP comparator, smooth-value MSE at5/20/50/100fs. These aggregate teacher states
and do not claim independent-source uncertainty or six-ps derivative supervision.
