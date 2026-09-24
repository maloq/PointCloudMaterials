# Does preserving physical neighborhoods improve predictive representations?

This two-seed factorial trains eight new spatial MACE encoders end to end.
The question is whether explicit physical-distance and future-information losses
preserve useful liquid distinctions and improve crystallization prediction.
A visually separated UMAP or a larger embedding rank is not a success criterion.
The [preceding repaired screen](../structural_state_20260923/README.md) found little
benefit from teacher/distance additions and almost unchanged ordinary future-order
prediction. This follow-up adds future supervision and repeats initialization.

## Matched interventions

| Arm | Geometry + current order | Physical distance | Future residual |
| --- | --- | --- | --- |
| B-relaxed | 1 + 0.25 | 0 | 0 |
| D-physical-distance | 1 + 0.25 | 0.1 | 0 |
| E-future-residual | 1 + 0.25 | 0 | 0.25 |
| F-distance-future | 1 + 0.25 | 0.1 | 0.25 |

All arms observe the same complete relaxed 8 Å patch, using the repaired native
MACE encoder with cuEquivariance, pooled64 plus learned64 export, fitting-only
normalization and bounded linear heads. No frozen encoder bypass is present.
All arms have identical heads, initial encoder weights and sample order within
seed. Seeds are 20260923 and 20260924; the downstream probe seed is fixed at
20260922. Each fit receives 4096 updates, batch256/micro64, encoder learning rate
1e-5 and head rate3e-4. The distance weight ramps over512 updates. Physical units,
head initialization, optimizer and duration match across the four arms.

The common geometry target is relaxed radial17 and l2/l4 Gram36, averaged equally
across three blocks. Every arm additionally predicts current original-MD order8.
This differs from the preceding screen: current order is now an encoder target
in every arm, so future arms do not gain exclusive access to structural labels.

The future target is original-MD order8 at the same atom after9ps, minus a fixed
ridge prediction from current original-MD order8, relaxed geometry89, temperature
indicators, elapsed time and squared elapsed time. Ridge alpha1, input scalers,
coefficients and residual scalers are fit only on the25 fitting roots. This tests
information beyond this declared linear present-state baseline; it does not
remove every nonlinear function of the present. These baseline inputs construct
labels and are never additional encoder inputs. No crystallization-onset label
is used to train or select an encoder. Future/current targets are standardized
separately using fitting rows. The calibrated initial future head is identical
across all arms; inactive heads receive no gradient.

The primary checkpoint is the fixed final update4096. Initial,1024,2048 and4096
weights are retained. Best geometry-tuning checkpoints are secondary only.
No outcome-dependent shortening, arm promotion or final-checkpoint selection.
The RMS-spread optimization guard stops below10% of initial fitting spread.

## Population and withheld measurements

Reuse structural-state-screen-20260922:45 independent Al2NN-MEAM roots,
25 fitting/5 tuning/15 reused development,2880 patches,1600 fitting observations.
All descendants remain within their assigned root split. No new simulations.
Relaxed patches come from full-cell minimization; the original-MD current/future
order targets have their original producer's spatial support. Thus supervision
uses privileged physical labels rather than claiming every label is a deterministic
function of the relaxed local patch. Archived coordinates are float16.

Angular distributions and l6 moment targets are withheld from encoder losses.
Original-MD future-order measurements at3 and12ps are also withheld;9ps is a
training target for E/F. Neighbor retrieval uses raw exported Euclidean distance,
k5 fitting neighbors matched on current temperature and coarse crystalline flag.
Report noncrystalline query results separately, so crystal/liquid separation alone
cannot improve the main liquid-neighborhood comparison. The legacy metric label
PTM_other means the complement of PTM types1/2/3 here, not strictly PTM code0.

Frozen regression probes, initial-encoder controls, geometry89/PCA64 controls and
temperature-only controls share capacities, source splits and probe seeds.
Crystallization uses the existing five-bin sustained-onset assay:643 development
at-risk windows,18 positive12ps windows from10 independent roots. Report AP,
Brier, binary log loss, event-time likelihood, false alarms and missed events.
Paired2000 source bootstraps preserve temperature strata. These reused development
sources are not a final test; the experiment is a mechanism screen.

## Predeclared interpretation

Compare D−B, E−B, F−B, E's increment F−D, and distance's increment F−E. Do not
choose a winner from a UMAP, rank or the largest observed AP.

For each contrast and seed, require at most2% worse noncrystalline present-state
MSE in **every** declared retention measure: final native radial/l2/l4 heads and
frozen ridge probes of relaxed radial/angular/l6 and current order. A future
mechanism screen additionally requires at least1% lower12ps noncrystalline ridge
MSE; a neighborhood screen requires at least1% lower mean relative discrepancy
across withheld relaxed angular/l6 neighbor targets. Require the respective rule
in both seeds. These practical thresholds are not statistical significance tests.
Source intervals and individual target changes accompany the rule results.

Compare each matched physical effect with its AP and Brier change. Opposite signs
would show that better physical neighborhoods and better event prediction are
separate properties; aligned signs are suggestive, not a causal mediation result.
Report both linear and nonlinear onset heads, including selected constant priors.
The mean-seed interval uses shared paired source draws, conditional on the two
fitted seeds; it does not estimate training-seed population uncertainty.

This experiment does not isolate a VICReg term or establish temporal continuity
of every embedding channel. It tests concrete alternative objectives on one
fixed architecture. A promising result requires replication on fresh sources and
then an otherwise matched VICReg/JEPA architecture comparison.

[Campaign recipe](../../configs/structural_state/future_metric_campaign_20260923.json) ·
[Execution](../../docs/structural_state.md#distancefuture-factorial-23-september) ·
[Metrics](../../docs/metrics/structural_state_future.md) ·
[Results](../../output/structural_state/future-metric-20260923/README.md)

[Completed findings](../../output/structural_state/future-metric-20260923/RESULTS.md): all eight fits finished; small distance/Brier gains, no predeclared mechanism success, and no replicated future-residual benefit. Initial encoder weights match bitwise; fitting-only head calibration has the independently audited CUDA reduction/solve noise described in the result metric contract.
