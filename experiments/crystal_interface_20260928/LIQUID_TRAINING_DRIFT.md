# LCD-MACE128-VC: interim training drift diagnosis

The strict liquid-only run is **overfitting, not showing numerical divergence**.
Its predictive validation objective is best at nominal block 2, then generally
worsens through block 10 while training losses decrease. This diagnosis uses full
selection-population logs through update 5120 and an independently frozen
checkpoint at update 4416. It is not a final test result.

Run: `crystal_liquid_distance/al64-dense-vcreg-20260928/distance_direction_vcreg`.
[W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/32f3b44a24e49a2113b2).
The [analysis bundle](/work/PERSO/vmorozov/analysis/crystal_liquid_distance/al64-dense-vcreg-20260928/distance_direction_vcreg/analyses/training-drift-v1)
contains frozen logs, the interim checkpoint, reproducible diagnostic scripts,
JSON results and implementation/input hashes. No live weights, configuration,
selection rules or training processes were changed by this audit.

![Training and validation drift](/work/PERSO/vmorozov/analysis/crystal_liquid_distance/al64-dense-vcreg-20260928/distance_direction_vcreg/analyses/training-drift-v1/plots/training-drift.png)

## What worsened

All entries below use the exact conditional selection population and its existing
weights. Lower is better. The constant distance is the conditional **training**
mean, 40.1371 Å. The no-input probability model fits one censored lognormal on
training distances using distance NLL plus the same proximity objective, with a
uniform directional density. These are local diagnostic baselines, not new
encoder fits. No selection or test labels fit these constants/distributions.

| Validation metric | Block 2 | Block 10 | No-input reference |
|---|---:|---:|---:|
| Complete predictive objective | 6.64749 | 6.71889 | 6.66372, fitted distribution |
| Distance NLL | 3.69693 | 3.72280 | 3.70356, fitted lognormal |
| Direction NLL, valid directions | 2.52610 | 2.56427 | 2.53102, uniform sphere |
| Capped-distance RMSE, Å | 10.60783 | 10.77064 | 10.69400, training-mean constant |
| Brier, distance ≤20 Å | 0.005005 | 0.005096 | 0.004971, training prevalence |
| Brier, distance ≤32 Å | 0.197058 | 0.201073 | 0.199234, training prevalence |

The predictive objective rises by 0.07140 between blocks 2 and 10. Its exact
decomposition is 52.02% direction, 36.24% distance and 11.74% proximity losses.
The direction contribution is its conditional NLL change multiplied by the
unchanged valid-direction probability 0.972995; proximity has configured weight 2.

Training means over the 32 logged minibatches in each complete block fall from
6.65427 at block 2 to 6.53395 at block 8. Training distance NLL falls from 3.71423
to 3.67699, and direction loss from 2.45223 to 2.39472. Training direction loss
includes zero for invalid directions; validation divides by valid-direction mass.
Their raw levels therefore must not be compared without that normalization.

The early distance improvement is small: about 0.8% RMSE below the constant
baseline. By block 10 that improvement has disappeared in the point estimate.
These are validation point estimates, not source-bootstrap significance claims.
They do not establish that liquid structure has no useful predictive information.

## What the population change exposed

The input restriction is now doing what the liquid-distance question requires:
every fitting query is outside established crystal, every consumed patch is free
of established crystal, and an established crystal exists elsewhere. Its distance
is checked against the inherited nearest-interface distance on this domain.
Partially ordered, subcritical liquid is retained. No phase, visibility, crystal
existence, temperature, age, species or time covariate enters the model.

This is a much harder conditional problem than recognition with crystal already
visible. The expanded fitting population has 183,596 rows, but only 88 source
trajectories and 1,590 source/frame combinations. Many contexts reuse the same
physical configurations. The row-weight concentration `1/sum(w**2)` is 17,810;
this is a sampling-weight diagnostic, **not** an estimate of independent physical
observations. Adding centers did not add independent trajectories.

The inherited proximity thresholds are a poor fit to this new population:

- No eligible training or selection target is within 8 or 12 Å. The minimum
  training distance is 14.03 Å. Those two BCE thresholds only learn negative cases.
- Only 0.4345% of training probability is within 20 Å: 913 rows, 2.225 expected
  examples per global batch of 512, and a 10.76% chance of none. Random batches
  are functioning as specified; a large batch does not create missing examples.
- Distance ≤32 Å has 25.84% training probability and remains informative as a
  proximity target. The inherited loss is not entirely degenerate.

These observations explain why the easy signal disappeared and why row count
overstates the diversity of the fit. They are not a causal proof that class
imbalance alone caused overfitting.

## Numerical and objective audit

All saved model tensors at update 4416 are finite. The reviewed distance and
conditional direction likelihoods have consistent signs and marginalization.
The checkpoint uses the unchanged, hashed frozen implementation. Learning rates
decline on schedule; no NaN or optimizer failure was observed.

The logged gradient norm is **before** clipping at 5. At block 8, 81.25% of the
logged steps exceed that threshold; their 95th-percentile norm is 19.51. Seeing
a value above 5 on that curve is not evidence that clipping failed. Frequent
clipping is an optimization consideration, but its finite noise is distinct from
the worsening generalization curve.

VCReg remains approximately 0.005 after warmup, versus a predictive objective
around 6.5. Its scalar/vector variance and covariance statistics remain bounded.
At the frozen checkpoint, 1,024 replacement draws per role give mean standardized
scalar standard deviation 1.009/1.015 and vector RMS 1.011/1.006 for train/selection.
These are patch statistics, not proof of representation usefulness or full rank.

A separate 64-row training-batch gradient decomposition gave total parameter
gradient norms 21.24 for distance NLL, 9.47 for weighted proximity, 5.59 for
direction and 0.215 for VCReg. VCReg's encoder gradient norm was 0.026, compared
with 14.30 for distance NLL. This single-batch check argues against VCReg causing
the observed instability; it is not a population-wide gradient comparison.

The direction output is normalized almost to unit length before applying a
distance-dependent concentration. On the sampled rows, pre-normalization norms
were at least 0.0053, much larger than the 0.0001 softening scale. Thus the head
cannot readily express uncertainty through that vector's magnitude. Uncertainty
must largely come from combining different directions across mixture components.
The fixed concentration can encourage fitting directional detail unsupported by
held-out liquid structure. This is a plausible design contributor, **not a proven
cause** without a matched ablation.

The 25 distance/direction components did not collapse onto one winning component:
their mean effective mixture count was about 24.3 on both sampled roles. Individual
lognormal widths approached their 0.15 floor (median about 0.158), but a mixture
of narrow components can still be broad. This is not evidence of predictive
variance collapse by itself. The sampled gradient analysis also does not show
direction dominating all parameter updates merely because it dominates validation
deterioration.

## A definite checkpoint-policy mistake

The inherited recipe only saves a best checkpoint among blocks 12–16. Block 2's
better model has already been overwritten by rolling `last.pt` saves. The frozen
audit copy preserves update 4416, **not** the earlier best model.

Requiring at least 12 blocks of training should not prevent preserving earlier
checkpoints. This rule does not cause learning to overfit, but it makes model
selection worse and prevents a clean early-versus-late representation comparison.
Changing the active frozen protocol silently would invalidate provenance, so this
audit records the issue without rewriting the run.

## Recommended next correction

1. Preserve the best validation-likelihood checkpoint from the first evaluation;
   a minimum training budget can remain independent of checkpoint eligibility.
   Keep intermediate checkpoints for comparing what the encoder learns over time.
2. Keep VCReg. Run a matched distance/proximity-only fit first to establish whether
   liquid geometry adds information beyond the fitted no-input distribution.
   Compare the same source split and sampling population; do not select on AP.
3. In a separate distance-plus-direction fit, let directional concentration express
   uncertainty, including a uniform prediction, while retaining the decreasing
   penalty with distance. Do not infer that simply reducing direction weight is
   sufficient without a matched comparison.
4. Predeclare proximity thresholds supported by this conditional population, e.g.
   20/32/48 Å, and report 8/12 Å only as unsupported diagnostics for this population.
   Changing thresholds defines a new objective; raw objective values then cease
   to be directly comparable to this run. Preserve common distance metrics.
5. Assess gains by source and against no-input baselines. More uniformly sampled
   centers alone is unlikely to resolve the observed generalization gap; independent
   liquid configurations with an external crystal are the relevant data expansion.

The current training remains unchanged. Its scheduled final evaluation can still
measure the frozen protocol, but the current trajectory provides no reason to
expect that additional blocks alone will repair this loss of generalization.

## Reproduction and definitions

The analysis bundle's `technical/population_audit.py` loads the sealed per-source
metadata, reproduces the actual population weights, and requires exact agreement
with the training observation-filter audit. The no-input lognormal is fitted in
float64 on all eligible training distances, with distance cap 64 Å and sigma
constrained to the model's [0.15, 2] range. It has no zero-distance point mass,
because all eligible distances are strictly positive. Censoring likelihood and
proximity log scores match the scientific objective.

`technical/checkpoint_audit.py` imports the frozen training implementation, draws
1,024 independent rows per role using their declared weights, and evaluates the
copied checkpoint without an optimizer. One 64-row draw supplies gradient norms.
These sampled diagnostics supplement, rather than replace, the complete validation
metrics in the main table. `technical/render.py` freezes the observed log prefix
and generates the plot, decomposition and SHA256 receipt. Scripts run with conda
`pointnet-torch214`; checkpoint evaluation uses the existing environment setting
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the installed e3nn constants. No diagnostic
creates an online W&B run. Test predictive outcomes were not evaluated here.
