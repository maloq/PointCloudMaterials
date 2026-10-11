# Can derivatives stabilize the autonomous Ni trajectory?

The original pilot produced unstable6ps free rollouts and did not outperform a
zero-sensitivity predictor on held-out physical derivatives. This factorial
separates three possible effects instead of attributing any longer-rollout gain
to simulator derivatives alone:

| Factor | Control | Treatment |
| --- | --- | --- |
| Physical response | Future values | Values plus existing100fs simulator responses |
| Learned sensitivity | No constraint | Penalize excessive1ps Jacobian gain |
| Training horizon | 1ps | 6ps |

Each cell has two seeds,3200 updates, batch256, identical d16/history4 networks
and frozen unrelaxed eSEN inputs. This dimension/history was chosen because the
original pilot exposed a severe instability there, not because of birth AP.
Initial observation anchors, source roles and data are matched. Original runs
remain immutable; these are new fits. Four observations span0.3ps; autonomous
prediction is evaluated through6ps with no later trajectory feedback.

The new penalty differentiates the learned stochastic transition, holding noise
fixed. It limits excessive perturbation amplification over a full1ps, at both
observed states and states reached after1/3/5ps of free evolution. It does not
add physical long-horizon derivative labels. The existing analytic GPU-EAM
teachers remain a separate100fs supervision factor. Derivative arithmetic in
the learned model is float32; cached physical teachers retain float64 MD/tangents
and float32 forces.

Predeclared comparisons pair Jacobian/no-Jacobian models within the same seed,
physical-teacher treatment and training horizon. Report structural, motion and
descriptor proper scores, MSE against persistence and training-mean controls,
coverage, predictive spread, severe-error/nonfinite fractions, and held-out
physical-response errors including the zero-response reference. Birth32/64 AP,
NLL, Brier and AUROC remain diagnostic probes. Stability without useful variation
and accurate futures is not success. This first pass has two seeds and a fixed
regularizer; it cannot establish an optimal gain cap or a global guarantee.

Recipe: `configs/ni_latent_sim/derivative_stability_20261010.json`.
Run via the existing `python -m src.research.ni_latent_sim.workflow submit --config`
command with that recipe, in pointnet-torch214. Online training uses the existing
W&B project; numerical probes and evaluations remain local. New full-cell
physical lifting is disabled: this experiment tests trajectory stability.

[Exact penalty and metrics](../../docs/metrics/ni_latent_sim.md),
[execution record](../../docs/ni_latent_sim.md).

## Completed results, October 10

All16 fits reached3200 updates, and all free-rollout, physical-response and
birth evaluations completed. Final worker completion was16:22:12 CEST,
22.68minutes after launch. The analysis bundle is
`output/ni_latent_sim/derivative-stability-20261010-v1/analyses/stability-comparison-v1/`.

The table gives standardized structural-feature MSE at6ps, averaged over the two
seeds and with equal held-out ancestor weighting within each seed. Lower is better.

| Treatment | Trained to1ps | Trained to6ps |
| --- | ---: | ---: |
| Values | 1.01067 | 0.95384 |
| Values + physical response | 0.98354 | 0.95543 |
| Values + Jacobian penalty | 0.96465 | 0.95480 |
| Values + physical response + Jacobian penalty | 0.96589 | 0.95526 |

The constant training-mean control scores1.00168; persistence scores1.79549.
For1ps training, adding the Jacobian penalty reduces6ps MSE by4.55% without
physical-response matching, or1.79% with it. Without the physical response,
the per-seed absolute improvements are0.08541 and0.00663; with it they are
0.02435 and0.01096. Thus both seeds improve, but effect size varies substantially.
For6ps training, the corresponding effects are a0.10% degradation and a0.017%
improvement: practically little added gain. Structural energy score and NLL
also improve for1ps training with the penalty;6ps differences are small/mixed.

The large original structural divergences did not recur in ANY new treatment:
at6ps, all16 models have zero structural component-mean and ensemble-mean
failures under the predeclared standardized MSE>100 criterion, and no nonfinite
structural component means. This is not evidence that the derivative penalty
eliminated the original failure. The new controls also changed from random
within-window starting times to fixed starts, to match available initial states
across1ps and6ps training. This factorial isolates penalty effects within the
new protocol; it does not isolate the cause of the difference from the pilot.
Tiny descriptor component outlier rates remain in the1ps controls (roughly
0.00002 of particles); no descriptor ensemble-mean failures occur at6ps.

Predicted structural variance remains0.923–1.017 across individual models at6ps,
so ensemble spread has not collapsed to zero. Coverage of nominal90% intervals
is approximately84–85%. These facts do not establish correct conditional future
distributions. Even the stronger6ps-trained value control improves structural
MSE by only4.78% over the constant training mean; descriptor MSE improves by9.58%
(0.73704 versus0.81514). Local future information remains modest.

Physical sensitivity is still not learned successfully: the group-weighted
held-out relative-response errors across the eight treatment averages span
0.52493–0.55041, all worse than the0.500 zero-response reference. The learned-map
regularizer and learning the simulator's actual sensitivity are distinct claims.
The Jacobian hinge was active in5.4–25.7% of1ps-training updates, compared with
2.1–3.6% of6ps-training updates; this is consistent with the weaker added effect
when training already includes long rollouts, without proving a mechanism.

For6ps-trained embeddings, nucleus64 AP at a6ps lead is0.71268 (values),0.69269
(physical response),0.70396 (Jacobian), and0.72078 (both). The corresponding
predictive NLLs are0.28437,0.28241,0.28310,0.28125. These are diagnostic readouts
on49 events from22 held-out ancestors,980 rows with0.20 prevalence; neither the
small AP difference nor two seeds establishes a robust new nucleation benefit.

Conclusion: Jacobian control modestly helps extrapolation from1ps training,
while training through6ps provides the more consistent future-prediction gain.
We have not shown that physical simulator derivatives solve stability or improve
learned motion. A matched reproduction of the original random-start instability
with/without the penalty would be needed to test that specific failure directly.
