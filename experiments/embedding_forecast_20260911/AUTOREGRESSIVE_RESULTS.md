# Autoregressive embedding prediction — September 11, 2026

**Autoregressive training on predicted rollouts is the strongest full-path
method in this pilot. Teacher forcing performs worse when evaluated on its own
predictions.** Five new fits completed, bringing the common pilot to fourteen
fits. The larger multi-seed campaign remains configured but unlaunched.

The encoder, cache, normalization, atom identities, source splits, 6 ps history,
12 future frames through 9 ps, optimizer, maximum 32 epochs and seed 20260911 are
the same as the [initial pilot](RESULTS.md). There are 432/144/144 training,
validation/test windows from 18/6/6 independent MEAM sources. All validation and
test scores below use full predicted rollouts; no observed future embedding is
supplied during evaluation or checkpoint selection.

## Method

The history GRU initializes a recurrent decoder. At every 0.75 ps step, a GRUCell
receives the previous predicted embedding, the fixed encoded history and the next
time. A residual head predicts the next increment. The first previous embedding
is the observed history mean, making the strongest untrained baseline available
without learning it through a bottleneck. Future predictions feed back without
detachment; late losses differentiate through earlier predictions.

Compare two explicitly different training protocols:

- **Rollout training:** predict the complete future from history alone. The
  main loss uses frame MSE + 0.25 bin-mean MSE + 0.1 increment MSE. A second fit
  uses frame MSE only to isolate the auxiliary-loss contribution.
- **Teacher forcing:** during fitting only, feed the true previous future
  embedding into each step, using one-step MSE. The current target is never its
  own input. The matched frame-MSE-only rollout fit isolates the conditioning
  strategy. Checkpoints are still selected by validation rollout MSE.

Mean-only and repeated-anchor AR controls use the same network capacity. The
autoregressive models have 579,072 trainable parameters, compared with 380,160
for the direct mean-residual reference. The AR-versus-direct result therefore
compares these implemented methods; it does not isolate autoregression at equal
capacity. Inputs/targets and optimization settings are matched.

## Held-out results

MSE uses the same training-standardized embedding coordinates. Bin MSE averages
the predicted trajectory into the requested separate (0,3], (3,6], (6,9] ps means.
Every test source has the same number of windows, so source- and sample-weighted
MSE coincide. Lower is better.

| Model | Full-path MSE | Three-bin MSE | +0.75 ps MSE | +9 ps MSE |
| --- | ---: | ---: | ---: | ---: |
| Observed history mean | 0.263882 | 0.129530 | — | — |
| Direct mean-residual reference | 0.251330 | 0.117005 | — | — |
| **AR rollout, frame/bin/increment loss** | **0.242799** | **0.108221** | 0.209272 | 0.248736 |
| AR rollout, frame MSE only | 0.242804 | 0.108229 | 0.209287 | 0.248695 |
| AR trained mean-only control | 0.248593 | 0.114099 | 0.210536 | 0.258919 |
| AR trained anchor control | 0.288592 | 0.152194 | 0.275033 | 0.303634 |
| AR teacher-forced training, evaluated as rollout | 0.272740 | 0.137014 | 0.219831 | 0.309065 |

| Paired full-path error reduction | Estimate | 95% source-bootstrap interval |
| --- | ---: | ---: |
| AR rollout versus direct reference | 3.39% | [1.65%, 5.09%] |
| AR rollout versus observed history mean | 7.99% | [1.28%, 16.97%] |
| AR rollout versus trained AR mean-only control | 2.33% | [−0.47%, 6.63%] |
| AR rollout versus trained AR anchor control | 15.87% | [10.53%, 20.90%] |
| MSE-only AR rollout versus teacher-forced training | 10.98% | [4.65%, 19.75%] |

AR rollout improves over the observed history mean on five of six sources. It
also improves by 39.07% over persistence. Its three-bin error is 2.64% lower than
the direct-bin mean-residual model's 0.111155, but the interval [−0.36%,5.92%]
includes zero. The original direct path lost some bin accuracy relative to the
direct-bin model; this AR pilot largely closes that gap while retaining all
twelve future frames.

The extra bin/increment terms change AR path error by only about 0.002% relative
to frame MSE alone, with interval [−0.047%,0.055%]. They add no measurable benefit
in this pilot. Frame-MSE-only rollout training is therefore a useful simpler
candidate for the next multi-seed comparison.

Teacher forcing has noticeably worse late-horizon performance: its +9 ps MSE is
0.309065, compared with 0.248695 for matched MSE-only rollout training. This is
consistent with the mismatch between conditioning on observed prefixes during
training and predicted prefixes during deployment. The comparison does not imply
that teacher forcing always fails, or that a different curriculum would not help.

## Interpretation and next experiments

Reverse-past intervention increases AR rollout MSE from 0.242799 to 0.253284;
repeating the anchor increases it to 0.333733. These are distribution-shift tests.
The paired interval against the trained mean-only control still includes zero,
so the additional value of ordered history remains unresolved. That is distinct
from the AR model's improvement over the particular direct decoder tested here.

This is one optimization seed and six previously examined test sources. The
source-bootstrap intervals are exploratory and conditional on the fitted models;
they do not measure seed uncertainty or confirm transfer to new trajectories.
Prioritize three-seed comparisons of AR rollout (both losses), the direct model
and trained mean/anchor controls. A direct decoder with matched parameter count
would help separate decoder structure from extra capacity. The shared larger
cache and 3 ps history ablation are already configured; no new simulations or
embedding preparation were needed for these five fits.

## Reproduction, evidence and file roles

The maintained command now selects `architecture: autoregressive_gru` and an
explicit `autoregressive.training` protocol. Reuse the existing pilot cache:

```bash
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot_autoregressive.json --stage train
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot_comparison.json --stage collect
```

The first command starts fresh and refuses existing fit directories. On this
completed pilot, use the second command to recollect results, or `--stage evaluate`
to reload retained checkpoints. To refit, choose a new output path in configuration.
The [main matrix](main.json) contains the five AR variants, and
[history3.json](history3.json) includes AR rollout on identical forecast anchors.

Evidence: [combined paired comparisons](../../output/embedding_forecast_20260911/pilot/runs/comparison.json),
[AR rollout metrics](../../output/embedding_forecast_20260911/pilot/runs/path_ar_rollout-seed20260911/test_metrics.json),
[AR horizon-error plot](../../output/embedding_forecast_20260911/pilot/runs/path_ar_rollout-seed20260911/forecast_scores.png),
[teacher-forced model rollout metrics](../../output/embedding_forecast_20260911/pilot/runs/path_ar_teacher_forced-seed20260911/test_metrics.json).

**23 forecast tests pass** in `pointnet`. New coverage verifies real prediction
feedback, gradients through earlier rollout steps, the initial observed baseline,
correctly shifted teacher-forcing targets, rejection of teacher forcing during
evaluation, and training/checkpoint/evaluation round trips for both protocols.
Existing direct and Gaussian tests also pass. All five real GPU fits completed
training, restored-checkpoint evaluation, history interventions and collection.

The model/trainer changes and tests are maintained code. This report and the new
pilot configuration are experiment records. Logs, checkpoints, scores and plots
are generated artifacts under `output/embedding_forecast_20260911/pilot/`.
