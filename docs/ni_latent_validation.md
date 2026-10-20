# Running the Ni validation follow-up

## October 18 evaluation recovery

All 42 fits, 30 convergence/conditioning diagnostics and 33 native-method
evaluations completed. The original full-forecast stage completed for two arms
and exhausted its attempts for 28: unchunked batched 4x4 eigendecomposition in
the validity audit requested 220 GiB of cuSOLVER workspace on A100 GPUs.
The nine-window integration panel had not exposed the full-batch problem.

The maintained audit now processes 1024 independent matrices per call. A
real-data check at the full `[32, 32, 60, 244]` measurement shape used 1.04 GiB,
retained all observations, matched CPU values on the comparison slice and found
no scalar or PSD violations in saved MD. Mathematical criteria are unchanged.

The separate execution revision
`output/ni_latent_flow/validation-20261018-v2-eval-r2` reuses all 30 completed
predictors and their frozen training identities. It re-exports full forecasts
under the revised audit contract; no training is repeated or selected. Original
failed receipts and exports remain untouched. Jobs **1046456–1046461** run six
evaluation workers from a recorded source copy (five A100, one A40), retaining
the original 23:35 deadline. All six passed the formerly failing first batch.
See its `technical/evaluation-revision.json` and `technical/submission.json`.

The [scientific protocol](../experiments/ni_latent_validation_20261018/README.md)
declares the comparisons and limits. The recipe is
[`validation_20261018.json`](../configs/ni_latent_sim/validation_20261018.json).
Its output is `output/ni_latent_flow/validation-20261018-v2`, linked to STORE.
Read actual execution receipts for current progress; this document is not a
completion record.

## Verified launch on October 18

The queue started at **11:35:19 Europe/Paris**, with a fixed deadline of
**23:35:19**. Both RTX PRO 6000 workers in allocation **1045373** passed update
500, wrote resumable checkpoints and connected to online W&B. The first two
fits are the raw and scalar-support direct block predictors. No test result
is implied by these startup checks.

Replacement workers **1045969** and **1045970** were submitted in advance,
eligible after the current allocation ends at approximately **16:30**. Their
partition list is `RTX6000PRO,A100,A40,L40S`; actual starts depend on Slurm
availability. They preserve the original global deadline, source snapshot and
scientific budgets. Receipts are `technical/continuation-0.json` and
`technical/continuation-1.json`.

The real-data preflight passed four debug training/reload paths and the full
assessment, including eight table exports, three plots, measurement-space phase
classification, scalar/joint constraints and resume. Debug runs used training
observations only and created no online scientific runs.

At the user's request, four additional workers were submitted at 11:58, bringing
training concurrency to six GPUs: the two existing RTX PRO 6000 GPUs plus four
A100 40GB GPUs. Jobs **1045996–1045999** use lanes 2–5 and the same frozen source,
data, budgets and global deadline. They have four-hour allocations and retain
the existing checkpointed continuation mechanism if work remains. Actual startup
and online training were verified; see `technical/additional-workers-20261018.json`
and `technical/additional-workers-verified.json`. These extra workers claim
independent arms under the existing locks; batch size remains 256.

A separate short-lived A100 job, **1046001**, prepares the four shared kinetic
control caches serially with the unchanged frozen evaluator. This avoids
concurrent evaluators treating a cache-lock collision as a deadline pause.
It creates no training/W&B run and changes no scientific definitions; its
ownership, command and progress are in `technical/controls-precompute-*.json`.

## Prepare and launch

Use conda `pointnet-torch214` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the
installed trusted e3nn constants. The saved-history expansion is already complete:
`technical/expansion-complete.json` binds its plan and payload hashes. Collection
uses `src.research.ni_latent_flow.expansion.collect(config, workers=12,
device='cuda')`; preparation and subsequent loads reuse the sealed payload. It
does not launch MD. The registered collection is `ni-native-expansion-20261018`.

```bash
python -m src.research.ni_latent_flow.queue prepare \
  --config configs/ni_latent_sim/validation_20261018.json
python -m src.research.ni_latent_flow.queue launch \
  --config configs/ni_latent_sim/validation_20261018.json
```

Launch once with two allocated GPUs visible. It verifies metric contracts and
checkpoint/data identities, freezes source/configuration, and starts two detached
workers. `technical/submission.json` records PIDs, allocation, devices and absolute
12-hour deadline. All scientific fits use online W&B with stable resumable IDs;
debug runs and diagnostic probes remain local. There is no dependency on a chat
review or automation. There is no duplicate birth-probe worker for this recipe.

The queue preserves bounded retries, independent stages, per-arm locks and
checkpoint recovery. Allocation-boundary continuations request one GPU per
required worker from `RTX6000PRO,A100,A40,L40S` as alternative partitions. Only
stage processes started by this queue may be terminated at their deadline.

## Inspect and reproduce

`technical/data-audit.json` records source counts, exact cadence and local-entry
coverage. `technical/preflight/` contains local real-data integration receipts.
Each arm's `technical/` contains `training-rows.json`, `progress.json`, resumable
`last.pt`, fixed `update-3200.pt`/`update-12800.pt`/`update-32000.pt`, `final.pt`,
`complete.json` and `wandb.json`. Encoder-only arms have no dynamics snapshots;
their completion must name only the representation stage.

Separate completion markers are `diagnostics-complete.json`,
`kinetics-complete.json` and `flow-evaluation-complete.json`. The last includes
the all-path validity audit; the shared assessor's `evaluation-complete.json`
alone does not mean that audit finished. Stage failure receipts remain visible.

The new metric family is [ni_latent_validation](metrics/ni_latent_validation.md).
Each CSV exports frozen definitions and implementation hashes. Main tables are
`trajectory`, `retention`, `validity`, `validity_channels`, `diagnostics` and the
`kinetics_*` families. The final comparison also reports old/additional source
collections separately. Plots include actual and generated order trajectories.

Individual stages accept `--name ARM --deadline UNIX_TIME`. For reproduction or
resumption, invoke the frozen source in `technical/source` with its `config.json`,
not the mutable repository recipe. `compare` aggregates already produced tables;
it does not select a checkpoint. See the shared
[queue mechanics](ni_latent_flow.md#detached-launch-and-allocation-boundaries).
