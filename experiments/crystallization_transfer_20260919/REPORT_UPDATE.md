# Pending completion update for the consolidated report

Status: **open**, recorded 19 September 2026. This is a durable reporting task,
not a request to launch training or an automatic notification.

Report: [Recent local encoders and crystallization](../../output/crystallization_transfer/recent-report-20260919/README.md).
Its current evidence snapshot is 2026-09-19T18:22:46 UTC.

The initial/scaling studies (102 fits plus baseline), the first path study
(10 fits), and path refinements (35 fits) are complete. The corrected encoder
study has completed all 52 screens; its ten promoted 12/24-epoch fits remain.

Update trigger: all 62 statuses in
`output/crystallization_transfer/mace-adaptive-20260919/technical/runs/`
are `complete`, with predictions, metrics and frozen metric contracts present.
A failure or interrupted fit is not completion; retain and explain it instead
of dropping it from the report.

## Required report update

- Preserve the current dated snapshot before recapturing evidence. The disposable
  capture/render helpers under the report's `technical/` directory contain
  snapshot-specific narrative: edit the narrative and pending labels as well as
  rerunning the tables. Do not blindly regenerate and call the result final.
- Add all ten promoted encoder outcomes, actual/selected updates, exact parent
  and selected settings. Compare budgets and modes using selection-chosen models;
  do not select the best test result across the sweep.
- Refresh frozen/fine-tuned/scratch and scalar/tensor comparisons. Include actual
  context radius/history/attention settings so changes are not attributed solely
  to pretraining or fine-tuning.
- Report AP/AUROC, calibrated FPR/recall/precision, event NLL and timing with misses
  at 9/24/96 ps. Keep six-bin hazard timing separate from dense path timing.
  Include repeated-alarm and sparse spatial outcomes where available.
- Add paired temperature-stratified whole-source intervals for predeclared
  contrasts. Reuse saved per-window predictions and preserve source identities.
  State that intervals condition on one training seed and that test sources
  were previously examined; do not imply confirmatory, multiplicity-corrected
  model discovery.
- Reconcile the trajectory physical-quality gate with event selection. A lower
  test NLL alone must not override selection or the physical retention criterion.
- Retain failed old diffusion/normalization results with their diagnoses. Preserve
  historical metric definitions and hashes; export any new CSVs using the metric
  documentation mechanism.
- Update the consolidated conclusion, appendices, status banner and this note to
  `complete`, recording capture time and remaining limitations.

No new simulation, model fit, or test-driven hyperparameter choice is authorized
by this reporting note itself.

## Trajectory figure supplement, September 21

The later completed context-night study has a separate paper-style PNG gallery:
`output/crystallization_transfer/trajectory-figures-20260921/README.md`.
Reproduce with `python -m src.research.crystallization_paths.figures --config
configs/analysis/crystallization_figures.json`. It uses the four development-chosen
family promotions, original held-out predictions, fixed-seed CPU path illustrations,
training-only UMAP fitting and actual cached spatial neighborhoods. This figure
supplement does not itself close the earlier report-update task above.
