# All available 0.10 ps nucleus-birth evaluation of response encoders

## Pooled test and neighborhood audit

The user requests pooling original validation, calibration and test sources.
The separate `all010-pooled-v3` analysis retains the parent cohort, fixed encoder
checkpoints and original role metadata. Its eligible test contains 11 physical
births from 9 melt ancestors, with 19 training births from 18 ancestors. No
neighborhood eligibility is changed in this revision.

The original readouts used validation labels for selection and calibration
labels for their probability transform. This revision refits all 153 diagnostic
readouts: NLL selection uses four deterministic original-training ancestors,
then final fitting uses all original-training ancestors. Report raw scores;
the combined test supplies no readout fitting, selection or calibration labels.
Frozen encoders retain their original label-free response selection. Earlier
results have been inspected, so this remains exploratory evidence.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.response_dense_birth.pooled launch --config configs/analysis/response_all010_pooled_20261006.json
```

Two GPU lanes refit/read out frozen embeddings; a CPU audit replays the exact
four sampled centers for all 210 candidate event views. It records blocking
component size, periodic distance, onset-atom overlap and prior establishment.
The audit also measures how many excluded events have other eligible onset
centers. The gate is reproduced against every original saved eligibility count.
See [definitions](metrics/response_birth_pooled.md). The detached job has a
two-hour cap bounded by the parent's original deadline. Ownership and progress
are in its `technical/launch.json` and `technical/coordinator.json`.

Output: `${storage:training_storage}/response_velocity/native500-f64-20261006-v1/analyses/all010-pooled-v3`.
`analyses/comparison-v1` holds the new pooled scores;
`analyses/neighborhood-audit-v1` holds event, center and component tables.
Existing readouts, four-event test scores, source labels and dataset releases
are preserved. No new simulation, encoder training or W&B run is created.
The environment setting is the same trusted e3nn constants loader used by the
parent successful response workflow. The first pooled startup omitted it and
failed before any readout fit; its logs and receipts are preserved under
`technical/startup-attempt1`. The corrected restart retains the original
17:26 Paris deadline and reuses the completed source-audit packets.

Completed at **15:32 Paris on October 6**: all **153 refits**, **92 audited
source views / 210 candidate event views**, pooled scoring and updates to the
existing W&B training summaries. Prediction and exported-table checksums were
verified. The pooled cohort has 11 births / 9 ancestors; the detailed audit and
[scientific findings](../experiments/response_velocity_20261004/README.md#october-6-native-velocity-encoders-pooled-nucleus-birth-evaluation)
explain why neighborhood eligibility excludes most candidates.

## Original separate-role evaluation

Compare all twelve frozen October 6 response models on exact observations
0.1, 0.5 and 1 ps before eight-atom emergence of a lineage that later establishes
at 64 atoms for 1.5 ps. Four same-source/time liquid controls accompany each case.
Small precursors are allowed; eight-atom components inside 8 A are excluded from
all inputs. Nonlinear and linear readouts evaluate embeddings and existing
future-prediction heads, with rich structural and constant-prior controls.
[Complete definitions](metrics/response_dense_birth.md).

The current immutable population joins exact 0.10 ps PTM labels from three v3
releases, retaining fixed Al64 ancestry roles. It includes 92 source views with
saved velocities: 88 native 0.10 ps views and four exact 0.10 ps views of native
0.01 ps histories. Seven views are historical partial archives. The view roles
are 60 train, 10 selection, 9 calibration and 13 test; these are not independent
melt counts. Source preparation starts with available complete labels; October 6
sources wait for their already-running lineage jobs.
Fitting begins only when all sources and catalogue releases are complete. These
are read-only dependencies: this workflow does not stop or change their jobs.
No new simulation, encoder training, relaxation or W&B run is launched.

Use conda `pointnet-torch214`:

```bash
python -m src.research.response_dense_birth.workflow launch --config configs/analysis/response_all010_birth_20261006.json
```

The launcher freezes code/config/metric definitions and detaches four CPU source
preparation workers. Two GPU lanes run frozen inference and 153 local diagnostic
fits after sealing. The revised launch retains the original nine-hour deadline,
including waiting for labels; it does not restart that budget. PID/process-group,
allocation, deadline and exact command are recorded in `technical/launch.json`.
Manage only those recorded processes, never the containing allocation or label
search jobs. `technical/coordinator.json` separates preparation, label wait,
fitting and analysis. Each source and fit has a checksum-bound completion receipt.
The combined comparison appears only after both lanes finish.

Results: `${storage:training_storage}/response_velocity/native500-f64-20261006-v1/analyses/all010-birth-v2`.
Prepared arrays: `${storage:cache}/response-birth/all010-v3-20261006-v2`.
`analyses/coverage-v1` records every source/event and eligibility exclusions;
`analyses/comparison-v1` holds metrics, paired uncertainty and plots.
Seven million-atom views and one older 100k-atom view lack saved velocities and
are explicitly excluded from the matched Q/QV population. The user authorized
omitting the million-atom collection. Other cadences, shooting, relaxed views
and collections outside the three pinned lineage releases remain separate.
This is a retrospective matched birth assay, not a population nucleation-rate
estimate or completion of all Al analysis. Inclusion in the 92-view inventory
does not guarantee eligible prediction rows: the recorded root-birth, input
structure and follow-up criteria still apply.

Related views retain their native cadence and origin. Equal full-cell onset
coordinates, velocities, box, first members, time and ancestry define an event
alias. A repeated positive atom with identical Q/V inputs at all three leads
contributes one matched case/control pair. Distinct histories remain explicit;
physical events receive equal total weight, and uncertainty resamples whole
melt ancestors. `duplicate-pairs.csv` records dropped pairs.

## October 6 expanded launch

Started at **14:23 Paris**, with the unchanged **22:48 Paris** deadline, in
allocation **1023686** on **node59**. Detached coordinator PID/process group
**43292** owns this revised evaluation. Fifty-four checksum-bound source
preparations from the earlier launch are reused. Manual preparation checks
covered a reused native source, a native 0.01 ps source mapped to exact 0.10 ps
observations, and an older partial archive.

Only the earlier evaluation coordinator PID 37428 was stopped after verifying
its current command, working directory, process group and this chat's launch
receipt. Its `technical/superseded.json` records the transition; its frozen code,
plan, cached observations and signal termination receipt are preserved. The
separate nucleus-search jobs and simulations were not changed.

## Superseded October 6 native-production launch

Historical config: `configs/analysis/response_dense_birth_20261006.json`.
Results: `${storage:training_storage}/response_velocity/native500-f64-20261006-v1/analyses/dense010-birth-v1`.
Prepared arrays: `${storage:cache}/response-birth/native010-v3-20261006`.

Started at **13:48 Paris**, deadline **22:48 Paris**, in allocation1023686 on
node59. Detached coordinator PID/process group **37428** is owned by this chat;
its launch receipt records the command and frozen bundle. Four CPU workers
prepare sources; GPUs0/1 are used only after the complete cohort seals.

The plan contains **83 trajectories:54 train,9 selection,9 calibration,11 test**.
54 trajectories already have completed v3 labels; the newest29 await the
independent October6 search. Trajectory counts are not independent melt counts.
The actual-source preparation check recovered exact coordinates/velocities and
the frozen QV model replay agreed within **8.93e-6** absolute prediction error
(required<=2e-4). No new training or W&B run was created for these startup checks.
`technical/preparation-attempt1` preserves the initial manual preparation before
adding execution-deadline and early-coverage reporting. Scientific rows and
labels were unchanged; the final launch has a separately bound implementation.
