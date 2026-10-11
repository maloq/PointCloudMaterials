# Failed-embryo appearance comparison

This is one cohort in the [shared Al pool](al_research_pool.md). Its completion
does not cover the entire Al workload; use the pool overview for all-Al status.

The [native 0.1 ps follow-up](birth_dense010.md) searches the completed dense
descendants separately, then compares six-frame original-MD histories with
snapshot and repeated-frame controls. It does not reuse this sparse-grid cohort.

The [Al shooting extension and fate experiment](al_shooting_prediction.md) adds
eligible new shooting cases on matched physical input offsets, and separately
predicts verified dissolution versus documented continued growth. Historical
results and this protocol's relaxed-input runs remain unchanged.

Use `pointnet-torch214` and the maintained module command:

```bash
python -m src.research.birth_prediction.appearance submit \
  --config configs/birth_prediction/appearance_transfer_20261002.json
```

This freezes the scientific code, binds the completed 184/28 fate catalogue and
eight original-pool readouts, and submits detached dependencies:

1. Eight CPU lanes extract strict crystal-free histories and liquid controls.
2. Seal coverage and physical patch identities. No model-specific exclusions.
3. Original-coordinate descriptor extraction and readout fitting proceed while
   sixteen independent 16-rank CPU lanes minimize new observed full cells.
4. Seal relaxed clouds, compute descriptors and fit paired relaxed readouts.
5. Collect all train/held-out comparisons, source intervals, probabilities and plots.

Two GPU fitting jobs (one per coordinate domain) use one GPU each; feature
extraction, linear fits and minimization use CPUs. CPU/GPU jobs have separate
receipts and resume completed sources/cells/fits. Full-cell convergence failures
block descendants and preserve failure artifacts. No new simulations or new
W&B runs are created: these are fixed-descriptor diagnostic readouts.

All source/cell caches are outside the repository on IDS; minimization work uses
SCRATCH and verified completed cells/failures are archived to STORE. Scientific
outputs live at `${storage:training_storage}/birth_prediction/appearance-transfer-20261002`.
Queue IDs and frozen code are in `technical/launch.json`; stage receipts record
live status. Four/six/sixteen-hour job limits are limits, not finish estimates.

Submitted 2 October: preparation `1018542`, original descriptors `1018544`,
relaxation `1018546`, relaxed descriptors `1018548`; original fits
`1018550`/`1018551`, relaxed fits `1018552`/`1018553`, collector `1018554`.
Preparation/descriptors group two one-CPU workers per job; relaxation groups
four independent 16-rank workers per job, preserving the configured parallelism
within the per-user submission limit. The rejected initial submission and its
frozen code remain in `technical/submission-attempt1`; its four accepted jobs
were canceled and completed source receipts are reused by this queue.

SCRATCH rejected directory creation before any minimization. The relaxed branch
was resubmitted as `1018565` (relaxation), `1018566` (seal), `1018567` (descriptors),
`1018568` (descriptor seal), `1018569`/`1018570` (fits), and `1018571` (collector).
Only temporary minimization files use IDS via the recorded
`technical/machine-relaxation.yaml` execution override; archival output still uses
STORE. The scientific configuration and cell identities are unchanged. Original
jobs `1018550`/`1018551` continued and completed all four original-coordinate fits.
The earlier execution receipt and failure records are preserved under `technical/`.

Completed eligibility screening retained 133 of 184 candidate events, including
18 of the 28 strong candidates. New training support is 76 events (12 strong),
and merged-test support is 57 events (six strong). Combined with the unchanged
original rows this gives 2,245 training rows and 1,425 test rows. The 51 exclusions
are recorded per candidate; insufficient crystal-free history and unrelated
earlier local crystal appearance are not waived to force all candidates into fitting.

Histories retain original atom identities and exact 0.75-ps cadence. Original and
relaxed variants use identical rows, and the old binary label records are not
rewritten. The new combined target marks both failed and established appearances
positive. The 28 strong episodes occur once within the 184 candidate union.

[Scientific protocol](../experiments/birth_prediction_20260930/APPEARANCE.md) ·
[Metric definitions](metrics/birth_appearance.md).

## Nested cross-validation

October3 recovery: the original-coordinate nested CV finished all40 outer fits
and its analysis. Relaxation completed1,096/1,208 cells; two grouped jobs hit
the16-hour allocation limit, and cell972-392 exceeded its two-hour subprocess
timeout. Job1019811 now distributes the112 unfinished cells across nine16-CPU
lanes, isolating972-392. The per-cell allowance is six hours; the generating
potential, precision,16 ranks, FIRE settings, force tolerance and iteration
limits are unchanged. No completed cell is recomputed.

The extension command is `python -m src.research.birth_prediction.appearance_resume
submit --config configs/birth_prediction/appearance_resume_20261003.json`.
It snapshots the original code dependencies, records the cell timeout extension,
freezes path resolution and reconnects seal job1018566 to the recovery array.
The existing descriptor, fit, comparison and relaxed-CV chain follows normally.
Recovery provenance is `technical/resume-20261003/`; interrupted attempts and
their failure archives are preserved. Incomplete minimizations with no final dump
restart from their original observed frame, with a separate attempt directory.

Requested October 2 after the fixed held-out stronger-candidate subset contained
only six events. Run with `pointnet-torch214`:

```bash
python -m src.research.birth_prediction.appearance_cv prepare \
  --config configs/birth_prediction/appearance_cv_20261002.json
python -m src.research.birth_prediction.appearance_cv submit \
  --config configs/birth_prediction/appearance_cv_20261002.json
```

This separate protocol evaluates all 3,670 eligible rows from119 recorded
ancestors using five outer folds and five inner selection folds, one seed.
All18 eligible stronger failed events receive an outer-held-out prediction:
the folds contain4,4,3,4,3 such events. Models use exactly the same observations,
event weights and controls as the previous comparison. The original train/test
results and fixed-Al64 release remain intact; CV uses all historical roles by
the user's explicit request and is labeled exploratory.

Two detached one-GPU jobs run the original and relaxed domains, respectively.
Each serially runs both readouts, both observations, both training pools and five
outer folds (40 final fits, with inner selection). The relaxed job depends on
the existing descriptor-ready job1018568 and creates no new relaxation work.
Each job exports its results after fitting, without waiting for the other domain.
Path resolution, catalogue and code are frozen with the run.

Results live at `${storage:training_storage}/birth_prediction/appearance-cv-20261002`.
Use `technical/launch.json`, `technical/worker-{domain}.json` and per-fit receipts
for progress. `analyses/{domain}-cv-v1/` contains exactly-once outer predictions,
train/test scores, calibration and paired2,000-ancestor-bootstrap intervals.
Resume the recorded frozen `worker --domain original|relaxed`; existing fits are
identity-checked and reused. Collectors can be replayed with `collect --domain`.

[CV metric definitions](metrics/birth_appearance_cv.md) describe nested selection,
ancestry grouping, conditional uncertainty and the mapping of shared fitter roles.

Submitted jobs: **1018745** (original, observed running on node26) and **1018746**
(relaxed, after successful descriptor-ready job1018568). Both run detached with
one GPU and eight CPUs. Allocation limits are12hours, not runtime estimates.
