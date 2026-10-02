# Failed-embryo appearance comparison

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
