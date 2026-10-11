# Nucleus birth transfer of response encoders

New October6 encoders and the newest native0.10ps lineage catalogues use the separate [dense birth evaluation](response_dense_birth.md); the original0.75ps results below retain their historical definitions.

The October4 diagnostic compares all12 frozen100/500fs FastAl response models
on the original unrelaxed preappearance cohort, including actual saved MD
velocities for QV. It retains the source-disjoint training, selection,
calibration and test roles. No simulation or encoder training is started.

```bash
python -m src.research.response_birth launch --config configs/analysis/response_birth_20261004.json
```

Launch requires the active two-GPU allocation. It reconstructs exact atom/velocity
correspondence, verifies checkpoint and cohort identities, freezes source and
configuration, then detaches two GPU lanes. Each lane extracts one checkpoint
and completes its eight small readouts before releasing its shared feature-cache
lease. Owned process IDs, allocation, GPU mapping, progress and failures are
recorded under `technical/`; other jobs are not modified.

The result is a named `analyses/nucleus-birth-v1` bundle under the existing
CUDA-MEAM-float32 response campaign. `README.md`, `tables/summary.csv`, paired
comparisons and current/history plots appear after both lanes finish. Fitted
readouts and immutable predictions live in `analyses/fits-v1`. Re-enter the
frozen `technical/code` directory and run `lane --lane N` with its `config.json`
to resume a failed owned lane with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the
established legacy e3nn loading environment); completed readouts verify and skip. Run `analyze`
only after both lanes complete. Diagnostics stay local under the tracking policy.
Final scores update the existing twelve training-run summaries, without new W&B
runs. The first launch failed before inference because the legacy e3nn loading
environment was missing; `technical/launch-attempt1` preserves that failure and
`technical/resume-1.json` records the environment-only restart in allocation
1020716, using GPUs 0 and 1. Scientific source and configuration were unchanged.

The current snapshot and8-frame history both end0.75ps before first observed
local crystal appearance; cases lead to isolated nucleus establishment. Whole
8A input neighborhoods are crystal-free. The15 held-out events from11 sources
form a retrospectively enriched case/control assay, not natural nucleation risk.

The shared [Al pool](al_research_pool.md) was consulted: earlier original/failed
appearances, shooting, dense and relaxed releases remain distinct. This run
uses the established isolated-birth subset. The sealed dense0.1ps assay has no
fixed-test ancestors; shooting has a different detection cadence; relaxed views
are outside the requested input domain. The all-cadence nucleus search remains
an independent running workflow.

[Recipe](../configs/analysis/response_birth_20261004.json) ·
[Metric and input definitions](metrics/response_birth.md) ·
[Original assay](birth_prediction.md)
