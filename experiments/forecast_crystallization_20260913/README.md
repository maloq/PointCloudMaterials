# Do embedding forecasts predict a local environment becoming crystal?

The question is **when the tracked local center atom becomes crystalline**, as
clarified on 2026-09-13. The primary assay is local PTM, not the first nucleus
anywhere in a simulation. Completed results and interpretation are in [RESULTS.md](RESULTS.md).
The prioritized follow-up proposal is in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md).

Configuration: [technical/local_config.json](technical/local_config.json).
Results: [local-crystallization-20260913](../../output/embedding_forecast/local-crystallization-20260913/).
The primary physical definition is the start of three consecutive crystalline PTM
frames (FCC/HCP/BCC, RMSD cutoff 0.1), with five/nine-frame sensitivity analyses.
Cadence is 0.75 ps. A fixed random subset of 64 existing embedded atom centers in
each of 125 independent simulation sources gives 8,000 local trajectories. The
existing whole-source split is retained: 74 train, 24 validation, 27 test.

A class-balanced linear crystal readout is trained on observed training embeddings;
regularization uses validation only. Frozen large AR and direct embedding predictors
use 6 ps observed history and forecast 12 samples through 9 ps. Separate validation
thresholds maximize F1 for future state and upcoming local transition. Comparators
are last embedding, history mean, linear extrapolation, and an observed-future
readout diagnostic. Test set thresholds are never fitted. The detailed definitions,
denominators, censoring, timing and source intervals are in
[the metric contract](../../docs/metrics/forecast_crystallization.md).

Run from the repository root in conda `pointnet`, in this order:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OVITO_THREAD_COUNT=1 \
python -m src.research.forecast_crystallization.local_data \
  --config experiments/forecast_crystallization_20260913/technical/local_config.json
python -m src.research.forecast_crystallization.local_data --verify-patches \
  --config experiments/forecast_crystallization_20260913/technical/local_config.json
python -m src.research.forecast_crystallization.local_predict \
  --config experiments/forecast_crystallization_20260913/technical/local_config.json
python -m src.research.forecast_crystallization.local_analyze \
  --config experiments/forecast_crystallization_20260913/technical/local_config.json
```

Preparation and prediction require a fresh output directory for a fresh attempt;
change only the config's output path when reproducing. CPU preparation uses 16
workers with one PTM thread each. Forecast inference uses the active H100 allocation.
The selected checkpoints, local labels/atom IDs, projected forecast paths, probe,
configs, hashes and logs remain under the output's `technical/`. Plots and metric
CSVs have their own `plots/` and `tables/`, with frozen definitions at export.

The preliminary global-onset branch is retained for provenance in
`crystallization-assay-20260913` and `technical/config.json`. It prepared global
progress and checkpoint snapshots, then stopped on explicit logistic-regression
nonconvergence. No global prediction result was produced. The user's local-event
clarification superseded that question. `data.py` and `predict.py` retain that
separate exploratory implementation; the local protocol uses the three commands
above and the established closed-form local structure readout.

New files are experiment-specific reusable implementation under
`src/research/forecast_crystallization/`, this dated research record, metric
documentation/contracts and focused tests. Generated launch records, labels,
checkpoints, predictions, figures and logs are run artifacts, not new scripts.
