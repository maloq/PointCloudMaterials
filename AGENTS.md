# Working rules

- Do not create or maintain a `tests/` directory or write new automated tests
  (user instruction, 2026-09-25), unless the user explicitly requests them again.
  Do not move deleted tests elsewhere or add test-suite dependencies to launchers.

- Research objective (user instruction, 2026-09-25): study information useful for
  crystallization prediction, not maximum AP. Never introduce AP-specific losses,
  differentiable ranking/AP replay, AP-based checkpoint selection, promotion,
  hyperparameter search or fitted ensemble weights. AP at 3/6 ps remains an
  evaluation diagnostic, alongside proper predictive scores, calibration,
  information/readout controls, stability and noise response. Supervised fits and
  their frozen probes use declared predictive likelihood objectives/selectors;
  self-supervised encoders use their own label-free objectives/selectors. Preserve
  historical AP-trained artifacts with their true labels. See
  docs/encoder_research/training_branches.md.

- Prediction-context policy (user instruction, 2026-09-25): do not feed temperature,
  simulation age, absolute time, or other explicit time covariates to encoders,
  predictors, probes or baselines unless the user explicitly requests them again.
  Timestamps/horizons may still organize observations and labels; temperature may
  remain audit/split metadata. Audit the actual tensor producer, not just config
  names or comments. Record encoder and predictor inputs separately for every new
  experiment (spatial support/halo, history, motion, conditions, relaxation and
  training-only teachers). Preserve historical runs and their true input records;
  do not relabel them as condition-free. See docs/encoder_research/prediction_context.md.

- Use conda `pointnet-torch214`
- Default new native spatial MACE encoders to 128 channels and a 128-dimensional
  exported embedding (user preference, 2026-09-25). Explicit capacity ablations
  retain their recorded widths; historical configurations are not rewritten.
- Default new encoder training to batch_size=256 and microbatch=256 (user
  preference, 2026-09-25). Record explicit deviations; preserve the recorded
  settings of frozen/running experiments.
- Keep Weights & Biases enabled in online mode for all future training runs
  (user instruction, 2026-09-25). Use the existing teshbek/PointCloudMaterials
  project, stable resumable run IDs and local receipts. Do not disable tracking
  or silently switch to offline mode; report authentication/network failures.
  Log only scientific training runs and their associated evaluation metrics.
  Keep debug runs, smoke checks and hardware benchmarks local; do not create
  online runs for these checks.
- Research correctness comes first: fail loudly with useful context. Trace values
  to their repository producer; use its actual types, shapes and fields. Avoid
  silent fallbacks, generic compatibility code and unnecessary defensive checks.
- Read [scripts/README.md](scripts/README.md) before adding a command there. Reuse existing
  workflows and config/CLI arguments for run variations. Keep entry points thin;
  implementation belongs in `src/`. Preserve distinct scientific protocols.

# Where things belong

- `experiments/` is **only scientific research**: questions, protocols, findings,
  reproduction commands and configurations. No storage, portability, cleanup,
  environment checks, simulation campaigns or dataset inventories there.
- Put operational documentation in `docs/`, simulation records in `docs/simulations/`,
  active recipes in `configs/`, simulation recipes in `configs/simulation/`, and
  simulation locations in `configs/datasets.json`. Keep analysis templates; retire
  unused training/simulation variants with their dependency references.
  See [storage](docs/data_storage.md) and [simulations](docs/simulations/README.md).
- Follow [the result layout](docs/research_layout.md): `output/<question>/<run>/`,
  readable `plots/` and `tables/`, machine artifacts/logs in `technical/`.
- Export metric CSVs with `tables/METRICS.md` and implementation hashes through
  `src/experiment_runner/metric_docs.py`. Change `docs/metrics/` and `contracts.json`
  with calculations; preserve historical exported definitions.
- Before choosing training data or planning new data collection, consult
  [DATASETS.md](DATASETS.md). Refresh with `python scripts/project.py datasets
  --refresh` when availability matters. Register new collections and their
  known materials, generating potential, provenance and ancestry in
  `configs/datasets.json`; keep unknown metadata explicit.
- New matched Al encoder comparisons use the versioned
  `configs/fixed_cohort/al64_v1.json` source/sample contract; see
  [fixed datasets](docs/datasets/fixed_al64.md). Record the release identity and
  all64/legacy16 evaluation track; never resplit sources or drop model-specific
  evaluation rows. Keep existing runs on their recorded data. Structural fitting
  uses only train ancestors; selection is validation, calibration/test are excluded.

# Preservation and cleanup

- Use `scripts/convert_trajectory.py` for conversions: new simulation positions are
  verified float16, boxes float32, identity/timeline arrays exact. Preserve LAMMPS
  integration/restart precision; record quantization error and checksums before deletion.
- Existing inputs/analysis use WORK, training caches IDS, new simulations SCRATCH;
  publish completed runs and stopped failures/restarts to STORE before SCRATCH purge.
  Keep machine paths in ignored `machine.local.yaml`; see [portability](docs/portability.md).
- Update the relevant index when adding a workflow; keep disposable diagnostics in
  the run output, with no live `scripts/archive/`.
