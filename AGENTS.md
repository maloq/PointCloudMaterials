# Working rules

- Dataset split default (user instruction, 2026-10-13): new experiments use only
  train and test. Preserve the declared training ancestry; map every former
  selection/validation/calibration/evaluation role to test, without carving out
  a new inner selection set. Fit transforms on train only. Freeze training
  budgets, hyperparameters and checkpoint rules before test evaluation; do not
  use test scores for selection, calibration or early stopping. Keep original
  roles as provenance and preserve historical/running recipes and results.
  This default supersedes older multi-split guidance for new experiments.

- Ni relaxation default (user instruction, 2026-10-11): use a maximum absolute
  force-component tolerance of 0.05 eV/Angstrom for new Ni relaxations. Start
  new recipes from `configs/simulation/ni_relaxation_default.json` and record
  the explicit settings in each frozen plan. Keep float64 minimizer states and
  float32 GPU force arithmetic. Historical/running recipes and declared
  tolerance comparisons retain their own values; do not relabel cached minima
  or change existing training targets to the new tolerance.

- Autonomous experiment queues (user instruction, 2026-10-11): submit a concrete
  detached training queue and verify actual training/checkpoint progress before
  relying on any scheduled chat review. A review may steer follow-up work, but
  its absence, failure or thread lock must never prevent the baseline queue
  from running. Keep bounded checkpoint recovery, independent evaluations and
  visible failure receipts; proceed to other independent fits after a failure.

- Nucleus dataset cadence (user instruction, 2026-10-07):
  `al-nuclei-birth-030` means uniformly saved native 0.30-ps data only, excluding
  subsampled finer histories and mixed-cadence raw prefixes. Check actual saved
  timestep arrays, not only source metadata. The existing exact 0.10-ps view
  retains its explicit finer-native sources. Pre-birth crystalline tails are
  history annotations of existing events, not extra births or revised onset labels.
  See docs/datasets/nuclei_birth.md.

- Retired Al data (user instruction, 2026-10-07): the original 150 native
  0.75-ps Al production histories have been deleted. Do not select or regenerate
  this retired campaign for new experiments. Keep its source IDs, split ancestry,
  prepared liquids, native melt restarts and historical results. Production
  `final.restart.bin` files were also deleted at user request; preserve the
  `melt_final.restart.bin` files for possible future use. New birth studies
  use the consolidated 0.10/0.30-ps protocols separately; the old Al64 contract
  remains an ancestry/split reference, not an available raw-history benchmark.
  See docs/storage/al075_removal_20261007/README.md.

- GPU scheduling (user clarification, 2026-10-07): compatible GPU types are
  alternatives, not a requirement to occupy each type. Prefer a combined
  partition list per required worker instead of duplicate A100/A40/L40S
  reservations. Preserve the actual settings of running experiments.

- Motion/birth data (user instruction, 2026-10-07): use the consolidated
  `al-nuclei-birth-010` dataset for new nucleus-birth prediction, pinning its
  fingerprint and retaining source ancestry and original roles. The broad
  1,056,768-row structural collection is for spatial pretraining only; do not
  load it or use structural replay from it during new motion training. Reuse
  completed spatial checkpoints and the separate saved-position/velocity motion
  cohort. Preserve historical recipes and their true data use.
  New birth prediction on this consolidated catalogue must not add neighborhood
  crystallinity/component/distance filtering (user clarification,2026-10-07).
  Keep only actual input availability, exact observation offsets, ancestry and
  duplicate-identity checks; preserve the catalogue's birth definitions.

- Job ownership (user instruction, 2026-10-04): stop or change only jobs started
  in the current chat. A request to stop experiments does not authorize stopping
  simulations or jobs from other chats. Establish ownership from this chat's
  launch receipts before acting; shared Unix ownership is not sufficient.

- Al scope (user instruction, 2026-10-03): an unspecified request about Al means
  ALL Al data and related experiments, including older trajectories, shooting,
  dense 0.1/0.01 ps descendants, million-atom data and relaxed views. Use the
  shared Al overview/data pool instead of silently selecting one study. Report
  completed, running, queued and not-yet-analyzed portions explicitly. Never
  declare the Al work complete based only on one finished sub-study. Keep native
  cadences, targets, ancestry and historical releases explicit when combining
  records; one pool does not authorize interpolated histories or duplicated
  independent-sample counts. See docs/al_research_pool.md.

- Do not write new unit tests unless explicitly requested (user clarification,
  2026-10-11). Add only integration or end-to-end tests when needed, using the
  existing workflows. Do not create or maintain a `tests/` directory, move deleted
  tests elsewhere, or add test-suite dependencies to launchers.

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

- Physical-reconstruction encoder pretraining is discontinued (user instruction,
  2026-09-25). Omit that treatment and its dependent fits from future queues
  unless explicitly requested again. Preserve historical artifacts and keep
  physical-information readouts as evaluation diagnostics.

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
- GPU MEAM response integration uses float64 for new runs (user instruction,
  2026-10-06), with float64 states, momenta, tangents, noise and observables;
  GPU MEAM force arithmetic remains float32. Preserve the earlier 2026-10-04
  float32-state experiments and their true precision records. The backend's
  float64 scalar energy reduction is recorded separately.
- For the fixed Al256 MACE simulator-response case at 20/100 fs, use the
  float32 cuEquivariance/GPU-graph/batch4 profile in
  `configs/simulation/response_atlas_fast_default.json` by default (user instruction,
  2026-10-02). Reuse training response values with one independent execution audit;
  keep test streams disjoint. Historical frozen recipes retain their own precision.
  This default does not extend to other potentials, MD campaigns or longer horizons.
- MD-history experiments must use the same physical observation spacing and
  offsets across training and held-out sources (user instruction, 2026-09-26).
  Use separate protocols for different cadences and check actual source timelines.
  Explicit exception approved 2026-09-26: the nominal 0.75-ps six-frame run may
  use exact 0.70-ps observations for external sources saved every 0.10 ps.
  Record actual per-source intervals/spans; do not interpolate or label 0.70 ps
  as an exact 0.75 ps. The separate 0.10-ps experiment uses exact 0.10 ps throughout.
- Default new native spatial MACE encoders to 128 channels and a 128-dimensional
  exported embedding (user preference, 2026-09-25). Explicit capacity ablations
  retain their recorded widths; historical configurations are not rewritten.
- Keep new native MACE encoders geometry-only with one constant atom channel,
  including mixed-material pretraining (user instruction, 2026-09-25). Do not add
  species embeddings, element one-hot inputs, material IDs or scale features to
  the encoder/decoder. Use the established fixed material length normalization
  in preprocessing; retain material and potential as audit metadata. See
  docs/datasets/structural_multimaterial_256.md. Historical models keep their
  actual recorded input contracts.
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
  Frozen diagnostic probes, descriptor controls and per-checkpoint evaluation
  sweeps also stay local (user instruction, 2026-09-26). Do not create W&B runs
  for each evaluation or cached readout. Associated final evaluation metrics
  update the recorded training run through its existing ID, without restarting
  that run. Scientific encoder/predictor training remains online.
  Do not log duplicate wall-time fields or fixed values as W&B metric history
- Research correctness comes first: fail loudly with useful context. Trace values
  to their repository producer; use its actual types, shapes and fields. Avoid
  silent fallbacks, generic compatibility code and unnecessary defensive checks.
- Internal data comes from this repository's own producers (user instruction,
  2026-10-11). Inspect those producers and their callers before changing consumers;
  use the established contract directly instead of adding speculative `if`,
  `assert`, type/shape checks or fallback fields. Remove duplicate checks already
  guaranteed upstream. Keep checks for actual runtime uncertainty and scientific
  requirements such as cadence, ancestry, identity and numerical validity.
  Favor short, readable code and shared implementations over repeated branches.
- Remove obsolete ad hoc code and one-off tools after checking their callers
  (user instruction, 2026-10-11). Git history is the archive: if needed again,
  recover the previous implementation and rebuild it as maintained shared code.
  Preserve active workflows and recorded scientific results/provenance.
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
  named scientific bundles in `analyses/`, readable grouped `plots/` and `tables/`,
  and execution/provenance in `technical/`. Preserve historical producer paths;
  use the [result publication workflow](docs/research_results_system.md) to expose
  existing analyses without recomputing metrics or replacing frozen definitions.
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

- Reuse the shared relaxation index before scheduling new quenches (user request,
  2026-10-03). Use `src.data.relaxation_index.ensure_patches` for exact local
  observations, or the indexed `relax_frame` for a native full cell. Record the
  reused entry and preserve each experiment's ancestry, rows and precision.
  Global float16 cells do not replace pre-quantization float32 neighborhoods.
  See `docs/data_storage.md#unified-relaxation-index`; refresh historical
  producers with `python scripts/project.py relaxations scan`.

- Keep the six most recently used generated encoder feature caches across GPU
  lanes (user instruction, 2026-09-25). Protect active leases; evict only
  disposable caches. Keep checkpoints, predictions and metrics.

- Use `scripts/convert_trajectory.py` for conversions: new simulation positions are
  verified float16, boxes float32, identity/timeline arrays exact. Preserve LAMMPS
  integration/restart precision; record quantization error and checksums before deletion.
- Existing inputs/analysis use WORK, training caches IDS, new simulations SCRATCH;
  publish completed runs and stopped failures/restarts to STORE before SCRATCH purge.
  Keep machine paths in ignored `machine.local.yaml`; see [portability](docs/portability.md).
- Update the relevant index when adding a workflow; keep disposable diagnostics in
  the run output, with no live `scripts/archive/`.
