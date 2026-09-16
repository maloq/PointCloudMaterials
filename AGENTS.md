# Working rules

- Use conda `pointnet` when available.
- Research correctness comes first: fail loudly with useful context. Trace values
  to their repository producer; use its actual types, shapes and fields. Avoid
  silent fallbacks, generic compatibility code and unnecessary defensive checks.
- Use the [research glossary](docs/research_glossary.md) when explaining methods
  and results: link relevant entries and briefly explain unfamiliar terms. Add
  or update terms central to our research and their protocol-specific
  meanings if asked to clarify; omit general textbook entries such as PCA/UMAP. Preserve distinctions
  between protocols and their exact metric definitions.
- Read [scripts/README.md](scripts/README.md) before adding a command. Reuse existing
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

# Preservation and cleanup

- Before retiring files, check imports, tests, configs, documentation and submitted
  jobs. Preserve uncommitted research and immutable source/manifests/resume configs.
  Inspect Slurm controllers before moving launchers; retain exact forwarding paths
  while any submitted job still needs them. Do not edit external job files.
- Archive and verify old results before removal; keep current research and required
  checkpoints, exact-resume state, paired test data and simulation restarts.
  Preview cleanup with `experiment_registry.py storage` / `clean`; size alone never
  makes a file disposable. See [the archive](docs/archived_research.md).
- Use `scripts/convert_trajectory.py` for conversions: new simulation positions are
  verified float16, boxes float32, identity/timeline arrays exact. Preserve LAMMPS
  integration/restart precision; record quantization error and checksums before deletion.
- Existing inputs/analysis use WORK, training caches IDS, new simulations SCRATCH;
  publish completed runs and stopped failures/restarts to STORE before SCRATCH purge.
  Keep machine paths in ignored `machine.local.yaml`; see [portability](docs/portability.md).
- Update the relevant index when adding a workflow; keep disposable diagnostics in
  the run output, with no live `scripts/archive/`.
