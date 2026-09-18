# Working rules

- Use conda `pointnet-torch214`
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

# Preservation and cleanup

- Use `scripts/convert_trajectory.py` for conversions: new simulation positions are
  verified float16, boxes float32, identity/timeline arrays exact. Preserve LAMMPS
  integration/restart precision; record quantization error and checksums before deletion.
- Existing inputs/analysis use WORK, training caches IDS, new simulations SCRATCH;
  publish completed runs and stopped failures/restarts to STORE before SCRATCH purge.
  Keep machine paths in ignored `machine.local.yaml`; see [portability](docs/portability.md).
- Update the relevant index when adding a workflow; keep disposable diagnostics in
  the run output, with no live `scripts/archive/`.