# Development workflow

Use conda `pointnet-torch214` for scientific work. Keep new commands in the
[command index](../scripts/README.md), implementation in `src/`, and variations
in configs. Preserve frozen recipes and numerical contracts when refactoring.

This is a closed-loop research repository: use the types, fields and shapes our
producers actually emit. Prefer direct field access and library operations over
validation wrappers, coercion fallbacks and repeated assertions. Keep a check only
for a plausible failure not already caught upstream; put it where that uncertainty
enters, rather than repeating it in every consumer or training batch.

## Small CPU check

The development profile installs only what the integration check needs. It does
not replace the [scientific environment](../environments/README.md).

```bash
python3.12 -m venv .venv-check
source .venv-check/bin/activate
python -m pip install -r environments/requirements-check.txt
python -m pip check
python scripts/project.py check
```

The same command runs in GitHub Actions. It compiles all Python sources, parses
JSON/YAML recipes, imports shared components, compares local and Slurm command
rendering, checks shell syntax and CLI entry points, and runs the CPU example
below. Hydra expressions stay unresolved during syntax checks. No scheduler jobs,
external datasets or W&B runs are involved. Receipts and logs go in a fresh
`output/maintenance/checks/run-*/` directory; `--output DIRECTORY` changes its parent.

Ruff checks syntax, undefined names and a narrow set of correctness rules across
all of `src/` and `scripts/`. Scoped annotations cover verified closure-lifetime
false positives where a model is explicitly released after its last use. This
fast check catches coding mistakes; scientific workflows still need their own
integration checks.

## Complete reference example

```bash
python scripts/project.py example --output output/examples/cpu-first
```

The output directory must be new. The recipe is
[`configs/examples/cpu_pipeline.json`](../configs/examples/cpu_pipeline.json);
`--config FILE` selects an explicit variant. The implementation is
[`src/project_runtime/example.py`](../src/project_runtime/example.py).

It creates 32 independent synthetic point clouds, runs the real native Q/V
converter and binary reader, and trains the retained MLP encoder with a small
regression head for six CPU updates. Normalization is fitted on the 24 training
rows. The eight test rows are evaluated only after the fixed budget; they never
select a checkpoint or hyperparameter. This checks software integration, not
material physics or scientific performance.

The example saves an intermediate checkpoint after three updates, restores model,
optimizer and random states, and requires exact agreement with uninterrupted
training, including dropout, sampled batches and losses. It reloads the final
checkpoint before evaluation. The output's `technical/` directory contains
`config.json`, `inputs.json`, `interrupted.pt`, `last.pt` and `verification.json`.
Metrics are in `tables/metrics.csv`, with definitions and source receipts. Scores
may differ across environments; resume agreement is checked within one.

## Failures and explicit continuation

The generic runner makes one attempt per fit. NaNs and other failures produce
failure receipts; it never retries with a changed learning rate. Use
`--continue-on-error` to run independent fits after a failure, and the existing
`--resume OUTPUT` workflow for a deliberate continuation. Historical reproduction
commands retain their original flags; omit removed `--nan-restart-*` options when
adapting an old command to current code.

## Lightweight metric provenance

Training and export do not compare source hashes against a manually maintained
allowlist. Export records the current metric definitions and available declared
source hashes. Each table points to its immutable provenance receipt. Changed
producer revisions preserve previous tables and their definitions; historical
receipts are not relabeled. A capture records what ran, not scientific approval.
Use the shared writers in `src/experiment_runner/metric_docs.py` for new tables.

Review registry drift explicitly when changing a calculation or preparing a
release:

```bash
python scripts/experiment_registry.py metrics-docs
python scripts/experiment_registry.py metrics-docs --family research_example --strict
```

The first reports differences without failing. Only `--strict` makes drift a
nonzero exit. Missing declared source files are recorded as missing; real I/O
errors and retired-family misuse still fail. Scientific checks for finite values,
ancestry, cadence and identity remain in their producers. Frozen plans with
explicit source pins retain their own reproduction checks. See the
[results system](research_results_system.md) for publication and preserved exports.
