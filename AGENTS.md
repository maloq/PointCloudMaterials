use conda environment `pointnet` if avilable

## No silent failures / no ambiguity

This is research code: correctness and clarity beat cleverness.

- **Never fail silently.** No empty `except`, no “best-effort” fallbacks, no ignoring return codes, no `pass` on errors.
- **Make errors loud and informative.** Raise explicit exceptions with actionable messages; include context (inputs, shapes, units, paths, assumptions).
- **Be explicit, not ambiguous.** Prefer readable code over implicit magic; avoid unclear defaults and side effects.
- **Validate inputs + invariants.** Assert/guard preconditions and key assumptions early (types, ranges, dimensions, units).
- **If uncertain, stop and say so.** Don’t guess—surface the uncertainty and propose a safe, checkable approach.

Silent errors are worse than crashes. Crashes with good messages are acceptable.

Do not add unnecessary checks, like any data checks in functions that only used once. I'm serious, If the code works than we don't need any checks, it works. It's a code for research to be use once

This repository is a closed loop: before handling a value, trace it to the repository-owned dataset or producer and use its concrete type, shape, and required fields directly.
Do not add generic broadcasting, arbitrary iterable/scalar coercion, compatibility fallbacks, or sentinel replacements for hypothetical external inputs that this repository never produces.

## Commands, experiments, and generated files

- Before creating a script, read `scripts/README.md` and inspect the existing
  command and producer for that workflow. Reuse them when the method is unchanged.
- Dataset, checkpoint, seed, temperature, horizon, and output-path changes belong
  in configuration or existing command arguments, when supported by that method.
  Do not copy a runner for another run. Preserve explicit differences in scientific
  protocols; do not merge different objectives or restart semantics behind defaults.
- `scripts/` contains maintained entry points only. Add a command only for a
  distinct reusable workflow, document its inputs and implementation in
  `scripts/README.md`, and use the existing family command when applicable.
- Shared scientific implementation and orchestration belong in the relevant
  `src/` package. Do not import implementation from `scripts/`, and do not add
  imports between command scripts. Keep entry points small.
- Keep experiment-specific code under `experiments/<topic>_<YYYYMMDD>/`, with a
  README stating the research question, configuration, reproduction command,
  output location, and findings (or a link to the research report). Date experiment
  records, not maintained command names. Keep reproducibility code versioned.
- Put disposable diagnostics, generated job scripts, logs, and results in the
  run's output directory. Do not create a live `scripts/archive/` or leave a new
  diagnostic in `scripts/` at task completion.
- Before retiring code, check imports, tests, configuration, documentation, and
  generated-job references. Update repository references together. Do not delete
  uncommitted research code or alter existing external job files during cleanup.
- Before moving or removing simulation launchers, inspect the live Slurm queue
  and the batch scripts used by active controller chains. Preserve their exact
  launcher paths with temporary forwarding entry points until the campaigns
  finish. Updating repository references alone does not preserve submitted jobs.
- Use `scripts/convert_trajectory.py` for the supported repository format
  conversions. Extend its format implementation for a new repository producer;
  do not create another migration script. Preserve verification and provenance.
- At task completion, identify new files as maintained tools, experiment records,
  or disposable diagnostics. Update the relevant index when adding a workflow.

## Simulation trajectory storage (user preference, 2026-09-07)

Store positions from new simulations as verified float16 trajectory artifacts.
Keep box bounds in float32, integer identity/timeline arrays exact, and LAMMPS
integration and restart precision unchanged. Record quantization error and
checksums before removing larger position exports. Use the maintained conversion
commands; update a producer's storage path when launching it if necessary.
