# Simulation and synthetic-data generation

This directory contains configurations that create atomistic or synthetic temporal
datasets.

- `atomistic/al/` contains Al MLIP, MD, phase-transition, crystallization, and
  potential-validation workflows.
- `temporal/` contains procedural temporal simulations.

Training and analysis loaders for already-generated datasets live under
`configs/data/loaders/`.

Fresh Al/Ti/Ta source/branch recipes are `al_crystallization.json`, `ti_crystallization.json`, and `ta_crystallization.json`. Launch through `elemental run --config CONFIG --run-name NAME`; new results start under the machine simulation_runs root and completed runs publish to its archive root. See [portable execution](../../docs/portability.md).
