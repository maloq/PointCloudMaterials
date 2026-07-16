# Dataset loading

This directory contains only configurations consumed by training, evaluation, or
analysis when loading an existing dataset.

- `loaders/` defines dataset paths, sampling, neighborhood, normalization, and cache
  settings.
- Dataset generation and molecular simulation recipes live under
  `configs/simulation/`.

Do not add MD potentials, integrators, temperatures, replica schedules, or simulation
output controls here.
