# How much independent data does the native local-state encoder need?

Train MACE itself end to end under the new direct motion constraints. There is no
fitted transformation after the encoder. Compare 10, 25, 45 and 90 independent Al
trajectories, two repeats, with fixed validation/development-test sources and an
equal optimizer-update budget. All start at the original common pretrained MACE;
none starts from the encoder already adapted to all current training trajectories.

Main embedding is the native 256 pooled structure channels, accompanied by the
32 activity and 16 flow channels of the velocity extension. Smoothness constraints
act directly on the native structure channels and backpropagate into MACE. Shared
auxiliary physical and direction heads train alongside the encoder. No forecasting
objective or fitted embedding map is used.

Evaluate physical information (bond order, instantaneous H0/H1/H2 topology,
velocity observables), 0.75 ps jumps, temporal bending, and held-out increment
energy captured by eight state-dependent directions. Separate low-order local
groups and normalize jumps also by within-context spread to expose apparent gains
caused only by liquid/crystal separation. Preserve train/held-out errors and both
seed curves. Primary results use the final common update, with validation-selected
checkpoints retained separately.

This is a short compute-matched study conditional on original MLIP pretraining.
It does not estimate fully converged sample complexity. Two repeats are a pilot,
and existing development-test sources are not a new blind confirmation population.

Recipe: [mace_data_amount.json](../../configs/analysis/mace_data_amount.json).
Exact definitions: [metrics](../../docs/metrics/mace_data_amount.md).
Execution and artifacts: [run guide](../../docs/mace_data_amount.md).
Terminology: [independent-source learning curve](../../docs/research_glossary.md#independent-source-learning-curve).

```bash
python -m src.research.mace_velocity data-prepare --config configs/analysis/mace_data_amount.json
python -m src.research.mace_velocity data-smoke --config configs/analysis/mace_data_amount.json
python -m src.research.mace_velocity data-study --config configs/analysis/mace_data_amount.json
```
