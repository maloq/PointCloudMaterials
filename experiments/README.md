# Current research

Only scientific questions, protocols, findings and reproduction configurations
belong here. Simulation campaigns and inventories are in [docs/simulations](../docs/simulations/README.md);
storage, portability and maintenance are in [docs](../docs/data_storage.md).

The [September 11–13 retention review](../docs/research_retention.md) lists retained
results and necessary checkpoints. [Older research and outputs are on STORE](../docs/archived_research.md).
The dated folder name is the start date; older experiments below support current research.

| Research | Current evidence / purpose |
| --- | --- |
| [Embedding forecast](embedding_forecast_20260911/README.md) | Direct versus autoregressive trajectories, completed pilot and enlarged fits; exact continuation and target-encoder provenance |
| [Observed history](forecast_context_20260913/README.md) | [24 completed fits](forecast_context_20260913/RESULTS.md), matched 0–24 ps history comparison |
| [Local crystallization](forecast_crystallization_20260913/README.md) | [Completed physical assay](forecast_crystallization_20260913/RESULTS.md), transition/timing readouts and [next questions](forecast_crystallization_20260913/NEXT_EXPERIMENTS.md) |
| [Spatial context and mixtures](forecast_spatial_mixture_20260913/README.md) | Active matched 12-fit study and [14 new short-history fits](forecast_spatial_mixture_20260913/SHORT_HISTORY.md); current paired reference fits retained |
| [Relaxed MACE / VICReg](mace_vicreg_relaxed_20260910/README.md) | [September 11 comparison](mace_vicreg_relaxed_20260910/RESULTS_20260911.md); supplies the frozen target encoder for every forecast |
| [Full-data MACE / VICReg](mace_vicreg_full_20260910/README.md) | September 11 analysis and retained full-data reference checkpoint |
| [Al topology denoising](mace_al_denoising_20260910/README.md) | Older dependency: selected independent sources, relaxation targets, frozen MACE model and controlled denoising results |

The [ideas backlog](ideas.json) retains research questions. The file-backed
[run dashboard](../output/registry/index.html) indexes results; its root settings
live in [configs/experiment_registry.json](../configs/experiment_registry.json).
See [registry documentation](../docs/output_registry.md) and [result conventions](../docs/research_layout.md).
Scientific implementation is in [src/research](../src/research/README.md) and the
maintained [training/analysis commands](../scripts/README.md).
