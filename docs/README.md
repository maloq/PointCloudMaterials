# Project documentation

- [GeoFrame checkpoint evolution and Ta/Zr structured-liquid regions](geoframe_evolution.md)

- [Encoder research handbook](encoder_research/README.md): GeoFrame-to-current training methods, evaluation protocols, searchable results database, analysis tools and workflow improvements.

- [Synthetic storage, LAMMPS CPU and GPU hardware benchmarks](hardware_benchmark.md)

- [Causal native MACE: atom-level geometry, motion and history](mace_causal.md)

- [Research glossary: terms specific to our local-state methods and comparisons](research_glossary.md)
- [Discarded frozen-state smoothness experiment](mace_local_smooth.md)
- [Current research and retained checkpoints](research_retention.md)
- [Archived research on STORE](archived_research.md)
- [Discarded frozen-encoder maps: scope and preserved artifacts](discarded_frozen_encoder_maps.md)
- [Active configs, analysis templates and retired recipes](../configs/README.md)
- [Data storage](data_storage.md) and [portable setup](portability.md)
- [Encoder-only transfer selection](encoder_transfer.md)
- [Simulations, collection catalog and run records](simulations/README.md)
- [Research/result layout and metric definitions](research_layout.md)
- [Run registry and retention commands](output_registry.md)
- [Maintained workflow details](workflows.md)
- [Shared-pretraining W&B metric layout](shared_pretraining_logging.md)
- [Current local structural GATr and MACE](shared_pretraining_local_structure_20260918.md)
- [Historical temporal-only backtracking continuation](shared_pretraining_temporal_backtracking_20260918.md)
- [Five-epoch mixed MACE with equivariant bond-order supervision](shared_pretraining_mace_bond_order_20260918.md)
- [Source refactor scope and validation](src_refactor.md)

Scientific experiment records belong in [experiments/](../experiments/README.md).
Operational changes, simulation campaigns and dataset inventories belong in this
`docs/` tree; they are not new experiments.
- [Discarded frozen-map consecutive-motion experiment](mace_local_motion.md)

- [Native MACE data-amount study](mace_data_amount.md): actual encoder training across independent-source counts on the allocated H100.

- [H200 causal MACE experiment and portable handoff](mace_causal_h200.md)

- [Causal MACE GPU execution and expanded-data handoff](mace_causal_runtime.md).

- [Expanded-data MACE on both node61 GPUs](shared_pretraining_mace_expanded_dual_20260919.md): immutable dynamic sampling, complete TDA, global two-GPU replay and detached dependency.

- [Two-GPU MACE pipeline optimization](shared_pretraining_mace_dual_optimization_20260919.md): process prefetch and verified checkpoint continuation.

- [Expanded MACE crystallization transfer execution](crystallization_transfer_20260919.md).
- [Literature-guided frozen-MACE forecast queue](crystallization_followup.md).

[Structural-path crystallization queue](crystallization_paths_20260919.md): detached local state trajectory forecasts and future-center cache.

- [Neighborhood JEPA execution](neighborhood_jepa_20260920.md): tracked-cache construction, frozen two-H100 MACE queue, exact resumes and model exports.

- [Neighborhood JEPA v2 execution](neighborhood_jepa_v2.md)

- [Neighborhood JEPA regularization queue](neighborhood_jepa_regularization.md): detached multi-allocation fits, order anchors and frozen crystallization probes.

- [Liquid geometry diagnostic workflow](liquid_geometry.md): frozen inputs, CPU metric experiments and native GPU checkpoint inference.

- [Structural-state encoder training and distance/future factorial](structural_state.md): verified data reuse, detached queues and automatic matched evaluation.

- [Broad encoder snapshot screen](encoder_screen.md): queue, cached analyses, and native input contracts.

[Encoder parameter-search queue](encoder_parameter_search.md): frozen28-fit campaign, convergence assays and8×spatial plots.

- [Paired MACE + Epi](mace_epi.md): submit/resume the matched24-pass encoder comparison and native structural/future analysis.
