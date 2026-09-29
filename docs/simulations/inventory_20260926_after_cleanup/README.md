# Simulation holdings

Observed 2026-09-26T14:56:03.455976+00:00 to 2026-09-26T14:56:40.658974+00:00.

[Collection CSV](collections.csv) · [Trajectory and snapshot CSV](trajectories.csv)

Sizes are measured allocated disk bytes (GiB = 2^30 bytes), including native restarts, logs and provenance. Nested registered collections are charged to their own rows; directory aliases are not followed. Different physical copies remain included. Sizes of active runs change during observation.

Lengths are last minus first saved time, not planned durations. Sampling is measured from binary/NPZ timelines and the nearest unambiguous producer timestep. Text cadence is declared; complete text frame counts are not rescanned. Unknown means the available evidence is insufficient.

Counts are saved representations, not independent simulations: precision variants, checkpoint prefixes, duplicate exports and shared parent lineages must not be summed as independent runs. Velocities refer to sampled trajectories; restart files and isolated ASE snapshots can contain momenta separately. Large arrays are not rehashed; duplicate groups use producer-declared array hashes.

| Collection | Material | Series exports | Atoms | Saved length (ps) | Sampling (ps) | Velocities | Format | Disk GiB | Classification |
|---|---|---:|---:|---|---|---|---|---:|---|
| .al_liquid_source_70304_mpa_seed12345_500K.generation-2b3c58f003ec | Al | 1 | 70304 | 3 | 0.1 | no | NPZ float32 | 0.044 | review_required |
| .al_liquid_source_70304_mpa_seed12346_500K.generation-6d03498342ff | Al | 1 | 70304 | 3 | 0.1 | no | NPZ float32 | 0.044 | review_required |
| .al_phase_context_70304x1.generation-5016671fb856 | Al | 3 | 70304 | 1, 3, 5 | 0.1 | no | NPZ float32 | 0.113 | review_required |
| .al_phase_context_70304x1_seed_12346.generation-559b40e8ef01 | Al | 3 | 70304 | 1, 3, 5 | 0.1 | no | NPZ float32 | 0.113 | review_required |
| al-birth-uniform-20260926 | Al | 2 | 70304 | 228 | 0.15 | yes | LAMMPS text, NPY float16, NPY float32 | 10.906 | building |
| al-birth-uniform-20260926-source000-T400 | Al | 2 | 70304 | 240 | 0.15 | yes | NPY float16, NPY float32 | 5.201 | unreviewed |
| al-birth-uniform-20260926-source001-T410 | Al | 2 | 70304 | 352.5 | 0.15 | yes | NPY float16, NPY float32 | 7.568 | unreviewed |
| al-birth-uniform-20260926-source002-T420 | Al | 2 | 70304 | 277.5 | 0.15 | yes | NPY float16, NPY float32 | 5.990 | unreviewed |
| al-birth-uniform-20260926-source003-T430 | Al | 2 | 70304 | 202.5 | 0.15 | yes | NPY float16, NPY float32 | 4.408 | unreviewed |
| al-birth-uniform-20260926-source004-T440 | Al | 2 | 70304 | 201 | 0.15 | yes | NPY float16, NPY float32 | 4.380 | unreviewed |
| al-birth-uniform-20260926-source005-T450 | Al | 2 | 70304 | 189 | 0.15 | yes | NPY float16, NPY float32 | 4.127 | unreviewed |
| al-birth-uniform-20260926-source006-T460 | Al | 2 | 70304 | 142.5 | 0.15 | yes | NPY float16, NPY float32 | 3.149 | unreviewed |
| al-birth-uniform-20260926-source007-T470 | Al | 2 | 70304 | 199.5 | 0.15 | yes | NPY float16, NPY float32 | 4.348 | unreviewed |
| al-birth-uniform-20260926-source010-T500 | Al | 2 | 70304 | 129 | 0.15 | yes | NPY float16, NPY float32 | 2.865 | unreviewed |
| al-eam-six-24ps | Al | 6 | 1048576 | 24 | 0.1 | no | NPY float16 | 8.506 | research |
| al_520K_remaining24_20260913T204114Z | Al | 24 | 70304 | 600 | 0.75 | yes | NPY float16 | 21.005 | research |
| al_homogeneous_campaign_16384_compiled_mpa_110ps_multiseed_20260717 | Al | 2 | 16384 | 86, 88 | 1 | no | NPZ float32 | 0.049 | review_required |
| al_homogeneous_campaign_16384_compiled_mpa_test_4h_20260715 | Al | 6 | 16384 | 185, 187 | 1 | no | NPZ float32 | 0.457 | review_required |
| al_homogeneous_campaign_70304_mpa_110ps_source12345_seed35803_20260718 | Al | 4 | 70304 | 114, 115 | 1 | no | NPZ float32 | 0.402 | review_required |
| al_homogeneous_campaign_70304_mpa_110ps_source12346_seed35831_20260718 | Al | 2 | 70304 | 85, 86 | 1 | no | NPZ float32 | 0.212 | review_required |
| al_homogeneous_campaign_70304_mpa_130ps_source12345_seed35803_20260720 | Al | 4 | 70304 | 5, 130, 134, 135 | 1 | no | NPZ float32 | 1.067 | review_required |
| al_homogeneous_campaign_70304_mpa_130ps_source12346_seed35831_20260720 | Al | 4 | 70304 | 5, 130, 134, 135 | 1 | no | NPZ float32 | 1.068 | review_required |
| al_homogeneous_campaign_70304_mpa_bf16_130ps_source12345_seed35803_20260724 | Al | 4 | 70304 | 5, 130, 134, 135 | 0.333;0.334 | no | NPZ float32 | 1.316 | review_required |
| al_homogeneous_campaign_70304_mpa_bf16_130ps_source12346_seed35831_20260724 | Al | 4 | 70304 | 5, 130, 134, 135 | 0.333;0.334 | no | NPZ float32 | 1.316 | review_required |
| al_homogeneous_campaign_70304_mpa_bf16_comparison_20260724 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.010 | review_required |
| al_homogeneous_lammps_eam_70304_110ps_4seeds_20260826 | Al | 8 | 70304 | 110, 115 | 1 | no | NPY float32, NPZ float32 | 1.004 | review_required |
| al_homogeneous_unseeded_2nn_meam_70304_12temps_390-500K_600ps_1ps_positions_velocities_6seeds_20260830 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.084 | review_required |
| al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.046 | retired |
| al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.139 | retired |
| al_homogeneous_unseeded_2nn_meam_70304_4temps_510-540K_600ps_1ps_positions_velocities_6seeds_20260830 | Al | 1 | 70304 | 600 | 1 | separate text, yes | LAMMPS text, LAMMPS velocity text, NPZ float32 | 9.993 | review_required |
| al_homogeneous_unseeded_2nn_meam_70304_500K_600ps_7velocityseeds_20260827 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.055 | retired |
| al_homogeneous_unseeded_2nn_meam_70304_500K_999ps_velocity35803_20260827 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.008 | retired |
| al_independent_sources_recovery_20260909T203221Z | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | review_required |
| al_liquid_source_70304_mpa_seed12345_500K | Al | 1 | 70304 | 3 | 0.1 | no | NPZ float32 | 0.051 | review_required |
| al_liquid_source_70304_mpa_seed12346_500K | Al | 1 | 70304 | 3 | 0.1 | no | NPZ float32 | 0.051 | review_required |
| al_meam_1m_450K_400ps_20260913T205149Z | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.001 | retired |
| al_meam_1m_450K_400ps_with_melt_20260913T205405Z | Al | 2 | 1000000 | 300, 400 | 0.1 | no | NPY float16 | 69.520 | review_required |
| al_meam_crystallization_100k_450K_20260911 | Al | 0 | 100000 | unknown | 0.1 | no | LAMMPS text | 23.900 | review_required |
| al_meam_crystallization_1m_450K_20260913T203458Z | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.001 | retired |
| al_meam_crystallization_1m_source_only_450K_20260913T203616Z | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.001 | retired |
| al_meam_independent_sources_70304_400-500K_30perT_float16_20260902 | Al | 90 | 70304 | 600 | 0.75 | yes | NPY float16 | 58.304 | research |
| al_meam_independent_sources_70304_510-520K_30perT_float16_20260903 | Al | 36 | 70304 | 600 | 0.75 | yes | NPY float16 | 23.324 | research |
| al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.002 | prepared |
| al_meam_nested_shooting_pilot_70304_400-500K_20260902 | Al | 146 | 70304 | 0.9–72 (varies) | 0.03;0.099;0.102, 0.03;0.099;0.102;0.3 | yes | NPY float32 | 19.823 | review_required |
| al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible | Al | 144 | 70304 | 24 | 0.3 | yes | NPY float16 | 9.665 | review_required |
| al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904 | Al | 160 | 70304 | 15 | 0.3 | yes | NPY float32 | 13.713 | research |
| al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831 | Al | 40 | 70304 | 48 | 0.3 | yes | NPY float32 | 10.448 | research |
| al_meam_position_shooting_70304_400-500K_48ps_2shots_local_followup_20260831 | Al | 80 | 70304 | 48 | 0.3 | yes | NPY float32 | 20.760 | research |
| al_meam_position_shooting_70304_400-500K_48ps_40branches_local_20260901 | Al | 40 | 70304 | 48 | 0.3 | yes | NPY float32 | 10.448 | research |
| al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903 | Al | 40 | 70304 | 48 | 0.3 | yes | NPY float32 | 10.450 | review_required |
| al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_protocol_flag_correction | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.003 | prepared |
| al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_window_spec | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.003 | prepared |
| al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830 | Al | 320 | 70304 | 48 | 0.3 | yes | NPY float32 | 82.639 | research |
| al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904 | Al | 20 | 70304 | 8.7, 12, 15, 24 | 0.3 | yes | LAMMPS text, NPY float32 | 2.076 | fixture |
| al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903 | Al | 17 | 70304 | 12, 48 | 0.3 | yes | NPY float32 | 4.194 | fixture |
| al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descriptor_preparation | Al | 1 | 70304 | 12 | 0.3 | yes | NPY float32 | 0.069 | fixture |
| al_mpa_70304_bf16_validation_20260724 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | review_required |
| al_phase_context_70304x1 | Al | 3 | 70304 | 1, 3, 5 | 0.1 | no | NPZ float32 | 0.161 | review_required |
| al_phase_context_70304x1_seed_12346 | Al | 3 | 70304 | 1, 3, 5 | 0.1 | no | NPZ float32 | 0.161 | review_required |
| al_relaxed_cells_expanded_20260917 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 1.877 | building |
| memory-al-precision-20260917-source000-T500 | Al | 2 | 70304 | 192 | 0.075 | yes | NPY float16, NPY float32 | 8.144 | review_required |
| memory-al-precision-20260917-source001-T500 | Al | 2 | 70304 | 192 | 0.075 | yes | NPY float16, NPY float32 | 8.145 | review_required |
| memory-al-precision-20260917-source002-T500 | Al | 2 | 70304 | 192 | 0.075 | yes | NPY float16, NPY float32 | 8.145 | review_required |
| memory-al-precision-20260917-source003-T500-failed | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | retired |
| memory-al-precision-20260917-source004-T500-failed | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.001 | retired |
| memory-al-precision-campaign-20260917 | Al | 0 | unknown | unknown | unknown | unknown | no time series found | 0.008 | mixed |
| mg-eam-six-24ps | Mg | 6 | 1048576 | 24 | 0.1 | no | NPY float16 | 8.506 | research |
| ta-eam-five-24ps | Ta | 5 | 10000422 | 24 | 0.1 | no | NPY float16 | 79.442 | research |
| ta-shooting-20260926 | Ta | 0 | unknown | unknown | unknown | unknown | no time series found | 0.226 | research |
| ta-shooting-20260926-parent00 | Ta | 4 | 1024000 | 24 | 0.1 | no | NPY float16 | 9.240 | unreviewed |
| ta-shooting-20260926-parent01 | Ta | 0 | 10000422 | unknown | 0.1 | no | LAMMPS text | 61.872 | active_or_stopped |
| ta-shooting-20260926-parent02 | Ta | 0 | 10000422 | unknown | 0.1 | no | LAMMPS text | 16.355 | active_or_stopped |
| ta_initial_model_1m_24ps_npt_20260905 | Ta | 2 | 1024000 | 1, 24 | 0.1 | no | LAMMPS text, NPY float16 | 1.846 | research |
| portability-ti-smoke-20260913 | Ti | 1 | 128 | 0.02 | 0.005 | no | NPY float16 | 0.000 | fixture |
| ti-meam-early-slurm-copies | Ti | 6 | 100000 | 240 | 0.1 | no | NPY float16 | 8.463 | duplicate |
| ti-meam-shooting-round2 | Ti | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | review_required |
| ti-meam-source-and-six-branches | Ti | 7 | 100000 | 240, 724 | 0.1 | no | NPY float16 | 14.020 | research |
| ti_ta_crystallization_20260907 | Ti | 7 | 128 | 0.2, 0.6 | 0.1 | no | LAMMPS text, NPY float32 | 0.651 | mixed |
| interrupted_attempts | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.001 | retired |
| local_overnight_lamedell11_20260905 | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | review_required |
| local_overnight_lamedell11_20260905.startup_attempt_010736 | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | retired |
| nested_shooting_prepare_logs | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | administrative |
| nonshooting_float32_migration_20260901 | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.000 | administrative |
| polycrystalline_balanced_geometries | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.111 | research |
| polycrystalline_balanced_geometries_v2 | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.150 | research |
| restart_boundary_audit_20260905 | unknown | 0 | 70304 | unknown | 0.3 | yes | LAMMPS text | 0.300 | administrative |
| shooting_float32_migration_20260901 | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.014 | administrative |
| superseded_preparation_al_meam_position_shooting_15ps_topup_manifest_metadata_20260904T0955Z | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.003 | prepared |
| superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z | unknown | 0 | unknown | unknown | unknown | unknown | no time series found | 0.001 | fixture |

Total allocated space across these owned collection rows: **687.282 GiB**.

Scope: local registered simulation collections, fixtures and detected active Ta runs. Static archives and training/feature caches are excluded. Remote H200 copies are not locally verifiable and are not added. No historical datasets are deleted or reclassified by this inventory.
