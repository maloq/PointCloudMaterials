# Maintained scripts

This directory contains reusable repository entry points only. Dated campaign
launchers, fixed-checkpoint analyses, and superseded conversion utilities belong
in git history rather than a live `archive/` directory.

## Experiments and plots

- `run_experiments.py`: run, resume, or collect an experiment plan.
- `plot_experiment_summary.py`: render experiment-runner summary JSON.
- `plot_grouped_metric_csv.py`: render an externally prepared grouped metric CSV.
- `plot_homogeneous_checkpoint.py`: inspect and plot any homogeneous campaign
  checkpoint selected by campaign config.
- `plotting_common.py`: shared implementation for the two metric plotters.

## Temporal data

- `export_trajectory_npz_to_lammps_dump.py`: convert a repository trajectory NPZ
  to the LAMMPS dump format consumed by temporal datasets.
- `inspect_temporal_lammps_dataset.py`: inspect a dump and optionally build its
  persistent temporal cache.
- `run_initial_configuration_dynamics.py`: install the exact published EAM
  tables and prepare, run, launch, inspect, or stop a position-conditioned
  fixed-cell trajectory from an Al, Mg, or Ta `initial_configurations` snapshot.
- `run_zr_initial_dynamics.py`: prepare and manage the provisional reconstructed-
  MEAM Zr trajectory used for the initial 10 ps analysis.
- `benchmark_zr_lammps.py`: run a standalone, provenance-recorded Zr LAMMPS
  benchmark before committing resources to a long trajectory.
- `run_zr_al_mg_24ps_branches.py`: preflight and manage the six-branch, 24 ps
  NPT campaigns for Al, Mg, and provisional reconstructed-MEAM Zr.
- `convert_initial_branches_float16.py`: convert the completed 18-branch
  campaign into checksum-verified, memory-mappable float16 trajectories.
- `archive_large_lammps_artifacts.py`: atomically relocate verified raw LAMMPS
  text trajectories and restart files outside an upload directory.
- `write_initial_branch_readmes.py`: generate campaign and per-branch simulation,
  provenance, binary-format, and quantization documentation.
- `run_temporal_vamp.py` and `evaluate_temporal_vamp.py`: staged temporal VAMP
  experiment entry points; see `docs/temporal_vamp.md`.

Interactive MD cluster rendering is owned by its implementation module:

```bash
conda run -n pointnet python -m src.vis_tools.md_cluster_plot ANALYSIS_DIR
```

## Atomistic campaign

- `run_optimized_al_homogeneous_campaign.sh`: select and resume the maintained
  optimized Aluminum campaign from an existing selection report. The launcher
  resolves the repository root from its own location and requires an explicit
  `PYTHON` executable and `DEVICES` list.
