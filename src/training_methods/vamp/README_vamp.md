# VAMP Extension

This folder adds a frozen-encoder VAMP baseline for local atomic dynamics on tracked LAMMPS trajectories.

## What it reuses from the repo

- The pretrained VICReg/VN checkpoint is loaded through the `VICRegModule` restore path.
- Local neighborhoods are built with `TemporalLAMMPSDumpDataset`, so the structural preprocessing stays consistent with the repo:
  - atom-centered neighborhoods
  - nearest-neighbor selection with the existing `closest` / `radius_then_closest` logic
  - centering on the tracked atom
  - normalization by the resolved cutoff radius `rc`
  - stable atom tracking through sorted LAMMPS atom ids

## Input assumptions

- The trajectory is a LAMMPS dump with a constant atom count across frames.
- The dump must contain an `id` column and either `x y z` or `xu yu zu`.
- Atom ids must be preserved across frames. If they change, embedding/pair construction stops with an explicit error.
- The checkpoint config must define either `data.radius`, `data.auto_cutoff`, or you must pass `--radius`.

## Environment

The repo instructions prefer the `pointnet` conda environment when available.

```bash
python -V
```

## Config-first usage

```bash
python -m src.training_methods.vamp.run_pipeline dump_first_10_512atoms_500frames
```

All VAMP entrypoints now take only a config name inside `configs/vamp/` or a YAML path:

```bash
python -m src.training_methods.vamp.embed_trajectory dump_first_10_512atoms_500frames
python -m src.training_methods.vamp.fit_vamp dump_first_10_512atoms_500frames
python -m src.training_methods.vamp.analyze_vamp dump_first_10_512atoms_500frames
python -m src.training_methods.vamp.verify_against_deeptime dump_first_10_512atoms_500frames
```

Stage selection for the full pipeline is controlled in the config via:

```yaml
pipeline:
  run_embed: true
  run_fit: true
  run_analyze: true
  run_verify: false
```

## Output layout

The VAMP configs now write into a single output root with the same broad idea as
`src/analysis`: readable figures live separately from saved machine-readable artifacts.

Typical layout:

- `<output_root>/embeddings/artifacts/trajectory_embeddings.npz`
- `<output_root>/embeddings/artifacts/trajectory_embeddings.npz.meta.json`
- `<output_root>/fit/figures/*.png`
- `<output_root>/fit/models/*.npz`
- `<output_root>/fit/artifacts/*.json`
- `<output_root>/fit/artifacts/*.csv`
- `<output_root>/fit/artifacts/*.npz`
- `<output_root>/analysis/figures/*.png`
- `<output_root>/analysis/representatives/*.png`
- `<output_root>/analysis/time_series/*.png`
- `<output_root>/analysis/temporal/*.gif`
- `<output_root>/analysis/md_space/*.png`
- `<output_root>/analysis/md_space/*.html`
- `<output_root>/analysis/md_space/raytrace/*.png` when enabled
- `<output_root>/analysis/artifacts/*.json`
- `<output_root>/analysis/artifacts/*.csv`
- `<output_root>/analysis/artifacts/*.npz`
- `<output_root>/verification/artifacts/*.json`

This keeps the top-level folders focused on the most important visual outputs while
moving `.json`, `.csv`, and `.npz` files into dedicated `artifacts/` directories.

## Fit outputs

Main figure outputs:

- `fit/figures/score_diagnostics.png`
- `fit/figures/singular_values_vs_lag.png`
- `fit/figures/implied_timescales_vs_lag.png`
- `fit/figures/ck_test.png` when available

Main saved artifacts:

- `fit/models/lag_<tau>_model.npz`
- `fit/models/selected_model.npz`
- `fit/artifacts/selected_projections.npz`
- `fit/artifacts/lag_diagnostics.csv`
- `fit/artifacts/lag_diagnostics.json`
- `fit/artifacts/deeptime_verification.json` when enabled
- `fit/artifacts/summary.json`

The manual estimator uses:

- separate means for `t` and `t+tau`
- `C00`, `C01`, `C11`
- `K_bar = C00^{-1/2} C01 C11^{-1/2}`
- SVD of `K_bar`
- VAMP-2 as the main fitting score
- VAMP-E on validation data

The implementation does not assume reversibility.

## Analysis outputs

Main figure outputs:

- `analysis/figures/cluster_populations.png`
- `analysis/figures/vamp_mode_dynamics.png`
- `analysis/figures/vamp_phase_space_clusters.png`
- `analysis/figures/vamp_phase_space_time.png`
- `analysis/figures/vamp_cluster_centers.png`
- `analysis/figures/spatial_cluster_snapshots.png`
- `analysis/figures/representative_neighborhoods.png`
- `analysis/figures/vamp_tsne_clusters.png`
- `analysis/figures/vamp_tsne_time.png`
- `analysis/figures/vamp_umap_clusters.png`
- `analysis/figures/vamp_umap_time.png`
- optional `analysis/figures/structural_cluster_populations.png`

Representative outputs:

- `analysis/representatives/04_cluster_representatives_k*.png`
- `analysis/representatives/08_cluster_representatives_spatial_neighbors_paper_k*.png`
- `analysis/representatives/09_cluster_representatives_knn_edges_k*.png`

Saved analysis artifacts:

- `analysis/artifacts/vamp_coordinates.npz`
- `analysis/artifacts/cluster_populations.csv`
- `analysis/artifacts/representative_neighborhoods.npz`
- `analysis/artifacts/summary.json`

Temporal and MD-space outputs:

- `analysis/time_series/cluster_proportions_stacked_area.png`
- optional `analysis/time_series/cluster_proportions_stacked_area_paper.svg`
- `analysis/temporal/md_space_clusters_diagonal_cut_k*.gif`
- `analysis/temporal/vamp_space_clusters.gif`
- `analysis/temporal/vamp_space_trajectories.gif`
- `analysis/md_space/md_space_clusters_k*.png`
- `analysis/md_space/md_space_clusters_k*.html`
- optional `analysis/md_space/raytrace/frame_*_clusters_raytrace.png`

The extra VAMP-specific plots are intended to make the dynamics easier to read:

- `vamp_mode_dynamics.png` shows how the leading singular functions evolve over time, with the frame mean and a 10-90% atom band.
- `vamp_phase_space_clusters.png` shows the leading VAMP coordinates directly, colored by cluster, with KMeans centers overlaid.
- `vamp_phase_space_time.png` shows the same leading VAMP coordinates colored by simulation time.
- `vamp_cluster_centers.png` shows the KMeans centers as a heatmap over the leading VAMP modes.

## Notes

- `window: early`, `window: middle`, and `window: late` split the selected trajectory into thirds before fitting.
- `fit_vamp.py` uses contiguous frame blocks for train/validation/test. Pair construction never matches different atoms across time.
- The repo includes a runnable large-dump smoke config at `configs/vamp/dump_first_10_smoke.yaml` for `datasets/dump_first_10.pos`.
- Phase-2 end-to-end VAMPnet fine-tuning is not implemented here yet. The frozen-encoder + linear VAMP baseline is the primary reference path in this extension.
