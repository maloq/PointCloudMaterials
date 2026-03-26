# Temporal Synthetic Generator

This package is a clean temporal benchmark generator built around a latent-first workflow:

1. Sample a hidden site-state process on a fixed site scaffold.
2. Evolve smooth nuisance variables such as orientation, strain, thermal jitter, and defect amplitude.
3. Render each frame into atomistic coordinates.
4. Save tracked neighborhood trajectories plus ground-truth labels and transition metadata.

The renderer uses a single dense persistent-box path: one space-filling atom cloud is generated for the whole box first, then each frame applies bounded local displacements toward liquid, precursor, interface, crystal, and grain-boundary targets. That keeps realistic atom density everywhere, prevents atoms from teleporting or disappearing, and still exposes interpretable latent phase trajectories. A low-probability hidden crystal variant is tracked inside the `C` phase without introducing an extra top-level phase label.

## Main entrypoints

- `python -m src.data_utils.synthetic.temporal --config configs/data/temporal_synth_v1.yaml`
- `from src.data_utils.synthetic.temporal import generate_temporal_dataset`
- `python scripts/visualize_temporal_synthetic.py --dataset-dir output/synthetic_data/temporal_atomistic_ssl_v1`

## Output layout

```text
output_dir/
  config_snapshot.yaml
  manifest.json
  transition_graph.json
  site_layout.npz
  validation_summary.json
  frames_chunk.npz
  frame_metadata.json
  visualizations/
    state_occupancy_over_time.png
    site_state_raster.png
    transition_matrix.png
    frame_snapshots.png
    local_trajectory_gallery.png
    site_state_evolution.html
    all_phases_diagonal_cut.gif
    solid_only_full_box.gif
  latent/
    site_latent_trajectories.npz
    grain_orientations.json
    transition_events.csv
  neighborhoods/
    trajectory_pack.npz
    manifest.json
```

The recommended fast path stores all rendered frames in one `frames_chunk.npz` archive plus a single `frame_metadata.json` sidecar. The loader and visualization tools also still understand the older `frames/frame_*/` layout when needed.

The visualization bundle is meant for quick quality assessment:

- state occupancy trends over time,
- site-state raster plots,
- transition-count heatmaps,
- selected global frame slices,
- tracked local-neighborhood galleries,
- interactive 3D site-state evolution,
- all-phase diagonal-cut 3D box GIF animation,
- full-box GIF animation with liquid and precursor hidden.

## Scope

The current implementation is a strong stage-1 / stage-2 benchmark:

- independent semi-Markov site trajectories are supported,
- a coupled mode with nucleation seeds and neighbor-biased forward progression is supported,
- grain-bearing states can share/inherit grain identities and orientations,
- tracked local neighborhoods are exported directly for temporal SSL and JEPA-style experiments.

It is not yet a full atomistically consistent polycrystal simulator. The architecture is designed so richer coupled rendering and boundary physics can be added without redesigning the latent dynamics or output format.
