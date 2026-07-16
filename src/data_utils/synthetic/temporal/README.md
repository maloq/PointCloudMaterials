# Temporal Synthetic Generator

This package is a clean temporal benchmark generator built around a latent-first workflow:

1. Sample a hidden site-state process on a fixed site scaffold.
2. Evolve smooth nuisance variables such as orientation, strain, thermal jitter, and defect amplitude.
3. Render each frame into atomistic coordinates.
4. Save tracked neighborhood trajectories plus ground-truth labels and transition metadata.

The renderer uses a single dense persistent-box path. It must start from an `(N, 3)`
`atoms.npy` snapshot produced by a repository-owned force-driven simulation; it does not
invent or rescale a liquid packing. The config also names the source generator config. Loading
verifies the current schema-4 manifest and exact generator/calculator provenance, bulk-liquid
phase labels, metadata, atom count, final trajectory cell/volume/positions, and hashes of all
selected source files. Legacy phase-context outputs are rejected and must be regenerated. Each
frame then applies bounded local displacements
toward liquid, precursor, interface, crystal, and grain-boundary targets. Coordinates,
neighbor queries, contact relaxation, and centered neighborhoods all use the same periodic
cubic box and minimum-image convention. Every rendered frame must retain the configured
minimum periodic pair distance.

The dense latent phase field tiles the complete `[0, box_size)` domain with an
endpoint-excluded periodic grid; optional jitter is wrapped. Domain padding applies only to
the sparse tracked neighborhood centers, never to the periodic phase field. This prevents
oversized phase cells at faces and corners.

This construction preserves atom identity and exposes interpretable ground-truth latent
trajectories. It does **not** make the procedural displacements into molecular dynamics.
A low-probability hidden crystal variant is tracked inside the `C` phase without introducing
an extra top-level phase label.

Every configured structural-audit frame is independently reclassified from its complete
rendered coordinates with OVITO Polyhedral Template Matching (PTM). FCC, HCP, and BCC are
counted as crystalline and pooled by the frame's true **per-atom** `state_ids`; central-site
labels are never used for this check. The audit requires an explicit aggregate atom floor for
every named state and requires the `C` crystalline fraction to exceed the `L` fraction by the
configured margin. Because `C` declares FCC, it also records each PTM structure type by true
state and applies a separate FCC floor; HCP/BCC matches cannot stand in for the intended
crystal. Its full per-frame and per-state record is copied into both
`validation_summary.json` and `manifest.json`. PTM remains a diagnostic observable, not a
replacement source of labels.

The committed qualification gates require at least `50%` total PTM FCC/HCP/BCC and `50%`
PTM FCC in `C`, at most `10%` FCC/HCP/BCC in `L`, and an explicit `40`-percentage-point
`C-L` separation. With at least 4096 pooled atoms per state, these are conservative,
fail-closed semantic checks rather than bulk-crystal quality claims. A development probe of
the current procedural recipe at frames
`[0, 6, 12, 18, 24, 47]` found only `0.0313%` PTM FCC/HCP/BCC among `C` atoms versus `0.0031%`
among `L` atoms. The current production config is therefore structurally **blocked** by this
audit. Do not weaken the threshold to make it pass: the coherent crystal rendering itself
must be redesigned and requalified. This result reinforces that the package is a procedural
ML benchmark, not MD crystallization or evidence of physically realized crystal growth.

## Main entrypoints

- Generate the atomistic phase-context producer first so its validated bulk-liquid
  `atoms.npy` exists.
- `python -m src.data_utils.synthetic.temporal --config configs/simulation/temporal/realistic_metal.yaml`
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

`neighborhoods/trajectory_pack.npz` distinguishes central-site context from pointwise
truth. `state_ids` and `grain_ids` describe the tracked central site; mixed neighborhoods
must use `point_state_ids` and `point_grain_ids`. `same_state_fraction` records how much of
each neighborhood agrees with its central context. `local_atom_indices` in the frame archive
maps every neighborhood point back to the corresponding rendered atom row.

The `structural_audit` record reports the exact selected frames, verified producer chemical
symbol, periodic cell, PTM cutoff, raw structure counts, per-state atom support, per-state
crystalline fractions, and the crystal-minus-liquid pass/fail margin. Generation raises an
actionable error before writing a qualifying manifest if any configured state lacks enough
atoms or if the structural ordering fails.

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

It is not MD, a crystallization-kinetics model, or a full atomistically consistent
polycrystal simulator. Use it only as a controlled ML benchmark with known procedural
labels. Physical claims must come from the force-driven atomistic workflow instead.
