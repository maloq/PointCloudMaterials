# Interactive lattice overlays and embedding paths

These are frozen-model visualization diagnostics, not new encoder training,
cluster fitting, crystallization forecasts, or checkpoint-selection metrics.
Existing PaCMAP coordinates, cluster IDs and optimal color permutations are unchanged.

## Sample lattice overlay

`src/research/spatial_vicreg_bias/sample_lattice.py` applies OVITO PTM to each
saved 80-atom example, in its existing PCA display orientation. Disconnected
examples are separated by more than four maximum sample radii for batched PTM;
the central atom's first shell remains entirely within its own sample.
FCC, HCP and BCC are enabled; an unthresholded best candidate is recorded, with
identification accepted only at dimensionless PTM RMSD ≤ 0.1. A poor candidate
is explicitly labeled **no crystal identified**. None is reported when PTM
cannot provide a candidate.

Orientation and interatomic spacing define an undeformed ideal periodic lattice.
Only rigid orientation and isotropic scaling are used; affine strain is not
fitted away. HCP's two AB basis offsets under its orientation symmetry are
resolved against the first shell. Rotated perfect FCC/HCP/BCC calibration checks
must reconstruct the sample to numerical precision before real overlays export.
The ideal lattice extends to the farthest observed radius plus one nearest-neighbor
spacing. A one-to-one least-squares assignment matches 79 outer atoms to ideal
sites, while the focal atom is fixed at the ideal origin. Full-sample mismatch
is the RMS displacement of those 79 atoms in Å; it differs from local PTM RMSD.
The displayed deviation lines are these assignments, not measured trajectories.

The local-structure toggle shows a convex hull of the observed nearest 12
(FCC/HCP) or 14 (BCC) neighbors. The grid connects ideal sites within 1.05 times
the fitted nearest-neighbor spacing. The display includes only the 80 ideal sites
assigned to the sample atoms and edges between those sites, with smaller markers
and faint lines. The extended candidate lattice and all assignments remain
unchanged; fitting and mismatch calculations still use the original assets.
This is a candidate comparison, including
for disordered/interface samples; it cannot establish that an entire patch is
one crystal or that an individual deviation is a defect.

References: [OVITO PTM](https://www.ovito.org/manual/reference/pipelines/modifiers/polyhedral_template_matching.html),
[PTM reference templates](https://github.com/pmla/polyhedral-template-matching/blob/master/ptm_constants.h),
[Larsen et al.](https://arxiv.org/abs/1603.05143).

## Embedding travel

`embedding_travel.py` selects the same atoms for every checkpoint in a snapshot:
2,048 uniform draws plus up to 64 atoms from each existing joint descriptor
cluster, without replacement within each draw and deduplicated across draws.
Snapshot-derived deterministic seeds fix selection. Descriptor stratification
improves representation of small clusters but is not a population weight;
rare neural clusters can be absent from this displayed subset.

Frozen inputs: each selected center's actual saved nearest-80 geometry, original
fixed length normalization, no temperature, age, time covariates, history, motion,
or teacher. Original frozen training source and checkpoints are used. Encoder
and projector outputs are exported separately in float32, with no per-frame
normalization. Replayed K=7 assignments must exactly match saved assignments.
No feature cache is evicted, and no W&B training run is created.

Two graphs use the undirected union of directed 16-nearest-neighbor edges:

- **Spatial**: Euclidean coordinate distance; minimum-image orthorhombic periodic
  distance for held-out source 908, nonperiodic distance for static Al.
- **Embedding**: Euclidean distance on the entire raw exported vector, not PaCMAP,
  PCA, cluster numbers, or a selected embedding coordinate.

The controls choose endpoint clusters and a deterministic endpoint example.
The start samples the cluster's ordered atom list; the endpoint is chosen among
spatially distant members of the requested cluster (excluding the start).
Dijkstra minimizes summed edge distance in the selected graph. No synthetic
bridge is inserted if the graph is disconnected. Paths are through this sampled
graph, not full MD nearest-neighbor graphs or continuous interpolated structures.

For each visited atom, show its cluster, full vector, Euclidean distance from
the first vector, and Euclidean change from the preceding vector. Horizontal
position is cumulative spatial edge length in Å in both path modes. Path markers, the selected atom and the background use the same original MD
coordinates inside the displayed cell. Line segments crossing periodic boundaries
are omitted rather than drawn across the box; cumulative length still uses
minimum-image edges. Embedding neighbors may be physically distant. This display
change does not modify the graph, selected path, distances or vectors. Raw coordinate values form
the heatmap; there is no fitted projection or per-path standardization. Coordinate
indices are not rotation-invariant and should not be compared across independent
models as if they were matched features. Distances within a frozen model are
meaningful diagnostics; their raw scales can differ across models/projectors.

These are spatial/feature paths in one snapshot, not temporal trajectories,
transition pathways, committors or evidence of a precursor. Graph smoothing,
subsampling and endpoint choice affect their appearance.

Provenance lives in `technical/rendering/lattice-*.json` and `travel-lane*.json`.
`lattice-data/` and `travel-data/` hold lazy, per-snapshot/per-model display assets.

## Rendering optimization

The browser implementation now uses a worker and a binary heap for the same
weighted shortest-path problem. Equal-distance heap entries use node index as
a tie-breaker, preserving the previous linear-search order. Edge weights and
raw coordinate profiles are unchanged. Path-cursor movement changes only marker
and guide positions. Lossless compact display assets restore the original
float32 values as ordinary JavaScript numbers; no quantization or rescaling is
introduced. Off-screen rendering and bounded browser caches affect when assets
are drawn, not the sample selection or scientific calculations.

Publication stages now read schema-tagged JSON sidecars for pages and lazy assets.
The browser JS and compact binary encodings remain display outputs; later Python
stages do not parse them to recover scientific inputs. The shared payload/template
writer preserves sample IDs, vectors and graph data. Comparison highlighting
does not change the path graph, endpoint population or correspondence counts.
