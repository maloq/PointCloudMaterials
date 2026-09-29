# Rich-MACE checkpoint: interface analysis

Frozen **MM-RD-MACE256-D3-C3-L3-Z256, optimizer step1088**: eight completed
training epochs plus56 of129 updates. This is the latest completed checkpoint
copied when requested, not a convergence claim or metric-selected checkpoint.
The original training run continues independently.

- [Held-out Al MD: thirteen frames](analyses/matched/index.html)
- [Relaxed Al static: six snapshots, 166–240 ps](analyses/static/index.html)
- [Checkpoint receipt](technical/checkpoint.json)
- [Input and inference receipt](technical/inference.json)

Both viewers contain neural and original descriptor 3D PaCMAP, full previous MD
populations, fixed optimal color matching, interactive atomic samples, optional
crystal-grid comparisons, and spatial/embedding-neighbor paths. All 256 frozen
scalar coordinates are used. K7 centroids fit only the original 74,880 training
observations. Neural weights and descriptor clusters remain fixed.

Joint-descriptor correspondence:

| Dataset | Population | Centers | Matched overlap | ARI |
| --- | --- | ---: | ---: | ---: |
| matched | All displayed centers | 24,960 | 37.7% | 0.266 |
| matched | Within 12 Å of interface | 10,682 | 41.8% | 0.267 |
| static | All displayed centers | 24,000 | 53.2% | 0.121 |
| static | Within 12 Å of interface | 6,044 | 64.4% | 0.345 |

The color assignment is fixed within each dataset: 24,960 held-out display
observations versus 684,723 full static-grid centers. Different populations and
mapping references prevent interpreting these numbers as a cross-dataset ranking.
MACE was trained on 442 rich local descriptors; correspondence is an evaluation
of retained descriptor structure, not independent unsupervised discovery or
crystallization prediction. Static inputs are relaxed; training inputs were raw.
See each bundle's `tables/METRICS.md` for definitions and implementation hashes.
