# Real-MD qualitative analysis

- Selected clustering k: `7`
- Frames analysed: `6`
- Frame order: 166 ps, 170 ps, 174 ps, 175 ps, 177 ps, 240 ps
- Cluster groups: cluster_0=[0], cluster_1=[1]

## Cluster proportions
- CSV: `frame_cluster_proportions.csv`
- Stacked area: `cluster_proportions_stacked_area.png`
- Paper SVGs: `cluster_proportions_stacked_area_paper.svg`

## Representatives
- Root: `representatives`
- Shared-style figure: `04_cluster_representatives_k7_pca_reciprocal.png`
- Edge-connected figure: `09_cluster_representatives_knn_edges_k7.png`
- Structure analysis JSON: `10_cluster_representatives_structure_analysis_k7.json`
- Structure analysis CSV: `10_cluster_representatives_structure_analysis_k7.csv`

## Latent projection
- Method: `pca`
- CSV: `latent_projection.csv`

## Transitions
- Match mode: `nearest_neighbor`
- Match tolerance: `1.9000`
- Aggregate flow diagram: `transition_aggregate_flow.png`

## Flicker metrics
- Enabled: `False`
- Reason: requires instance_ids to track local structures across frames
