# Rich-descriptor / neural cluster correspondence around interfaces

Independent clusters of TDA, bond order, CNA and the family-balanced joint vector are fitted on training sources. Neural cluster assignments are the frozen original global clusterings. Descriptor fitting uses both all phases and the declared interface ±12 Å population. No neural encoder or predictor is trained.

![Interface correspondence trajectories](plots/interface-correspondence-trajectories.png)

[Metrics](tables/METRICS.md) · [Numerical results](tables/cluster-correspondence.csv)

Plots are grouped into correspondence matrices, rich feature signatures and sparse matched spatial projections. Cluster numbers/colors from independent fits have no shared meaning; use the correspondence matrices. Spatial projections contain 64 sampled centers per snapshot and must not be interpreted as dense interface maps. Classical descriptors are reference partitions, not unique physical ground truth.
