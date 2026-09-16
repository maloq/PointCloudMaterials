# Why are liquid clusters incoherent in the joint MACE representation?

The complete static analysis used the raw 512-dimensional concatenation of
smooth-inner and tracked-center features. The user observed a connected liquid
latent cloud divided into colors with little spatial organization. This diagnostic
compares that partition with archived GeoFrame V2 labels on exactly matched
centers and changes only the representation/preprocessing used for clustering.

The completed [analysis and recommendations](RESULTS.md) distinguish continuous
geometric stability, temporal persistence, spatial coherence and physical meaning.
None alone establishes the other three.

The [literature review](LITERATURE_REVIEW.md) compares closely related work on
liquid structure, temporal coordinates, metric learning and state discovery,
and revises the proposed experiments in light of published and prior repository
results. It contains proposals, not additional encoder-training results.

```bash
python -m src.research.mace_context.cluster_diagnosis --config configs/analysis/mace_context_clusters.json
python -m src.research.mace_context.cluster_probe --config configs/analysis/mace_context_clusters.json
```

The config selects all six static Al frames and archived original, VICReg-best,
and VISReg-best GeoFrame V2 analyses. Current and archived labels are compared
on the same 684,723 centers, using six nearest spatial neighbors per frame.
Independent full-frame PTM at RMSD 0.1 marks disordered endpoints; Other is not
assumed to be a ground-truth liquid label. Physical descriptors use the existing
`bond_order` producer on true source coordinates. Each new clustering fits 6,000
centers per frame and predicts every center. This is descriptive same-source
diagnosis, not independent validation or a new training experiment.

Eight ablations compare raw dual, inner, center, and existing inner-projector
features; eight-PC truncation; and Euclidean versus spherical k-means. The
separate retained six-source temporal cohort checks 0.75–12 ps increments and
controlled boundary responses. Static grids do not provide temporal atom identity.

The follow-up probe freezes the checkpoint and fits linear readouts for eight
physical observables. Train/validation/test occupy x < 85 A, 125 < x < 145 A,
and x > 185 A, respectively; each observable chooses ridge alpha on validation
only. Float64 SVD fits avoid the numerical conditioning warnings of the retained
initial probe. This separates spatial prediction from cluster geometry, with no
claim of independent-trajectory generalization.

Results are in `output/mace_context_clusters/diagnosis-20260915/`, backed by WORK.
The original checkpoint, static analyses and archived reference bytes are unchanged.
Metric definitions and implementation hashes accompany the exported CSVs.
