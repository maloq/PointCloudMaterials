# Liquid-cluster diagnosis — 2026-09-15

The user's concern is supported by matched spatial measurements. Concatenating
the rapidly varying center feature into the clustering representation largely
destroyed the inner block's spatial coherence. The stable architecture is useful,
but the exported representation and training/selection objective do not yet meet
the intended goal of persistent, physically meaningful structural regimes.

This was a gap in our evaluation: boundary continuity and physical readout error
were treated as sufficient progress before checking liquid-only spatial clusters.
The full 512-dimensional vector should not have been described as retaining all
of the smooth-inner representation's useful stability properties.

## What the controlled ablations show

All rows below use identical centers in the 166 ps Al frame, restricted to the
116,248 centers marked Other by full-frame PTM. This frame is 98.81% PTM Other.
Spatial agreement uses directed edges to the six nearest sampled centers, keeping
only edges whose endpoints are both Other. Adjusted agreement is
`(observed-chance)/(1-chance)`; zero means the agreement expected from cluster
occupancies and one means perfect agreement. PTM Other is a disordered-region
mask, not a definitive liquid phase classification.

Collective-order explained variation is descriptive eta-squared for qbar6,
computed from cluster means on 6,000 selected centers, then restricted to Other.
qbar6 averages the complex bond-order vectors of the atom and its 12 neighbors
before taking their norm; it detects collective angular organization. It is not
an average of scalar q6 values. Higher separation in this observable alone does
not prove that a partition identifies distinct phases.

| Representation / partition | Spatial agreement above chance | qbar6 variation explained | 0.75 ps normalized feature change |
|---|---:|---:|---:|
| Current joint 512D, saved full-data clustering | 0.0039 | 4.8% | 0.1804 |
| Same checkpoint, raw inner 256D | 0.2922 | 11.5% | 0.0263 |
| Same checkpoint, raw center 256D | 0.0004 | 4.8% | 0.2954 |
| Same checkpoint, existing inner projector 128D | 0.3803 | 5.0% | 0.1021 |
| Archived GeoFrame V2, VICReg-best projector | 0.1003 | 26.6% | Not a matched temporal comparison |

The temporal column uses the earlier six-source, tracked-atom cohort, not these
static frames. Mean squared change is divided by mean training-coordinate
variance separately for each representation. Thus the dual result 0.1804 differs
from the earlier 0.1608 result that first gave the two blocks equal variance.
Both calculations show the same loss of temporal stability after including center
features. These quantities are not temporal cluster-switch probabilities.

In the current partition, actual neighbor label agreement is 20.81%, versus
20.50% expected from occupancy: effectively chance. GeoFrame's corresponding
values are 45.38% versus 39.29%. Inner-only gives 47.01% versus 25.14%.
The pattern persists across all five early/intermediate snapshots: current
adjusted agreement is 0.004–0.046, inner-only 0.288–0.326, and GeoFrame VICReg
0.100–0.199. The late mostly crystalline frame is not used to establish liquid
coherence; its Other population includes defects and interfaces.

![Matched spatial slice](../../output/mace_context_clusters/diagnosis-20260915/plots/spatial-slice-comparison.png)

Identical centers in a 5 A slice at 166 ps. Colors denote each model's own labels
and are not matched between models. The numeric agreement above uses the full 3D
neighbor graph, not visual assessment of this slice.

## Collective information is present but poorly used by clustering

A frozen-feature linear readout provides a useful distinction between missing
information and a poor clustering distance. Train, validation and test centers
occupy separate x slabs, with 40 A gaps exceeding twice the 17 A embedding
support. All splits use PTM Other. Feature/target scaling uses training rows;
each target selects ridge regularization on validation rows. Fits use float64
and an SVD solver. The test scores below are for 1,604 centers at 166 ps.

| Frozen representation | Collective qbar6 test R² | Neighbor orientation coherence test R² | Single-atom q6 test R² |
|---|---:|---:|---:|
| Combined inner + center | 0.724 | 0.746 | 0.800 |
| Inner only | 0.467 | 0.390 | 0.125 |
| Center only | 0.699 | 0.733 | 0.803 |
| Existing inner projector | 0.378 | 0.313 | 0.072 |

R² is one minus prediction MSE divided by the test observable's variance. Zero
matches a constant prediction at the test mean; one is exact prediction. These
are predictive readout scores, unlike the descriptive cluster explained variation
above. The combined representation carries collective information, but its
current K-means distance barely uses it. Inner features also retain appreciable
collective information, supporting a frozen-backbone structural mapping as the
first intervention. The center block is useful for prediction even though its
raw distances give poor state coherence.

This is spatial holdout within the same six frames and trajectory, not an
independent-source or temporal generalization result. Archived GeoFrame raw
features were not retained, so a matched linear probe is unavailable for it.
The initial float32/default-solver pilot is retained in `physical-probe/`; it
reported conditioning warnings. The quoted results use `physical-probe-float64/`.

## What went wrong

1. **The center feature is a poor state variable on its own.** Within the 166 ps
   disordered region, its standardized squared difference to a spatial neighbor
   is 0.972 times its difference to a random center. Inner-only is 0.447, and the
   inner projector is 0.411. The center block contributes about 60% of the
   combined standardized spatial increment energy in this frame, increasing to
   83% by 177 ps. Its normalized 0.75 ps change is 11.25 times the inner block's.
   This is rapid local variation, despite continuity under infinitesimal motion.

2. **The loss did not directly organize the representation that was clustered.**
   `joint_loss` passes only the first 256 channels to `loss_from_features`.
   `compute_spatiotemporal_loss` then applies the VICReg projector before its
   spatial and temporal losses. The center channels receive shared-backbone and
   physical-head gradients, but no direct matching loss on their exported
   distances. Historical GeoFrame analyses used their trained 128D projector.
   Here we exported the raw 512D concatenation. This is a substantive change of
   representation, not merely another encoder backbone.

3. **The physical targets favor a different scale of structure.** At 166 ps the
   current cluster partition explains 46.3% of local nearest-shell density
   variation and 47.5% of single-atom q6 variation, but only 4.8% of qbar6 and
   5.2% of neighboring bond-orientation coherence. GeoFrame VICReg explains 26.6%
   and 22.6% of the latter two. The colored liquid partitions are therefore not
   devoid of physical information: they largely organize instantaneous local
   packing, which has little coherence between sampled centers. The joint
   structural targets were q4, q6, nearest-shell density and mean distance;
   collective qbar6/bond coherence and persistent liquid motifs were absent.
   The frozen readouts above establish that collective information is nevertheless
   encoded; the missing objective did not ensure that clustering distances use it.
   This does not prove supervision caused the geometry: the selected backbone
   changed very little during the joint pilot.

4. **The pilot was not trained or selected as a discovery model.** It used 3,456
   training anchors, 13 updates per epoch and 12 epochs: 156 updates. Checkpoint
   selection minimized physical-head validation loss, which improved only 0.88%
   over epoch zero. Earlier paired readout tests found negligible changes in
   encoded information from these updates. Neither liquid spatial coherence nor
   state persistence/physical discrimination selected the checkpoint. In
   contrast, the archived GeoFrame temporal fine-tune used 4,320 updates with
   batch 8,192, on top of its existing structural pretraining. This comparison
   does not isolate architecture or batch size as a causal factor.

5. **Seven-means partitioning provides seven labels regardless of state evidence.**
   A connected liquid manifold can contain useful continuous order or transition
   pathways; disjoint UMAP islands are not required. But a forced partition is
   not proof of persistent states. Channel standardization can give fast channels
   substantial weight. Keeping only eight PCs does not solve this: dual adjusted
   agreement is 0.0029 at 166 ps. Switching dual features to Euclidean k-means
   gives only 0.0063. The dominant problem is the representation, not these
   clustering settings or the UMAP plot.

The existing projector is also not an automatic solution. It has the highest
spatial agreement but explains only 5.0% of qbar6 variation at 166 ps; its temporal
change is almost four times inner-only. Higher coherence can come from averaging
or discarding information. Likewise, GeoFrame's stronger collective-order
separation does not justify restoring its known canonical-frame discontinuities;
the archived continuity audit found large jumps at approximately 2e-6 A
geometric perturbations.

## Recommended experiments, in order

**First, use an explicit slow structural representation and a separate local
detail representation.** Keep complete message context, smooth radial cutoffs,
tracked centers, and smooth pooling. Use the inner block as the initial structural
baseline; retain center features for instantaneous TDA and local-property heads.
Do not concatenate the raw fast block back into the clustering distance. This
retains the local information that motivated the center branch without forcing
every local fluctuation to change the structural state.

**Train the representation that will actually be clustered.** A compact smooth
mapping from inner features (and, later, smoothly pooled multiscale descriptors)
can produce 32–128 structural channels. Apply the spatial/temporal and variance/
covariance losses directly to that exported output; a bypass through an unused
projector would recreate the mismatch. Start with the backbone frozen so the
effect is attributable to representation learning, then test limited fine-tuning.
Compare matched budgets and source splits; checkpoint selection must include
within-liquid physical and persistence criteria, not only combined TDA error.

**Add collective targets and preserve their distinctions.** Predict averaged bond
order qbar4/qbar6, neighboring orientation coherence, bond-order distributions,
ring/topology summaries and their persistence across several physical lags.
Smooth pooling at more than one radius can expose correlations that one weighted
mean hides. Radius changes require a correspondingly complete message halo and
fresh boundary checks. Keep instantaneous q4/q6, density and hard-neighborhood
TDA in the local-detail task; do not demand that the slow state be invariant to a
real structural transition. The completed frozen linear probe confirms that
collective targets are partly decodable from inner features already; test a
learned structural mapping before assuming a larger backbone is necessary.
Prefer topology targets with consistent support or smoothly defined density
fields: fitting the exact jumps of a hard nearest-80 target conflicts with a
continuous geometric representation.

**Discover soft states and retain continuous transition coordinates.** Compare a
continuous structural embedding with soft memberships. Candidate state counts
must be selected using source-held-out persistence, reproducibility, physical
separation and transition behavior. Time-lagged predictive/VAMP-style objectives
are candidates for learning persistent modes, not established fixes here. Handle
the nonstationary crystallization/quench protocol explicitly; do not interpret a
single stationary Markov model across the whole quench without testing it.
Spatial penalties should not force neighboring atoms across genuine interfaces
to share a state. Post-hoc label smoothing alone could make attractive maps while
concealing a poor representation.

**Require all four properties in acceptance tests.** Repeat controlled membership,
rotation and permutation checks; measure 0.75–12 ps tracked-atom increments and
state persistence; measure spatial agreement inside liquid/disordered regions;
and verify independent collective descriptors, target prediction and transition
behavior on fresh source trajectories. Check within-condition variance and
effective rank so a crystal/liquid axis or collapsed state cannot pass by itself.
Evaluate shuffled-time/spatial controls and seed stability. More clusters or a
higher silhouette is not an acceptance criterion on its own.

The immediate recommendation is to retain the stable inner architecture and
develop a directly trained structural output with collective supervision. Merely
changing the cluster count, lowering PCA dimension, or adding more of the same
joint training is not supported as a sufficient fix by this diagnostic.

## Evidence and limitations

All six frames and 684,723 centers were analyzed. The eight new cluster fits use
36,000 selected centers in total, whereas saved current/GeoFrame fits used their
full original grids. The new dual refit reproduces the same poor liquid coherence
(0.0032 versus 0.0039), reducing concern that fit subsampling explains the effect.
Physical explained variation is descriptive on the selected fit centers; it is
not cross-validated predictive accuracy. Overlapping inner regions mechanically
increase spatial coherence, which is why physical discrimination must also pass.
No encoder was trained in this diagnostic and no new method is claimed to have
resolved the combined objective. Zr requires its own follow-up acceptance tests.

- [Spatial agreement](../../output/mace_context_clusters/diagnosis-20260915/tables/spatial.csv)
- [Physical descriptor separation](../../output/mace_context_clusters/diagnosis-20260915/tables/physical.csv)
- [Continuous spatial variation](../../output/mace_context_clusters/diagnosis-20260915/tables/continuous.csv)
- [Temporal and boundary checks](../../output/mace_context_clusters/diagnosis-20260915/tables/temporal.csv)
- [Spatially held-out physical readouts](../../output/mace_context_clusters/diagnosis-20260915/physical-probe-float64/tables/probe.csv)
- [Frozen metric definitions](../../output/mace_context_clusters/diagnosis-20260915/tables/METRICS.md)
- [GeoFrame V2 study](/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/output/geoframe_v2_spatiotemporal_analysis_20260905/RESULTS.md)
- [GeoFrame continuity audit](/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/output/geoframe_continuity_20260905/RESULTS.md)

Method references: [VICReg](https://arxiv.org/abs/2105.04906) defines agreement and
anti-collapse regularization, not a physical state-discovery guarantee.
[Lechner–Dellago](https://arxiv.org/abs/0806.3345) motivates averaged bond-order
descriptors. [VAMPnets](https://doi.org/10.1038/s41467-017-02388-1) learns molecular
kinetic representations and soft states; applying that approach to these
crystallizing trajectories is a proposal requiring validation.
