# When does an encoder lose interfacial and liquid structure?

The requested baseline is the **epoch-34 GeoFrameTransformerV2 recipe**, trained
from scratch, not the recent short MACE screen. Train through epoch index 34
(35 complete dataset passes), retaining every epoch. Keep the original 160-epoch
learning-rate schedule and stop after pass 35; compressing its ten-epoch warmup
into a twelve-epoch schedule would change the experiment. FactorVAE starts at
epoch 5 and ramps for five epochs. Measure encoder and once-applied VICReg
projector separately. The archived picture used the latter.

## What we are looking for

A useful representation resolves physically distinct environments while
preserving continuous variation within them. A high silhouette or attractive
UMAP is insufficient. Neither all-liquid collapse nor arbitrary fragmentation
into seven groups is success. The picture's colors are hypotheses, not labels.

Use two independent axes: **local motif** (FCC, HCP, BCC, icosahedral,
unclassified; continuous order retained) and **spatial context** (crystal
interior, planar defect, crystal–liquid boundary, noncrystalline surroundings).
HCP in an FCC Al grain can indicate a fault; HCP in Zr is not intrinsically a
defect. Do not call every PTM-unclassified atom liquid or every ordered liquid
patch a nucleus. Retain an uncertain category and report coverage.

For Ta and Zr distinguish crystal-compatible ordering from competing five-fold
ordering. A *structural precursor candidate* is a connected ordered region
outside established crystal. A *dynamical precursor* additionally requires
tracked persistence and subsequent growth into crystal. Static snapshots alone
cannot establish that latter claim. Original static potential identities are
unknown; material names do not establish comparability with another potential.

## Literature reviewed before implementation

- [Lechner & Dellago (2008)](https://doi.org/10.1063/1.2977970): neighbor-averaged
  bond-order vectors improve discrimination among crystal structures. Keep
  local and averaged order: averaging alone can hide a defective central atom.
- [Larsen, Schmidt & Schiøtz (2016)](https://arxiv.org/abs/1603.05143): polyhedral
  template matching provides crystal templates, mismatch and orientation.
  [OVITO's implementation](https://docs.ovito.org/reference/pipelines/modifiers/polyhedral_template_matching.html)
  exposes RMSD and requires explicit enabling of the icosahedral template.
  Assess RMSD-cutoff sensitivity, rather than treating its default as truth.
- [OVITO planar-fault analysis](https://docs.ovito.org/reference/pipelines/modifiers/identify_fcc_planar_faults.html)
  separates intrinsic faults, coherent twins and multilayer faults using PTM
  orientations and distances. This is the appropriate independent check of
  internal colored bands in Al; their colors do not identify these defects.
- [Hu & Tanaka (2022)](https://www.nature.com/articles/s41467-022-32241-z):
  crystal-compatible liquid preordering and competing icosahedral environments
  behave differently, including in Ta and Zr. Their quantitative thresholds
  depend on neighbor weighting and model. We do **not** transfer NiAl's Q6
  threshold to all materials, or mix unnormalized w6 with normalized hat-w6.
- [Becker et al. (2022), Ta/Al/Mg](https://www.nature.com/articles/s41598-022-06963-5):
  persistent-homology descriptors and mixture models resolve nucleus interiors,
  boundaries and different liquid environments; low five-fold symmetry is
  associated with nucleation in their simulations. An independently computed
  topology descriptor is useful validation, but its learned clusters are not
  universal phase ground truth.
- [Becker et al. (2022), Zr](https://arxiv.org/abs/2109.08126): topology-based
  unsupervised identification covers liquid, crystals and nucleation in Zr.
  This motivates retaining multiple liquid environments instead of a binary
  crystal/liquid target.

## Independent reference assay

Build full-snapshot PTM reference fields before sampling centers, with free
boundaries and a two-input-radius exclusion margin where cell metadata is
absent. Never invent a periodic box from coordinate extrema. Enable FCC/HCP/BCC
and ICO; retain best-template RMSD. Use cutoffs 0.08/0.10/0.12 as a declared
sensitivity panel. For Al, compute planar-fault labels on the complete snapshot.
Compute q4/q6, normalized w4/w6, averaged q6 and normalized q6 bond coherence
from actual neighbor coordinates. Declare neighbor counts explicitly (12 and
14 sensitivity); these are not Hu–Tanaka's Voronoi-weighted definition.

Separate crystal interior from boundary using the fraction of nearby atoms
matching crystalline templates, retaining this continuous field. Defect labels
and mixed-boundary labels may overlap: preserve both axes in exported arrays.
Within noncrystalline surroundings, report continuous order and ICO similarity;
fit a transparent order threshold on the assay's fitting spatial region only,
and call detections candidates, not verified nuclei. Plot candidates in real
space with their classical descriptors and proximity to existing crystal.

## Predeclared measurements

1. **Interface/defect resolution:** frozen linear probes, per-class AP, balanced
   accuracy and confusion matrices on spatially separated regions, with an
   exclusion gap exceeding two input radii. Report class counts and missing
   classes. These are reference-assay scores, not independent-material transfer:
   the baseline deliberately trains on the original static collection.
2. **Liquid nuance:** reconstruction of continuous local/averaged order and
   topology within noncrystalline centers, with density-only and shuffled
   baselines. Binary crystal separation cannot satisfy this test.
3. **Our boundary-aware coherence diagnostic:** compare full-embedding distances
   on nearby atom pairs that agree versus disagree on independent structural
   labels; report AUROC and same-label distance normalized by random-pair
   distance. Spatial labels are not inferred from embedding clusters. All-one
   or constant embeddings score chance, not perfect coherence. This is a
   proposed diagnostic, not an established physical order parameter.
4. **Continuity without collapse:** median/95th-percentile embedding change for
   fixed small displacements, divided by the population RMS pair distance,
   alongside effective rank and response to larger displacements. Deterministic
   grouping in evaluation; diagnose local-frame switching separately. Noise is
   a perturbation assay, not MD time or evidence of temporal persistence.
5. **Dynamics/prediction:** reuse the latest source-disjoint Al future-order
   assay where input conventions are compatible, reporting conditional gains
   beyond current order and distance to existing crystal. Relate changes along
   checkpoints descriptively; checkpoints are correlated, not independent
   replicates. Verified Ta dynamics can establish candidate persistence; absent
   accepted Zr trajectories, Zr future-fate validation remains unmeasured.

Use one fixed assay, spatial split, seed and preprocessing across epochs. Keep
raw high-dimensional metrics primary. Refit UMAP only for explicitly labeled
within-checkpoint exploration; independently fitted coordinates cannot measure
temporal drift. Report shuffled and collapsed controls. There is no justified
single composite score yet: resolution must improve without loss of rank,
liquid descriptor fidelity or continuity. The one-seed replication is an
intuition experiment, not a confidence interval over independent training runs.

## Reproduction and findings

Active recipe: `configs/geoframe_evolution/epoch34.yaml`. Implementation and
commands are indexed in `docs/geoframe_evolution.md`. Results belong under
`output/geoframe_evolution/epoch34-reproduction-20260923/`.

Initial audit: the retained reference checkpoint is epoch 34/global step 3965.
The random static train/validation split is transductive. The older continuity
audit found discrete triad-frame switches in this architecture, so smooth
projections alone cannot establish physical continuity. No new training result
is asserted here before its measured export exists.

Completed findings: [35-pass checkpoint review](../../output/geoframe_evolution/epoch34-review-20260923/RESULTS.md).
All 35 fresh checkpoints, initialization and the archived reference were evaluated.
The encoder's liquid-order readout improves from 12 to 35 passes in Al/Ta/Zr,
while the projector's declines; interface/fault separation improves. Conditional
forecasting gains are not established across held-out sources. Dense Ta/Zr
candidate regions and their truncation/distance fields are available in the
linked results; they remain structural hypotheses pending future-fate validation.

**Historical-comparison correction:** the archived `geoframe-v2-visreg-epoch159`
configuration has `model_type: visreg`, `vicreg_objective: visreg`, lambda 0.4
and 4096 projections. The epoch-34 reference has `model_type: vicreg` without
those VISReg settings. Both use grouped FactorVAE. Their visual difference alone
cannot establish deterioration with training time. This fresh fixed-objective
trajectory avoids that confound.

**Reference-label limit:** context 2 is a *mixed crystalline-neighborhood proxy*.
Its neighbor-fraction rule can also include isolated crystalline motifs or grain
boundaries. It does not identify a unique solid–liquid dividing surface. The Al
planar-fault labels are a separate, more specific reference. Region and color
interpretations must retain that distinction; a high boundary-proxy AP is not
proof that every interfacial or grain-boundary subtype is resolved.
