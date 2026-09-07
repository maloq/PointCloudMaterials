# Why predictive density loses the older spatial structure map

The user's preferred GeoFrame model is a better reference for the desired static
structural field than the GeoFrame control used in the twelve-hour campaign.
Calling predictive density the strongest overall replacement was too broad:
it performed well on the selected future assays, but those assays did not
require preservation of the older representation's structural organization.

## Comparison and direct measurements

Reference: `output/detached/vicreg_geoframe_v2_factor_sn_grouped_scratch_20260831_160541/analysis_best_static`.
New analysis: `output/temporal_hypotheses_12h_20260906/static_pipeline_encoder_only`.
The older analysis exports a VICReg **projector**; the new standard analysis
exports the raw density **encoder**. Density's projected representation from
the preceding exploratory static run was included as a separate control.

Both full analyses contain 772,953 centers from the same six source snapshots,
but a direct audit found shifted grids. Their row numbers are not interchangeable.
The spatial comparison uses **684,723 exactly shared centers**, matched within
each snapshot with a maximum coordinate error below 1e-6 Angstrom and a one-to-one
index mapping. Neighbors are the six nearest shared centers in each frame.

| Quantity | Older GeoFrame projector | Predictive density encoder |
|---|---:|---:|
| Full-analysis cosine silhouette, k=7 | 0.7103 | 0.3737 |
| PCs retaining 99% of clustering-space variance | 8 | 33 |
| 166 ps: continuous neighbor difference / random-pair difference | 0.659 | 0.950 |
| 175 ps: continuous neighbor difference / random-pair difference | 0.328 | 0.481 |
| 166 ps: chance-adjusted neighbor-label agreement, PTM-Other | 0.067 | 0.004 |
| 175 ps: chance-adjusted neighbor-label agreement, PTM-Other | 0.093 | 0.010 |

Continuous differences are standardized squared feature differences, averaged
over 10,000 identical centers per frame and compared with random same-frame
neighbors. A ratio near one means that physical neighbors are barely closer
than random pairs in this metric. Cluster agreement is corrected for endpoint
cluster proportions: zero is chance, one is complete agreement. PTM-Other is
an imperfect geometric subset, not a ground-truth liquid definition. Even the
older model's liquid-region agreement is modest after this correction, but it
is consistently greater than density's. These are descriptive measurements,
not independent-source confidence intervals.

Two controls matter:

- Retaining only eight PCs for density increases its cosine silhouette to
  **0.4383**, but its PTM-Other adjusted neighbor agreement remains **0.005** at
  166 ps and **0.009** at 175 ps. Dimensionality reduction makes clusters look
  cleaner without restoring the desired spatial organization.
- Density's projected representation also remains weakly spatially coherent:
  adjusted agreement is **0.005/0.011** at 166/175 ps. Its continuous difference
  ratios are **0.969/0.795**. Removing the final representation map is therefore
  not the sole explanation.

The main crystalline domain is captured. At 240 ps whole-frame neighbor-label
agreement is high for density because most centers occupy one cluster. Restricting
the measurement to PTM-Other reveals what that bulk-crystal score hides.

## What changed and what we can actually conclude

**Data and selection changed.** The exact older training configuration includes
these six static Al snapshots, along with Mg, Ta, Zr and AlNi sources. It trained
with VICReg and the grouped FactorVAE regularizer, and selected its checkpoint
by validation clustering silhouette. The new campaign trained on Al/Mg/Ta MD
continuations and shooting trajectories, selecting by material-balanced motion
prediction. Its large data count does not guarantee adequate representation of
static interfaces, defects, or the distinctions visible in the older map.
The named static frames are training-domain diagnostics for the older model,
not an independent generalization test of it.

**Our objective did not express the entire goal.** The new predictive loss acts
on future EMA embeddings, with a variance/covariance penalty. The common motion
head is detached from the encoder except in the explicit motion ablation, and
its validation error selects all models. There is no term preserving the older
structural geometry, interfaces, defects or spatial field coherence. The campaign
also omitted a spatial-neighbor-view loss requested earlier in the project.
That is a scope mismatch, although it is not a proven cause of this comparison:
the exact older configuration itself has `vicreg_neighbor_view: false`.

**Smoothness was conflated with structural relevance.** Continuity with respect
to small coordinate changes says nothing about whether two different atoms in
a liquid have similar embeddings. A smooth function can encode instantaneous
cage fluctuations very accurately and still produce a noisy spatial field.
Predicting that field more accurately than persistence does not force it to
represent the slow structural distinctions of interest. The earlier temporal-
invariance control gives the opposite extreme: a useful low-dimensional slow
signal, but insufficient evidence of a rich predictive state.

**The representation class also changed.** Density uses fixed central density
power invariants followed by an MLP. GeoFrame uses learned local patches and
their interactions. The former is not an information-complete description of
arbitrary atomic environments; it may be less suited to distinctions involving
relationships between neighboring environments. This is a plausible limitation,
not a demonstrated architecture ceiling. Data, loss, selection, head and capacity
changed together. A static-distillation control is needed before blaming the
architecture alone. The older run's spectral normalization is on its FactorVAE
discriminator; it does not certify continuity of its canonicalizing encoder.

**Clustering exposes the problem; it does not solve it.** Seven-cluster k-means
must allocate seven groups. If one direction separates crystal and the remaining
variation describes a diffuse liquid continuum, it will partition that continuum
whether or not those subdivisions correspond to useful physical regimes. Replacing
those labels with arbitrary PCA or UMAP coordinates does not make the underlying
geometry physically meaningful. Coordinates themselves need the appropriate
training and validation criteria.

## A concrete route to the desired encoder

Preserve the useful old geometry while removing its discontinuities, then add
prediction. Do not replace all three pieces at once again.

1. **Establish the exact older model as a reference.** Evaluate this checkpoint
   on the same future benchmark and on dense temporal trajectories. Earlier
   GeoFrame controls were not this exact grouped-FactorVAE checkpoint. Preserve
   its continuous feature relationships and connected-regime directions; do not
   turn its seven cluster IDs into presumed physical ground truth.
2. **Train a smooth student on the static structural task first.** Compare
   multiscale density and equivariant message-passing students on the same static
   states, with fresh static sources reserved for validation. Distill robust
   pairwise similarities or local continuous coordinates from the older model.
   Estimate teacher consistency under small perturbations and rotations, and
   downweight unstable affinities so canonical-frame jumps are not copied into
   the student. Use independent order, strain, coherence and TDA assays to test
   which distinctions are retained. Teacher agreement is a preservation target,
   not a restriction that all new discoveries must agree with the teacher.
3. **Expose a structural coordinate head and retain richer local features.**
   A small structural head should describe ordering, intermediate environments,
   defects and other reproducible directions. Its dimension should be chosen
   by held-out structural and kinetic tests, not fixed to seven. A richer feature
   branch can retain fast local information needed for prediction. These may
   share a smooth multiscale backbone. Neighbor aggregation should retain
   orientation/coherence information where needed, not average unrelated scalar
   states indiscriminately.
4. **Add temporal learning without erasing structure.** Mix static replay and
   temporal examples. Preserve the structural geometry while learning lagged
   predictive coordinates, and use modest spatial regularization only between
   compatible environments. Uniform spatial smoothing would erase interfaces
   and defects. A lagged-correlation/VAMP-style objective is a relevant candidate
   for slow coordinates, with temperature and nonstationarity handled explicitly.
   Reversible equilibrium objectives should not simply be applied across mixed
   temperatures or a nonstationary crystallization trajectory.
5. **Predict distributions, with history when available.** Learn
   `P(structural coordinates at t+lag | current structural coordinates, local features, history, temperature)`.
   A position-only snapshot cannot determine every stochastic future. Report
   conditional-mean skill, uncertainty and multi-step behavior against persistence,
   shuffled futures and a learned condition-only baseline.

Learning continuous slow coordinates without discretizing into clusters is an
established approach, for example [state-free reversible VAMPnets](https://arxiv.org/abs/1902.03336).
For the broader kinetic-learning framework see
[VAMPnets](https://www.nature.com/articles/s41467-017-02388-1). These papers support
the method family, not a claim that it will automatically solve this metal problem.
The proposed structural/fast-feature separation and teacher-preserving training
schedule remain hypotheses to test here.

## Revised acceptance criteria

Evaluate at least three axes separately: **structural information**, **spatial
and temporal stability**, and **future prediction**. Selection should not be a
single motion-MSE or silhouette score. Require a candidate to preserve useful
structural distinctions and boundary/defect sensitivity before accepting a
prediction improvement. Check stability within comparable environments and
retain actual physical changes; maximum smoothness is not the objective.

The immediate useful experiment is a short, controlled static teacher/student
test, followed by a static-replay versus temporal-finetuning ablation. It will
distinguish an insufficient input/architecture from loss-induced destruction of
useful geometry. A further broad twelve-hour sweep would be premature.

## Reproduction and artifacts

[Experiment recipe](../experiments/temporal_hypotheses_12h_20260906/README.md#spatial-geometry-diagnosis),
[alignment and metric definitions](../output/temporal_hypotheses_12h_20260906/spatial_diagnosis/protocol.json),
[spatial label coherence](../output/temporal_hypotheses_12h_20260906/spatial_diagnosis/spatial_label_coherence.csv),
[continuous spatial variation](../output/temporal_hypotheses_12h_20260906/spatial_diagnosis/continuous_spatial_variation.csv),
[clustering-only control](../output/temporal_hypotheses_12h_20260906/spatial_diagnosis/clustering_metrics.json).

Only the PCA dimensionality control refits a clustering model. No encoder was
trained, selected or changed during this diagnosis.
