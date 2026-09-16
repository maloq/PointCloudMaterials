# Literature review: stable, coherent and meaningful atomistic representations

Research date: 2026-09-15. This is a targeted literature review prompted by the
[MACE liquid-cluster diagnosis](RESULTS.md), not a new training result or an
exhaustive systematic review. Sources are original papers, author manuscripts,
and author-maintained software documentation. Publication years below refer to
journal publication when verified; preprints are identified explicitly.

**Assessment.** Several parts of our problem have strong precedents: suppressing
irrelevant fast variation, discovering persistent coordinates, recovering
collective structure through spatial averaging, identifying statistically
supported motifs, and tracking neighbor exchange separately from rearrangement.
I did not find an established method demonstrated to jointly solve these for
our particular MACE checkpoint and pure Al/Zr trajectories. The evidence favors
testing representation selection and physical scale before a larger encoder.
Applications to water, model glass formers, proteins, and metallic surfaces must
be distinguished from validation on our materials.

**The closest papers and what they establish**

| Paper | Published evidence | Proposed use here; limitation |
|---|---|---|
| Lionello et al., JCTC 2025, [high-dimensional information and noise](https://doi.org/10.1021/acs.jctc.5c00374); [author manuscript](https://arxiv.org/html/2412.09412v2) | In water/ice coexistence, adding SOAP dimensions could hide an interface environment resolved by a lower-dimensional time-series analysis. A component carrying less than 0.001% of total variance was informative. | A particularly close analogue of our inner–center concatenation failure. Select dimensions by useful physical/temporal information; neither more channels nor a PCA variance threshold guarantees better states. Their demonstrations do not prove all high-dimensional representations are inferior. |
| Qiu et al., PNAS 2025, [short fluctuations and structural relaxation](https://doi.org/10.1073/pnas.2427246122); [indexed article text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12012455/) | Time-lagged canonical correlation analysis (TCCA) and a time-lagged autoencoder learn order parameters in the Kob–Andersen glass former. Short-lag training yields correlations with much later bond-breaking propensity; medium-range descriptors contribute substantially. | First compare a regularized linear temporal transformation of frozen inner and combined features. Their equilibrium binary-liquid result supplies a hypothesis, not a transferable lag or a guarantee for crystallizing monatomic metals. |
| Donkor et al., JPCL 2024, [collective structure near water's critical point](https://doi.org/10.1021/acs.jpclett.4c00383); [full manuscript](https://arxiv.org/html/2401.16245v2) | Local SOAP descriptors did not resolve the two macroscopic density regimes. Averaging descriptors across larger neighborhoods exposed collective domains. | Test several smooth pooling radii and structural correlations. A complete message halo ensures valid computation but does not establish that the pooled output preserves the relevant collective scale. Water's density regimes and numerical radii are not assumptions about Al/Zr. |
| Becchi, Fantolino and Pavan, PNAS 2024, [Onion clustering](https://doi.org/10.1073/pnas.2403771121); [full manuscript](https://arxiv.org/html/2402.07786v2) | Iteratively identifies recurrent time-series environments at a chosen minimum duration. The number of environments is an output; unresolved windows remain unclassified. Demonstrations include water freezing, metal-surface dynamics and experimental trajectories. | Test whether apparent liquid subpopulations have resolvable lifetimes. Its Gaussian local-noise model and duration rule are assumptions. It is an offline trajectory analysis; persistence imposed by the rule cannot validate raw encoder stability. |
| Caruso et al., PNAS Nexus 2025, [LEAP](https://doi.org/10.1093/pnasnexus/pgaf038); [full manuscript](https://arxiv.org/html/2409.18844v2) | Jointly analyzes neighbor-identity changes (LENS) and structural-descriptor changes (TimeSOAP), including their spatial and temporal ordering. Metal-surface examples distinguish exchange from geometric rearrangement. | Build an event audit alongside our structural states: identity turnover, geometry change, and both. Actual exchange is physical information even when an arbitrary patch-boundary substitution should be suppressed. These are event descriptors, not automatically a static structural embedding. |
| Wild et al., Nature Communications 2025, [Differentiable Information Imbalance](https://www.nature.com/articles/s41467-024-55449-7) | Learns feature weights by preserving neighborhoods of a specified reference metric; supports sparse feature selection. Demonstrated on molecular collective variables and force-field input selection. | A targeted response to our good physical readouts but poor clustering distances. Learn a metric against a declared collective-structure reference. Choosing that reference is supervision and can introduce bias; it does not independently establish new phases. |

**Additional evidence that changes the design**

1. **Spatial averaging has a strong, inexpensive baseline.** Boattini,
   Smallenburg and Filion, PRL 2021, found that local descriptors plus averages
   over neighboring environments predicted dynamic propensity comparably to a
   much more complex GNN in the model liquids they studied. This supports a
   controlled multiscale readout experiment before changing MACE depth. It is a
   supervised dynamics result, not unsupervised state discovery.
   [Paper](https://arxiv.org/abs/2105.05921).

2. **Average angular vectors before making them invariant.** Lechner and Dellago,
   JCP 2008, improved crystal discrimination by averaging complex bond-order
   vectors across a particle and its neighbors. The resulting invariant is
   different from averaging scalar bond-order magnitudes. For our study, explicit
   angular correlations or pooled equivariant quantities could preserve
   collective orientation that scalar means obscure. This is a proposal to test;
   the paper establishes crystal discrimination, not arbitrary liquid motifs.
   [Paper](https://arxiv.org/abs/0806.3345).

3. **Liquid heterogeneity can be meaningful without crystalline categories.**
   Boattini et al., Nature Communications 2020, combined bond-order descriptors,
   an autoencoder and Gaussian mixtures to obtain a structural order parameter
   correlated with dynamic heterogeneity in three model glass formers. Its
   association with slow particles weakened away from the glass transition.
   This motivates a classical structural baseline and temperature-stratified
   assessment; it gives no reason to demand the same number of populations at
   every temperature. [Paper](https://www.nature.com/articles/s41467-020-19286-8).

4. **Define communities through physical distributions.** Paret, Jack and
   Coslovich, JCP 2020, infer structural communities using information in two-
   and three-body correlations and compare them with mobility and Voronoi
   structure. This supplies a physically interpretable alternative reference
   to Euclidean distance in arbitrary encoder channels. Communities of similar
   environments need not be contiguous spatial domains.
   [Paper](https://arxiv.org/abs/2002.02726).

5. **Learn kinetics in the distance itself.** Noé and Clementi, JCTC 2015,
   construct kinetic maps by scaling dynamical coordinates so Euclidean
   distances reflect kinetic separation. This gives a principled alternative
   to equally weighting raw channels. The equilibrium kinetic-distance
   interpretation depends on the underlying process and estimated operator;
   a projection of a quench should not automatically receive that interpretation.
   [Paper](https://pubs.acs.org/doi/10.1021/acs.jctc.5b00553).

6. **VAMP is broader than equilibrium TICA.** Wu and Noé's VAMP framework
   optimizes singular components and offers cross-validation scores for general
   Markov processes, including nonreversible and nonstationary settings.
   Mardt et al.'s VAMPnets learn nonlinear features with fuzzy state memberships.
   These support temporal objectives and soft states. Neither supplies missing
   conditioning variables or guarantees that our local position-only process
   is Markovian. Distinguish changing sampling populations from a changing
   externally controlled dynamical protocol.
   [VAMP](https://arxiv.org/abs/1707.04659),
   [VAMPnets, 2018](https://doi.org/10.1038/s41467-017-02388-1).

7. **A future reconstruction loss is not a guarantee of slow modes.** Chen,
   Sidky and Ferguson, 2019, show theoretically and numerically that nonlinear
   time-lagged autoencoders can mix slow and high-variance directions, and can
   miss the slowest mode. This tempers the attractive Qiu result: a temporal
   autoencoder is an ablation, while a regularized TCCA/VAMP baseline helps
   identify whether its gain comes from temporal predictability.
   [Paper](https://arxiv.org/abs/1906.00325).

8. **Spatial smoothness depends on the physical timescale.** Jiang et al.,
   JCP 2023, use graph-based measures of dynamical smoothness and a
   geometry-enhanced GNN for glassy dynamics. Their manuscript distinguishes
   rough short-time motion from smoother long-time dynamics. This supports
   treating local detail and persistent collective structure separately, and
   measuring spatial variation at several lags. Uniformly forcing all adjacent
   atoms to have identical embeddings would be an unjustified extrapolation.
   [Paper](https://arxiv.org/abs/2211.12832).

9. **Estimate supported density peaks and their connections.** Density Peaks
   Advanced (DPA), d'Errico et al., Information Sciences 2021, uses density
   estimates with uncertainty to distinguish peaks from sampling fluctuations
   and reconstruct their connections. It addresses the fixed-K issue directly.
   Density peaks still need kinetic and physical checks; significance estimates
   need care with heavily correlated trajectory samples.
   [Paper](https://doi.org/10.1016/j.ins.2021.01.010).

10. **Probabilistic motifs support intermediate membership.** Gasparotto,
    Meißner and Ceriotti, JCTC 2018, develop density-based motif recognition
    and demonstrate local coordination environments in Lennard-Jones clusters
    and molecular conformers. PAMM's probabilistic motif representation offers
    a candidate for smooth assignment across connected structural regions.
    Such assignments are statistical memberships, not automatically calibrated
    probabilities of a physical phase or its future.
    [Paper](https://arxiv.org/abs/1801.08633).

11. **TDA remains relevant to collective organization.** Hiraoka et al.,
    PNAS 2016, use persistence diagrams to characterize hierarchical geometry
    in silica, Lennard-Jones systems and Cu–Zr metallic glass. This supports
    investigating medium-range topology rather than dropping TDA because
    nearest-patch targets fluctuate. Their Cu–Zr alloy glass is not our pure
    Zr liquid, and their structural analysis does not establish a kinetic
    clustering method. [Paper](https://arxiv.org/abs/1501.03611).

12. **Topology stability requires stable inputs and a declared metric.**
    Cohen-Steiner, Edelsbrunner and Harer bound persistence-diagram bottleneck
    distance by the sup-norm change of suitable filtration functions. Adams
    et al., JMLR 2017, establish stability of persistence images under stated
    weighting and distance conditions. Arbitrary nearest-80 membership swaps
    do not meet a small-input-change premise just because the central atom moved
    little. A smooth density filtration on fixed spatial support is a candidate;
    its discretization and rotation sensitivity would require separate checks.
    [Stability theorem, conference version 2005](https://math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/Edelsbrunner,%20Harer,%20Stability.pdf),
    [persistence images](https://jmlr.org/papers/v18/16-337.html).

13. **Recent descriptor benchmarking reinforces the local/global distinction.**
    Yoshikawa et al., Communications Chemistry, July 2026, compare 16 water
    descriptors through temperature classification. The authors explicitly
    distinguish their whole-snapshot classification from identifying individual
    high-/low-density environments. This is a useful contemporary caution:
    strong condition classification is insufficient evidence of meaningful
    within-liquid local clusters.
    [Paper](https://www.nature.com/articles/s42004-026-02097-1).

A 2026 preprint found during this search is
[Fiorentino et al.'s feature-selection preprint](https://arxiv.org/abs/2602.00660).
Its glass/liquid terminology concerns a hypothetical model of feature selection
on protein datasets. It is not evidence that DII identifies states of an atomic
liquid. It is therefore not part of the proposed Al/Zr validation argument.

**What is already in this repository**

The September 1–3 research already considered kinetic coordinates and
isoconfigurational futures. In particular,
[the earlier atlas assessment](../../docs/predictive_atlas_current_progress_20260903.md)
reports that linear VAMP did not improve the tested future-law retrieval over
static GeoFrame PCA. The
[experience record](../../docs/predictive_atomic_dynamics_experience_20260902.md)
identifies thermal variation, phase progression, run identity and insufficient
independent parents as distinct obstacles. These negative results remain
relevant; VAMP is not a newly discovered cure for this project.

The existing [LinearVAMP implementation](../../src/temporal_vamp/linear_vamp.py)
already accumulates covariances in float64, filters near-null directions,
regularizes whitening and scales coordinates by singular values. The narrow
new question is whether it can expose useful persistent directions in the
new, continuous MACE features, evaluated within liquid conditions. Reuse it
for a bounded diagnostic, not an automatic return to a large VAMP training run.
Historical configurations listed in `docs/temporal_vamp.md` are archived and
must not be treated as current launch recipes.

Our current [physical readout result](RESULTS.md) makes a second intervention
especially plausible: collective qbar6 is predictable from the combined features
on a spatial holdout even though their clustering barely separates it. Metric
learning is therefore an alternative to collecting more features. That spatial
holdout result itself is not fresh-source encoder validation.

**Revised experimental order — proposals, not established results**

| Priority | Bounded experiment | Question answered | Main acceptance evidence |
|---|---|---|---|
| 1 | Freeze MACE. Compare raw inner, combined features, and their regularized linear temporal coordinates; keep the existing projector as a reference. | Does the available representation already contain persistent collective directions? | Raw-coordinate temporal increments and correlations, within-liquid spatial variation, independent physical observables, source-held-out predictability. |
| 2 | Learn sparse feature weights with DII against a declared collective-structure reference; compare with a small linear physical mapping. | Is the main failure the chosen clustering distance? | Physical-neighborhood retrieval on new sources; generalization to observables excluded from training; no boundary/quantization amplification. |
| 3 | Compare continuous coordinates with DPA/PAMM partitions and low-dimensional Onion analysis over several durations. | Are there supported recurring motifs or lifetimes, and where are transitions ambiguous? | Population and assignment reproducibility, physical distributions, membership uncertainty, transitions, and unclassified fraction. |
| 4 | Add multiscale smooth pooling, angular correlations, and optional neighborhood-distribution summaries. | Is the relevant collective information absent or diluted at the current spatial scale? | Improvement over the best frozen-feature baseline at matched physical support and matched evaluation points. |
| 5 | Fine-tune a compact exported structural representation with temporal and collective objectives, preserving a local-detail branch. | Can the encoder improve beyond readout/metric selection? | Joint stability, physical and kinetic gains across held-out source trajectories and materials; no reliance on label smoothing for the gain. |

For priority 1, start with the available 0.75, 1.5, 3, 6 and 12 ps lags as a
screen, then set useful lag ranges from measured structural decorrelation and
transition durations. The 144 short diagnostic tracks are an exploratory cohort,
not sufficient evidence for rare-state kinetics or high-rank fitting. Use
independent training and validation trajectories for selecting dimensions and
regularization; preserve final sources for testing. Keep pre-crystallization
liquid, interfaces and crystal as separate evaluation strata. Report whether
the leading mode just predicts temperature, crystallinity, time or source.

Unlike unconditional concatenation, a learned transformation of combined
features might retain a slow, useful part of the center block. Its useful local
prediction signal should not be discarded in advance. Compare that possibility
with inner-only under exactly the same protocol.

For priority 2, possible reference features are qbar4/qbar6, orientation
coherence, smooth shell-density profiles and suitable medium-range topology.
Reference scaling must be declared and fitted on training data. Reserve some
structural descriptors and all final dynamical outcomes for validation/testing;
showing separation in exactly the descriptor used to teach the distance is
circular evidence. A physical metric is an interpretable supervised baseline,
not a claim of label-free discovery. Do not use raw 512D distances as the sole
reference and expect their present physical shortcomings to disappear.

For priority 3, use the original features or learned coordinates for clustering;
UMAP remains a display. Report raw instantaneous assignments separately from
duration-filtered or history-dependent assignments. An Onion analysis can use
future points to classify a time window. A forecasting encoder must instead
use a snapshot or causal past history. Any teacher derived from full trajectories
needs split boundaries and an explicit account of what the deployed student
can observe. Stable structural motifs and brief rearrangement events are
different outputs; no single partition has to represent both.

For priority 4, distinguish larger context, larger pooling, and preserving
cross-neighbor correlations. Averaging scalar invariants is not equivalent to
averaging angular quantities before forming invariants. Start with physically
defined, smooth multi-radius statistics, and add learned tensor correlations
only if their benefit is identifiable. Determine radii from each material's
structure; do not transfer water's radii or silently conflate physical and
rescaled coordinates in the existing Zr transfer setup.

For priority 5, use separate ablations for temporal-only, collective-only and
combined supervision. A term that matches neighboring states should weaken
across physically distinct interfaces. Genuine structural changes must remain
detectable. Retain instantaneous TDA/local-property targets in the detail branch
while assessing persistent topology in the structural branch. If history is
necessary to separate environments, report that requirement explicitly instead
of attributing its gain to a snapshot encoder.

**Keeping stability while gaining information**

A finite linear map of a continuous embedding remains continuous, but its
operator norm can amplify small changes: `||A delta_z|| <= ||A|| ||delta_z||`.
Consequently, preserving MACE weights alone does not establish practical
stability after whitening or feature reweighting. Repeat rotation, permutation,
membership-crossing and storage-round-trip tests on the exported coordinates.
Measure amplification relative to natural structural changes, not only in
arbitrary raw units. Use the already demonstrated smooth-inner result as the
reference, rather than the less coherent concatenated vector.

Neighbor turnover deserves its own measured channel. Keep representation
continuity through arbitrary sampling boundaries, while recording physical
exchange within a declared neighborhood. Sweep that neighborhood and use
residence-time checks to identify cutoff chatter. A LENS-like signal computed
on our analysis patch is not automatically a physical exchange observable.
For a direct literature baseline, TimeSOAP is defined on SOAP spectra;
substituting MACE increments should be named as an adaptation and tested.
The original [TimeSOAP paper](https://arxiv.org/abs/2302.09673) provides that
descriptor baseline.

Maximum spatial agreement is not the objective. Overlapping averages can raise
it mechanically and blur interfaces. Require physically interpretable spatial
correlation lengths, independent descriptors and dynamics in addition to label
agreement. Likewise, if a well-tested liquid remains a connected distribution,
continuous coordinates and uncertain motif memberships can be the correct
result. Require evidence before assigning a fixed count of liquid states.

**Reusable public implementations checked**

- Qiu et al.'s [author code](https://github.com/YunruiQIU/supercooled-dynamics)
  accompanies the TCCA/autoencoder study.
- DII is documented in [DADApy](https://dadapy.readthedocs.io/en/latest/diff_imbalance.html).
- [Onion's maintained documentation](https://onion-clustering.readthedocs.io/en/stable/)
  currently identifies `tropea-clustering` as the package. The
  [Dynsight interface](https://dynsight.readthedocs.io/en/latest/onion.html)
  distinguishes the original segmented algorithm from a newer unsegmented
  version. Select and record the version/protocol before any comparison.
- DPA's original paper links its [author implementation](https://github.com/mariaderrico/DPA).

No external package was installed and no training was launched for this review.
The recommended first result is a small, reproducible comparison that separates
temporal-coordinate learning, physical metric learning and state assignment.
