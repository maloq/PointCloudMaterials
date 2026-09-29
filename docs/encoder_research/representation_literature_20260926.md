# Learning local structural and dynamical representations: literature review

Date: 26 September 2026. Scope: the representation problem behind our Al/Ta/Zr
encoder experiments, including relevant ideas outside atomistic materials.
This is a research synthesis and proposed direction, not an implemented protocol.

**Repository-history correction, 26 September:** the subsequent
[experiment audit](../../experiments/encoder_mechanisms_20260926/README.md)
found that linear VAMP and conditional-future-law learning were already tested
here. Linear VAMP lost to static PCA on the harder historical assay. Consequently
the original kinetic-learning priority below is superseded by the audit's staged
readout/alignment/adaptation experiments; kinetic learning is a conditional
follow-up. Literature relevance alone does not establish an untried direction.

## Answer and main implications

**Yes. There is close atomistic precedent, and a substantial broader literature.**
The closest methodological precedent is GDyNet [1]; continuous dynamical coordinates
and meaningful distances are addressed by SRV and kinetic maps [3–4]. Predictive
states and local causal states [8–9] give a broader formulation of what we want.
The recent Ta/Zr study [20] makes competing liquid order especially relevant.

Our objective contains several distinct requirements:

| Requirement | What it means here | What does not establish it |
| --- | --- | --- |
| Structural information | Liquid motifs, interfaces and defects remain accessible | Global variance or a liquid/crystal classifier |
| Useful geometry | Nearby embeddings correspond to similar declared physical or dynamical properties | Successful decoding with an unrestricted nonlinear head |
| Appropriate continuity | Small nuisance changes have small effects; meaningful rearrangements remain detectable | Minimizing every temporal or spatial difference |
| Predictive information | Present observations distinguish distributions of future outcomes | An attractive manifold or retrospective nucleus pictures |
| Spatial organization | Coherent domains and interfaces are resolved at the declared observation scale | Making every neighboring pair similar |

These requirements can conflict. A useful representation may have several
continuous directions, branches and metastable regions. We should not require a
single liquid-to-crystal line, seven discrete states, or visibly separated UMAP
clusters. In particular, green/blue/purple image colors are hypotheses about
distinct structures, not physical labels by themselves.

The proposed organizing idea is **a local structural state with a validated
dynamical interpretation**. Its dimensions should preserve competing forms of
order and useful future distinctions while identifying truly equivalent views.
This is our synthesis of the literature, not an established solution for our data.

## Closest atomic precedents

**[1] Xie et al. (2019), “Graph Dynamical Networks for Unsupervised Learning of
Atomic Scale Dynamics in Materials.”**
[Paper and methods](https://arxiv.org/html/1902.06836);
[Nature Communications](https://www.nature.com/articles/s41467-019-10663-6).
GDyNet combines local graph representations with a VAMP dynamical objective.
For a Si–Au solid/liquid interface it distinguishes bulk solid, bulk liquid,
solid-side interface and liquid-side interface, without supplying these four
physical classes as training labels. The authors check transition dynamics.
Crucially, they restrict target atoms to liquid and the first two solid layers:
including deeper solid would emphasize slower processes outside their question.
The model uses a chosen number of soft states. It does not establish a universal
continuous metric or transfer to Al/Ta/Zr. **Closest methodological precedent.**

**[19] Becker, Devijver, Molinier and Jakse (2022), “Unsupervised topological
learning approach of crystal nucleation.”**
[Scientific Reports](https://www.nature.com/articles/s41598-022-06963-5);
[accessible article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8873400/).
Persistent-homology descriptors and mixture models distinguish local environments
during Ta, Al and Mg crystallization. In Ta, the analysis separates solid,
distorted solid, a predominantly boundary-associated class and different liquid
classes. It uses quenched structures and interprets topology and spatial maps.
This is close to our desired interface/defect picture, but does not establish
prospective, calibrated prediction of new nuclei. Per-system clusters and their
numbers are not universal physical species. **Closest structural-analysis precedent.**

**[29] Qiu, Jang, Huang and Yethiraj (2025), “Unsupervised learning of structural
relaxation in supercooled liquids from short-term fluctuations.”**
[PNAS](https://doi.org/10.1073/pnas.2427246122);
[article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12012455/);
[earlier preprint, different title](https://arxiv.org/abs/2404.04473).
Time-lagged canonical correlation and autoencoding extract order parameters from
short-time structural correlations in a Kob–Andersen glass former. Lags roughly
a thousand times shorter than relaxation can expose long-time propensity;
medium-range radial structure matters. This is evidence that temporal structure
can reveal distinctions inside a liquid without propensity labels. It studies
glassy relaxation, not metal nucleation, and uses species-resolved descriptors.
The transferable idea is short-lag correlation; physical-reconstruction
pretraining remains discontinued in this repository.

**[20] Hu et al. (2025), “Monatomic glass formation through competing order
balance.”**
[Nature Communications](https://www.nature.com/articles/s41467-025-63221-8).
Directly relevant Ta/Zr simulations connect different crystallization behavior
with competition among crystalline, icosahedral and quasicrystal-related order.
Both systems crystallize to BCC under the studied conditions; Ta shows stronger
competing order and glass-forming ability. Our inference is that a representation
should distinguish crystal-compatible and crystal-frustrating structured liquid.
“More ordered” need not mean “more likely to crystallize.” These phases, potential
choices and thresholds must not be transferred unverified to our trajectories;
in particular this does not make every Zr experiment a BCC problem.

## Dynamical representations: learn evolution rather than erase it

**[2] Mardt, Pasquali, Wu and Noé (2018), “VAMPnets for deep learning of molecular
kinetics.”** [Nature Communications](https://www.nature.com/articles/s41467-017-02388-1).
VAMP learns features whose lagged evolution admits a useful linear model;
whitened cross-covariances provide a variational score. Original VAMPnets use
soft state memberships and validate kinetics with implied timescales and
Chapman–Kolmogorov checks. VAMP does not require detailed balance, but this does
not guarantee a closed, time-homogeneous model for a partially observed atom
during cooling. A high training score is insufficient validation.

**[3] Chen, Sidky and Ferguson (2019), “Nonlinear Discovery of Slow Molecular
Modes using State-Free Reversible VAMPnets.”**
[Paper](https://arxiv.org/html/1902.03336).
SRV learns continuous slow functions rather than forcing a finite softmax state
partition. This directly addresses our wish to retain continuous organization.
Its reversible equilibrium formulation needs conditions our crystallization
data may not satisfy. It motivates a continuous dynamical representation, not
uncritical use of a reversible objective on a cooling trajectory.

**[4] Noé and Clementi (2015), “Kinetic distance and kinetic maps from molecular
dynamics simulation.”** [Paper](https://arxiv.org/html/1506.06259).
Kinetic distance compares conditional future distributions. Under the paper's
Markov/equilibrium assumptions, suitably scaled slow coordinates give a Euclidean
map of that distance. Thus the metric has a specified physical interpretation,
beyond information being decodable. Lag determines which differences matter;
fast structural distinctions may receive little weight. Its particular
equilibrium-weighted distance should not be presumed valid for our pooled
nonequilibrium trajectories.

**[5] Bittracher et al. (2018; preprint 2017), “Transition manifolds of complex
metastable systems: Theory and data-driven computation of effective dynamics.”**
[Paper](https://arxiv.org/html/1704.08927).
Under transition-reducibility assumptions, low-dimensional coordinates can
preserve dominant relaxation timescales by representing transition distributions.
This provides a theoretical reason to compare futures rather than instantaneous
Cartesian similarity. It does not guarantee such a low-dimensional manifold
exists for an arbitrary local atomic observation.

**[6] Wang and Tiwary (2021), “State predictive information bottleneck.”**
[Paper](https://arxiv.org/html/2011.10127);
[JCP](https://doi.org/10.1063/5.0038198).
SPIB compresses present observations while predicting future states and refining
state assignments. Delay controls the resolution of discovered states. It offers
a useful predictive-state perspective, but compression may merge structures
irrelevant to its chosen future labels. It is not a guarantee that interface
morphology or all liquid motifs survive, nor that a finite-horizon predictor is
automatically a committor.

Our proposed mathematical distinction is:

\[
z_t=E(O_t),\qquad
\mathbb E[z_{t+\tau}\mid O_t]\approx K_\tau z_t,
\]

where \(O_t\) is the declared local observation. This asks the representation to
support evolution. Directly minimizing \(\|z_{t+\tau}-z_t\|^2\) instead asks it to
change little. Slowness can help reject vibrations, but it can also suppress
rearrangements we want to study. A predictor in JEPA can already accommodate
change; we must inspect actual pairs, targets and exported layers before claiming
that the implementation enforces direct temporal invariance.

For an exploratory continuous VAMP variant, a standard objective is the squared
Frobenius norm of \(C_{00}^{-1/2}C_{01}C_{11}^{-1/2}\), estimated at declared lags.
Covariance rank, regularization, effective sample size and held-out scores matter.
With our default 256-sample batches and 128-dimensional exports, correlated
neighborhoods can make covariance estimation noisy. We must measure conditioning
and consider a smaller kinetic head or an explicitly declared covariance-estimation
scheme before assuming this is a stable drop-in loss.

## Related ideas beyond atomic encoders

**[7] Schneider, Lee and Mathis (2023), “Learnable latent embeddings for joint
behavioural and neural analysis” (CEBRA).**
[Nature](https://www.nature.com/articles/s41586-023-06031-6).
Contrastive sampling based on temporal or auxiliary-variable relationships gives
continuous neural embeddings evaluated for consistency and decoding across runs
and recording conditions. This is useful precedent for reproducible latent
organization. Behavior-supervised and temporal modes have different supervision;
the method does not make arbitrary positives physically valid. We should borrow
consistency checks and explicit pair semantics, not feed simulation age or
temperature into our models.

**[8] Shalizi and Crutchfield (2001), “Computational Mechanics: Pattern and
Prediction, Structure and Simplicity.”**
[Authors' paper page](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm).
Predictive states group histories that imply the same conditional distribution
of futures. This is a principled definition of predictive equivalence, rather
than grouping states by a single future binary label. “Causal state” here is a
technical predictive concept; it is not evidence that a learned coordinate has
an experimentally established causal effect.

**[9] Rupe and Crutchfield (2018), “Local Causal States and Discrete Coherent
Structures.”** [Paper](https://arxiv.org/html/1801.00515).
Local past/future lightcones identify coherent structures in discrete
spatiotemporal systems, demonstrated with cellular automata. This is a close
conceptual match to discovering spatially organized states by their behavior.
Atomic neighborhoods are continuous, moving, irregular and partially observed;
the paper does not validate an atomic implementation. Translating it would be a
substantial modeling project, not a loss-function replacement.

**[10] Zhang et al. (2021; preprint 2020), “Learning Invariant Representations for
Reinforcement Learning without Reconstruction.”**
[Paper](https://arxiv.org/html/2006.10742).
Bisimulation-based representations compare states using reward and transition
behavior, excluding irrelevant distractors. This illustrates how latent distance
can be tied to behavior without reconstructing inputs. Equivalence is explicitly
task-dependent: if the only task is eventual crystallization, physically distinct
interfaces may legitimately be merged. We need a richer declared set of futures
if those distinctions are part of the scientific objective.

**[18] Locatello et al. (2019), “Challenging Common Assumptions in the Unsupervised
Learning of Disentangled Representations.”**
[ICML paper](https://proceedings.mlr.press/v97/locatello19a.html).
Theoretical non-identifiability and a large empirical study show that
disentanglement needs inductive assumptions about models and data. This is
relevant to our historical Factor/VAE experiments: independent-looking latent
coordinates alone cannot certify recovered physical factors. Dynamics, known
symmetries and independent physical checks supply additional assumptions/evidence.

**[35] Dietrich and Salvalaglio (2025), “On the reproducibility of free energy
surfaces in machine-learned collective variable spaces.”**
[JCP](https://doi.org/10.1063/5.0287912);
[authors' code and synopsis](https://github.com/mme-ucl/MLCVs_FE).
The authors study dependence of learned-coordinate free-energy surfaces on the
particular trained map and discuss geometric formulations. This reinforces the
need to distinguish coordinates from physical invariants. A density landscape
in UMAP is especially insufficient evidence of a free-energy barrier. This
review screened the primary synopsis/available excerpts, not the complete derivation.

One elementary issue applies independently of any particular method: an
invertible nonlinear remapping of \(z\) can retain its information while changing
Euclidean distances dramatically. Predictive sufficiency and metric quality are
therefore separate claims. A kinetic metric adds a definition of similarity;
it does not solve the separate problem of preserving every scientifically useful
structural observable.

## Is loss of useful nuance a recognized learning problem?

**[11] Bardes, Ponce and LeCun (2022), “VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning.”**
[Paper](https://arxiv.org/html/2105.04906).
The loss combines paired-view agreement, a per-coordinate spread constraint and
decorrelation. Our inference from its scope is that global spread does not
certify retention of a particular distinction inside the liquid population.
Decorrelation also does not make coordinates independent physical mechanisms.
Check encoder and projector separately: constraints on one are not measurements
of the other.

**[12] Bardes, Ponce and LeCun (2022), “VICRegL: Self-Supervised Learning of Local
Visual Features.”**
[NeurIPS paper](https://papers.nips.cc/paper/2022/hash/39cee562b91611c16ac0b100f0bc1ea1-Abstract-Conference.html).
Local and global objectives serve different image tasks; local feature learning
helps dense prediction. This motivates checking what pooling and the training
head retain. It does not justify forcing neighboring atoms into identical
representations: image-view correspondence and atomic spatial proximity are
different relationships.

**[13] Tian et al. (2020), “What Makes for Good Views for Contrastive Learning?”**
[Paper](https://arxiv.org/abs/2005.10243).
The InfoMin argument makes preservation of task-relevant information a condition
of useful view construction. Reducing shared information indiscriminately is
not the prescription. For us, relaxation and displacement augmentations need
scientific validation; they are not equivalent to coordinate rotations.

**[14] von Kügelgen et al. (2021), “Self-Supervised Learning with Data
Augmentations Provably Isolates Content from Style.”**
[Paper](https://arxiv.org/html/2106.04619).
Under specified assumptions, augmented views identify shared content up to an
invertible map while variable style can be discarded. This explains why pair
construction defines what invariance means. Its theorem does not identify a
unique physical metric, and applying it to our augmentation scheme requires
checking the assumptions rather than citing it as a universal guarantee.

**[15] Robinson et al. (2021), “Can contrastive learning avoid shortcut
solutions?”** [Paper](https://arxiv.org/html/2106.11230).
Contrastive representations can suppress features when easier shared features
suffice; the paper analyzes and intervenes on this behavior. It supplies a
plausible mechanism to investigate. It is not a causal diagnosis of our
noncontrastive VICReg/Epi training.

**[16] Jing et al. (2022), “Understanding Dimensional Collapse in Contrastive
Self-supervised Learning.”** [Paper](https://arxiv.org/abs/2110.09348).
Even contrastive training can concentrate representations into a low-dimensional
subspace. Rank and eigenspectra are useful checks, but dimensional collapse,
semantic feature loss and an appropriate low-dimensional physical manifold
are different phenomena.

**[17] Papyan, Han and Donoho (2020), “Prevalence of neural collapse during the
terminal phase of deep learning training.”**
[Paper](https://arxiv.org/abs/2008.08186).
Late classification training can exhibit shrinking within-class variation and
organized class prototypes. It resembles the feared liquid/crystal endpoint
picture, but its setting differs from our self-supervision and hazard likelihood.
Use it as an analogy and a reason to measure conditional variation, not a proof.

Consequently, the claim to investigate is not simply “longer training destroys
physics.” We need to distinguish actual information loss, redistribution into
harder-to-read directions, changed distance scaling, projector behavior,
different objectives, and projection artifacts. Our handbook already records
that the historical epoch34 VICReg and epoch159 VISReg images were **not** an
unchanged-objective training trajectory. The 35-pass reproduction also separates
improving encoder readout from declining projector readout. See
[the evidence summary](README.md) and [quality criteria](quality_criteria.md).

## Physical evidence, alternatives and counterexamples

**[21] Hu and Tanaka (2022), “Revealing the role of liquid preordering in
crystallisation of supercooled liquids.”**
[Nature Communications](https://www.nature.com/articles/s41467-022-32241-z).
The study relates crystal-compatible liquid order to nucleation and growth and
uses suppression of preordering to probe its role. This is stronger mechanistic
evidence than a colored map. It does not make all locally favored order
crystal-promoting; thresholds and interventions are system-dependent.

**[22] de Jager, Smallenburg and Filion (2023), “In search of a precursor for
crystal nucleation of hard and charged colloids.”**
[Paper](https://arxiv.org/html/2306.05886).
Classical order, topology and unsupervised analysis reveal no detectable
pre-nucleation structural precursor in the studied systems. This includes
tracking the birthplace and relevant structural correlation times. It is a
necessary counterexample to a universal precursor narrative, not evidence that
Al/Ta/Zr must behave identically. A useful analysis must permit a negative result.

**[23] Díaz Leines and Rogal (2018), “Maximum Likelihood Analysis of Reaction
Coordinates during Solidification in Ni.”**
[Paper](https://arxiv.org/abs/1810.04782).
Transition-path sampling and likelihood analysis find value in the surrounding
prestructured liquid when describing the solidification reaction coordinate.
This motivates evaluating regional context and reaction-coordinate quality,
rather than assuming nucleus size alone is sufficient. Local atom arrival is a
different target from a nucleus's fate.

**[24] Chapman et al. (2023), “Quantifying disorder one atom at a time using an
interpretable graph neural network paradigm” (SODAS).**
[Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10328988/).
A GNN supplies a continuous local ordering score, demonstrated on disordered Al
including interfaces. Its supervision is tied to thermal ensembles. This is a
precedent for continuous atomic disorder, not a demonstrated multidimensional
liquid-state metric or precursor predictor. Temperature-derived training targets
must be recorded separately from actual encoder inputs; this is not proposed
as a silent replacement for our condition-free setup.

**[25] Freitas and Reed (2020), “Uncovering the effects of interface-induced
ordering of liquid on crystal growth using machine learning.”**
[Nature Communications](https://www.nature.com/articles/s41467-020-16892-4).
The work links interface-induced liquid ordering to growth in Si and Cu.
It supports studying liquid-side interface structure as a separate object.
An existing interface is available in this setting; it should not be counted
as evidence of forecasting independent homogeneous nucleus birth.

**[26] Boattini et al. (2020), “Autonomously revealing hidden local structures
in supercooled liquids.”**
[Nature Communications](https://www.nature.com/articles/s41467-020-19286-8).
Unsupervised structural analysis exposes distinctions inside supercooled liquids
associated with heterogeneous dynamics. This challenges a binary phase-only
view, but relaxation associations do not establish crystallization propensity.

**[27] Schoenholz et al. (2016), “A structural approach to relaxation in glassy
liquids.”** [Paper](https://arxiv.org/abs/1506.07772).
Softness connects learned local structural distinctions to rearrangement.
It is a behavior-linked structural coordinate, with supervision defined by
rearrangement, rather than a universal unsupervised liquid/crystal coordinate.

**[28] Bapst et al. (2020), “Unveiling the predictive power of static structure
in glassy systems.”**
[Nature Physics](https://www.nature.com/articles/s41567-020-0842-8).
Graph learning predicts glassy dynamics from static structure. This supports
testing structure-to-dynamics information and observation range. It does not
show that every single trajectory's future is fixed by a small local patch.
The present review uses the publisher's abstract-level evidence for this paper.

**[30–32] Classical reference measurements.**
Lechner and Dellago (2008),
[averaged bond-order parameters](https://arxiv.org/abs/0806.3345), improve crystal
structure separation through neighbor averaging. Larsen, Schmidt and Schiøtz
(2016), [polyhedral template matching](https://arxiv.org/abs/1603.05143), improve
structural identification in distorted/thermal environments and provide
orientation information. Malins et al. (2013),
[topological cluster classification](https://arxiv.org/abs/1307.5517), identify
local motifs by bond topology. These are complementary reference views;
“unclassified by a crystal template” does not automatically mean unstructured
liquid. Average-order, motif, template-residual and connectivity measurements
should retain their separate meanings and uncertainties.

**[33] Sheriff, Freitas, Trewartha and Torrisi (2024), “Simultaneous Discovery of
Reaction Coordinates and Committor Functions Using Equivariant Graph Neural
Networks.”** [Workshop paper](https://openreview.net/pdf?id=NX2ROvVb2Y);
[authors' implementation](https://github.com/TRI-AMDD/interstate).
This AI4Mat workshop work demonstrates an equivariant route on alanine dipeptide
and CrFeNi solidification. It is a relevant implementation lead, with a narrower
evidence base than an established general solution. The primary indexed text
and author repository were screened; the complete validation was not audited.

**[34] Dietrich et al. (2023), “Machine Learning Nucleation Collective Variables
with Graph Neural Networks.”**
[JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00722).
GNN surrogates accelerate specified nucleation collective variables, including
transfer from a colloidal model to copper crystallization. This is useful
precedent for efficient representations of chosen observables. Approximating a
known order parameter is a different scientific claim from discovering new
predictive information.

For classical references in our setting, use high-confidence bulk phases,
template residuals, grain orientation/fault structure, several bond-order and
topological measurements, and connections to established crystal. Calibrate
thresholds on permitted training/reference sources for the actual potential.
Retain continuous values and ambiguous cases. Color-based cluster purity should
be secondary to these independent checks. This is our proposed evaluation
practice, not a claim of uniquely defined ground truth at an interface.

## What this says about our current evidence

The [eight-checkpoint MACE report](../../experiments/encoder_quality_20260926/RESULTS.md)
already exhibits the distinctions above:

- Observed-input Epi fine-tuning improves calibrated 6 ps log loss from 0.07073
  to 0.06916 relative to its pretrained export, while liquid-neighbor error and
  physical/latent change association worsen. This is a trade-off across objectives.
- Adding physical descriptors to the Epi readout improves 6 ps log loss from
  0.06916 to 0.06743. This establishes a limitation of the tested readout, not
  whether the missing utility is absent from the embedding or hard to access.
- Al boundary AUROC around 0.54–0.57 is weak despite accessible class information.
  Decodable structure and useful latent distances are not interchangeable.

These measurements condition on the particular trained models, available static
frames and reused evaluation cohort. They do not establish a universal objective
ranking. Existing onset labels predominantly concern arrival from existing
crystal. A good result there does not establish pre-nucleus prediction in isolated
liquid, especially in Ta/Zr, where current static proxies have no future validation.

## Recommended next scientific comparison

This section is a proposal, not a submitted training queue. Consult
[DATASETS.md](../../DATASETS.md) and refresh availability before selecting any
new trajectories; the review does not certify that all required observations exist.

**Conditional direction: continuous local dynamical coordinates.** Retain the efficient
geometry-only MACE backbone and compare current VICReg/Epi controls against a
continuous, nonreversible VAMP-style objective. Start with individual declared
lags; consider a multi-lag variant only after identifying the lag trade-off.
Use exact geometric equivalences for invariant views. Do not declare genuine
structural evolution, arbitrary spatial neighbors or relaxation to be exact
invariances. Evaluate the exported encoder and any kinetic/projector head
separately. This combines precedents [1–4,29]; it is not a published, validated
Al/Ta/Zr recipe. The repository-history correction above changes the order of
work: establish benefit in a bounded diagnostic before new kinetic training.

**Priority 2: test where the representation loses utility.** With the same
checkpoint, compare linear, controlled-capacity nonlinear and neighbor readouts;
inspect within-liquid and interface conditional spectra and neighbors. Match
sample counts and fit capacity for descriptor add-back. Compare nuisance
perturbations with real rearrangements at comparable displacement. This is more
informative than enlarging the architecture based on UMAP alone.

**Priority 3: competing-order and genuine birth evaluation.** Separate at least
ordinary liquid, crystal-compatible liquid, competing ordered liquid, liquid-side
interface, defective solid and well-ordered solid where reference evidence
supports them. Allow overlap/ambiguity rather than forcing a universal partition.
Use ancestry to separate an advancing interface from new regional establishment.
Before claiming prediction before birth, exclude existing crystal from the
**entire actual receptive field**, including halos and any relaxation support.

Keep at least 12 epochs for matched training, observe trajectories of metrics
throughout, and use multiple independent training seeds (three is a practical
starting point, not a guarantee of power). Keep frozen cohort/source roles and
the recorded data contract. Select self-supervised checkpoints with their
declared label-free objective; select predictive fits using likelihood.
Repeated scientific audits are exploratory until checked on untouched sources.

New native encoders remain geometry-only with one constant atom channel,
128 channels/export dimensions and batch/microbatch 256 unless an explicit
ablation records a deviation. Encoder/predictor inputs and any history, halo,
motion or training teacher must be separately recorded. No temperature, age or
absolute-time inputs; lags only organize observations and targets. No
physical-reconstruction treatment, AP objective/selection, or undocumented
switch to offline training tracking is proposed.

## Evidence that would support or falsify the hypothesis

| Scientific claim | Required measurement/control | Failure or narrower interpretation |
| --- | --- | --- |
| Within-liquid nuance survives training | Held-out liquid-only physical readouts, neighborhood errors and conditional spectra across epochs; compare encoder/head | Good global rank with poor liquid readout is insufficient |
| Interface types are distinct | Classwise reference agreement and distance-matched boundary tests, excluding bulk-only shortcuts; inspect ambiguity and coverage | Accessible classification with near-random boundary distance supports readout utility only |
| Temporal coherence is physically selective | Compare exact symmetry, small vibration and true rearrangement response; match displacement and coarse order | Equal suppression of noise and rearrangement means over-invariance |
| The learned coordinates describe kinetics | Held-out lagged scores and transition predictions; lag sensitivity; CK and timescale checks where Markov/time-homogeneous assumptions apply | Training score without held-out consistency, or failure on new sources, rejects the kinetic claim |
| Embedding distance has predictive meaning | Future-distribution agreement among fitting-set neighbors, controlling current order; proper held-out probabilistic scores | A nonlinear head succeeding while neighbors fail indicates a geometry problem |
| Gains exceed easier physical information | Descriptor-only, z-only and controlled-capacity z-plus-descriptor readouts; source bootstrap and multiple training seeds | Add-back gains reveal remaining accessible utility, not automatic proof of information loss |
| Structured liquid is a precursor | Independent birth trajectories, lead-time analysis, matched non-birth regions and complete visibility audit | Mere proximity to an existing front is an interface result |
| A coordinate approximates a committor | A/B basins defined in advance; independent shooting/replicas under declared dynamics; conditional outcome dispersion | Finite-horizon arrival probability or a single observed fate is not a validated committor |

Future likelihood, Brier score and calibration remain central predictive checks;
AP at 3/6 ps stays diagnostic. Independent trajectories/sources are statistical
units, not millions of correlated atoms. No weighted universal leaderboard is
implied. Any proposed new calculation must receive its own metric definition,
implementation hash and export contract when implemented.

## Reading order and review limits

Read [1] for the closest atomic architecture/objective, [3–4] for continuous
coordinates and distances, [29] for liquid information in short-time fluctuations,
[19–20] for relevant structures, and [22] to challenge the precursor assumption.
For the abstract representation idea, read [8–10,18]; for loss/pair-design
limitations, read [11,13–17].

This is a targeted narrative review of **35 primary works**, not a systematic
review with exhaustive database coverage. Searches on 26 September 2026 covered
atomic local dynamics, interfaces/nucleation, supercooled-liquid heterogeneity,
continuous slow modes/transition geometry, predictive states, contrastive
consistency, augmentation identifiability and feature suppression. Recent-paper
searches supplemented citation-following from the closest papers. Primary
publisher pages, author manuscripts/repositories and proceedings support the
claims; search dates and HTML rendering dates were not treated as publication dates.

Reading depth varies. Substantive main-text passages, including methods or
results, were inspected for [1–4,6–7,9,14,19–20,22,29]. Other entries use
abstracts, primary indexed
excerpts or author descriptions rather than an audit of every supplementary
method; [28,33,35] explicitly flag this. Some publisher/OpenReview pages blocked
direct retrieval. No cited code was run, no paper's reported numbers were
independently reproduced, and no claim that our proposed combination is novel
follows from this search.

Related local review:
[spatial warning, interface order and precursor interpretation](spatial_warning_literature_20260926.md).
Historical literature notes remain historical; discontinued treatments in them
are not revived by this document.
