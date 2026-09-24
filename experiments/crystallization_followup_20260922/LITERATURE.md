# Literature review: predicting local Al crystallization

Reviewed 22 September 2026. This is a targeted review, not a systematic survey.
The task is onset in a tracked local environment over the next 3–12 ps, with
secondary 96 ps structural forecasts. Recognizing an existing crystal, predicting
bulk growth rates, and predicting glassy relaxation are related but different tasks.

## Evidence and its limits

| Primary paper | Relevant result | Experiment suggested here | Limitation |
|---|---|---|---|
| [Hu and Tanaka, Nature Communications (2022)](https://www.nature.com/articles/s41467-022-32241-z), *Revealing the role of liquid preordering in crystallisation of supercooled liquids* | Perturbing crystal-like preordering in NiAl changes nucleation and growth; preordered liquid wets the interface. | Preserve order and its recent development; subsequently measure interface geometry and orientation coherence. | NiAl B2/bcc ordering is not pure-Al FCC ordering; this is a mechanism hypothesis for our data, not a transferred quantitative law. |
| [Freitas and Reed, Nature Communications (2020)](https://www.nature.com/articles/s41467-020-16892-4), *Uncovering the effects of interface-induced ordering of liquid on crystal growth using machine learning* | Local liquid structure near silicon crystal interfaces affects attachment kinetics; interface orientation matters. The analysis uses briefly minimized saved configurations. | Test original and relaxed observations together; distinguish a nearby advancing interface from locally isolated ordering. | Their seeded silicon growth setting and short steepest-descent minimization differ from our spontaneous Al trajectories and converged full-cell quenches. It does not establish that more order always means faster growth. |
| [Jung, Biroli and Berthier, Physical Review Letters (2023), author manuscript](https://arxiv.org/html/2210.16623v2), *Predicting dynamic heterogeneity in glass-forming liquids by physics-informed machine learning* | GlassMLP uses multiscale inherent-structure descriptors, including energy and density-related quantities, to predict isoconfigurational propensity and its spatial organization. | Retain relaxed structure, complement it with observed fluctuations, and keep spatial diagnostics. | The models are glass formers; the 2D model suppresses crystallization. Propensity averages randomized velocities, unlike one observed onset trajectory. This does not establish optimal Al descriptors or an attainable onset AP. |
| [Qiu et al., author manuscript (2024)](https://arxiv.org/abs/2404.04473), *Unsupervised machine learning for supercooled liquids*; [published study (2025)](https://doi.org/10.1073/pnas.2427246122), *Unsupervised learning of structural relaxation in supercooled liquids from short-term fluctuations* | A time-lagged autoencoder extracts a structural variable correlated with long-time propensity using short-time changes; the manuscript stresses radial information over multiple length scales. | Test dense recent structural observations against repeated-current controls. | Correlation with glassy propensity is not demonstrated local crystallization timing. Our first wave tests input information, not reproduction of their autoencoder. |
| [Lechner and Dellago, Journal of Chemical Physics (2008), author manuscript](https://arxiv.org/abs/0806.3345), *Accurate determination of crystal structures based on averaged local bond order parameters* | Averaging spherical-harmonic bond-order vectors over neighboring particles improves structural classification in their systems. | Later compare explicit inter-neighborhood orientational coherence with scalar-only contexts. | Structural recognition is not forecasting. Our current descriptors already include qbar6 and mean q6 coherence; simply adding those again would not be a new test. |
| [Gensheimer and Narasimhan, PeerJ (2019), author manuscript](https://arxiv.org/abs/1805.00917), *A scalable discrete-time survival model for neural networks* | A conditional-hazard likelihood accommodates censoring and time-dependent effects. | Retain the existing hazard likelihood and add a correctly censored 12 ps auxiliary term when testing horizon emphasis. | The survival formulation already exists in our predictor. It supplies a statistical objective, not missing physical information or proof of calibration. |

## How repository evidence changes the priority

The [earlier information audit](../../output/crystallization_information/short-horizon-20260920/RESULTS.md)
found larger improvements from outer geometry and recent structural history than
from the tested motion summaries. For example, Epi-MACE plus outer geometry had
12 ps AP 0.4843 versus 0.1939 for the embedding-only MLP, on that audit's own
population. Those values must not be compared directly to the current sparse
archive cohort. The current model already has geometry-aware spatial attention,
temporal attention, physical/order descriptors and two scalar shell summaries.
Its shell entries are weighted count and mean radius in 7–17 and 17–25 A annuli,
not interface distance, cluster size or crystal orientation.

The [matched relaxed comparison](../../output/crystallization_transfer/symmetric-relaxed-reuse-20260921/RESULTS.md)
supports testing complementary domains: relaxed AR AP12 was 0.6250 versus 0.5787
for original AR. This combines checkpoint, input-domain and local-support changes.
The old [forecast refinement study](../crystallization_transfer_20260919/PATH_REFINEMENT.md)
already tested event-stratified mixtures and improved diffusion numerics without
consistent gains over deterministic forecasts. Another broad generative-model
sweep is therefore lower priority than diagnosing missing observed information.

## Testable hypotheses

1. A relaxed snapshot preserves useful underlying order while original short-time
   fluctuations add kinetic information. Compare separately normalized domain
   branches with a same-capacity repeated-relaxed branch; never subtract unrelated
   learned latent coordinates.
2. Structural secants computed at their actual physical intervals are more useful
   than unscaled differences across irregular observations. Explicit lag features
   distinguish an observed change from its rate. This is an engineering hypothesis,
   not a result established by the cited papers.
3. Original descriptors at t-12,t-6,t-3,t-.75,t add predictive information beyond
   the latest original descriptor. A separately trained repeated-current control
   holds auxiliary capacity and input width fixed.
4. Original-minus-relaxed *physical descriptors in the same basis* encode useful
   susceptibility to quenching. This is distinct from additional original latent
   features. The first implementation does not claim energy or displacement data.
5. Explicit short-horizon likelihood and physical prediction terms improve imminent
   onset skill while retaining 96 ps trajectories. All candidates use the same
   12 ps selection objective so checkpoint selection is not the experimental change.
6. Absolute simulation age may act as a protocol-specific shortcut. Remove only
   that condition in a diagnostic; retain temperature and actual history offsets.

## Subsequent experiments, not included in the first executable queue

**Interface/coherence extraction:** use existing observed coordinates and past
frames to derive nearest ordered-component distance, component size and motion,
plus q4m/q6m contractions across context neighborhoods. Compare to scalar-order
controls and stratify by the presence of a nearby observed crystal. Distinguish
hard current structure labels from future-confirmed onset labels; the latter
must never enter inputs. This requires a new descriptor extraction, not new MD.

**Paired support control:** encode both domains with matched trained support, or
train matched-support encoders, before attributing gains specifically to relaxation.

**Coherent stochastic paths:** if timing uncertainty persists, couple sampled
onset time to a shared trajectory-level latent state. Check transition durations,
temporal correlations, marginal calibration and proper scores rather than only
mean-path MSE. Prior mixture results justify requiring these mechanism checks.

**Branch uncertainty:** audit existing shooting ancestry and perturbation protocol
before estimating conditional event probabilities. Hold all descendants of a root
in one split. Branch variability alone cannot establish a universal prediction ceiling.

No paper reviewed here justifies expecting a larger backbone, stronger smoothing,
or better-looking UMAP clusters to improve the specific onset task.
