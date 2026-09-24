# Encoder training after BCR: literature and next experiments

Research proposal, 22 September 2026. Repository inspected at 6183f916 plus the
current uncommitted research results. This is a targeted literature review and
proposed protocol, not a completed experiment or a launch configuration.

**Recommendation:** end the BCR training campaign. First test whether a simpler
objective can put verifiable structural information into the actual exported
MACE embedding. Use existing relaxed configurations as both a directly observed
domain and privileged training targets. Then test explicit physical distance
supervision. Keep temporal spectral learning and masked reconstruction as gated
alternatives, rather than starting another large collection of objectives.

## Evidence that changes the recommendation

The [completed BCR audit](../bcr_followup_20260922/RESULTS.md) found a real but small
conditioning benefit alongside structural deterioration. Final versus initial
exported radial RMSE worsened by 31.5% on the original melt assay, 20.7% on observed
transfer inputs, and 16.0% on relaxed transfer inputs. A fresh decoder improved
noise MSE by 3.07%, but the original decoder gained only 0.166% from the correct
code over a fitted constant. These are different comparisons; the constant
experiment does not explain away the fresh-decoder benefit. They show that the
denoising task and the desired structural state are insufficiently aligned.

The [expanded relaxed study](../../output/relaxed_encoder/expanded-20260921/RESULTS.md)
also changes the baseline. Historical 12 ps MLP average precision was 0.368 for
relaxed geometric descriptors, 0.343 for the relaxed encoder control, and 0.299
for a relaxed VICReg variant whose raw rank rose from 1.17 to 8.45. These are
single-seed point estimates on a reused cohort with overlapping event windows,
not a statistically established ranking. They nevertheless make descriptor
baselines essential and rule out treating higher rank as success. That study used
tracked nearest-80 support; its scores cannot be imported as matched controls for
the newer full-radius assay.

Physical, TDA, order, moment, neighbor/future JEPA and conditional rank losses
already exist in the [previous recipe](../relaxed_encoder_expanded_20260921/README.md).
The expanded hot-to-cold arm was stopped and excluded from its completed results;
earlier hot-to-cold pilots also exist. Thus neither physical reconstruction nor
relaxed targets are new ideas for this repository. The proposed change is to
**isolate their contribution, fix their targets, train from a common fresh
initialization, and evaluate the representation actually exported**. This also
addresses the [separate representation audit](../../output/representation_audit/liquid-structure-20260922/RESULTS.md).

## What the literature supports

These papers motivate components. None establishes that the proposed combination
will improve Al crystallization prediction.

| Primary source | Relevant result or mechanism | Consequence for this project |
| --- | --- | --- |
| Fang et al., [Geometry-enhanced molecular representation learning for property prediction](https://doi.org/10.1038/s42256-021-00438-4), Nature Machine Intelligence, 2022 | Pretraining predicts molecular bond lengths, angles and atom-pair distances. | Fixed geometric supervision is a credible starting point. Its node/pair heads do not establish sufficiency of one pooled embedding; our heads must read only that embedding. |
| Bartók, Kondor & Csányi, [On representing chemical environments](https://arxiv.org/abs/1209.3140), Physical Review B, 2013 | Smooth atomic-density representations provide symmetry-respecting structural descriptors, including SOAP. | Use smooth multiscale density/angular targets and a strong descriptor baseline. Preserve weighted counts as well as normalized shape. |
| Pozdnyakov et al., [On the Completeness of Atomic Structure Representations](https://arxiv.org/abs/2001.11696), Physical Review Letters, 2020 | Distinct environments can have identical low-body-order descriptor values. | Excellent descriptor reconstruction is not proof of complete geometry. Reserve independently defined structural observables for evaluation. |
| Jung, Biroli & Berthier, [Predicting dynamic heterogeneity in glass-forming liquids by physics-inspired machine learning](https://arxiv.org/html/2210.16623v3), Physical Review Letters, 2023 | GlassMLP predicts dynamical propensity from multiscale inherent-structure descriptors with a small neural network. | Relaxed geometry and spatial scale deserve priority over model width. This is supervised glass-former evidence, not an unsupervised Al result. |
| Lopez-Paz et al., [Unifying distillation and privileged information](https://arxiv.org/abs/1511.03643), ICLR, 2016 | Generalized distillation transfers information across training-time representations. | A relaxed configuration can teach an observed-input encoder without requiring relaxation at student inference. Our atomic implementation is an adaptation. |
| Park et al., [Relational Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html), CVPR, 2019 | Transfers relations between examples through distance and angle losses. | Test whether physically meaningful relations can organize the exported embedding, beyond making observables decodable. The paper's validation is in computer vision. |
| Qiu et al., [Unsupervised learning of structural relaxation in supercooled liquids from short-term fluctuations](https://doi.org/10.1073/pnas.2427246122), PNAS, 2025 | Short-lag TCCA and time-lagged autoencoders extract coordinates correlated with much later relaxation in a binary Kob–Andersen liquid. | Supports a cheap descriptor-level kinetic diagnostic; it does not supply a transferable lag in ps or guarantee improvement in monatomic crystallization. |
| Chen, Sidky & Ferguson, [Capabilities and Limitations of Time-lagged Autoencoders](https://arxiv.org/abs/1906.00325), Journal of Chemical Physics, 2019 | Nonlinear lagged reconstruction can mix slow and high-variance modes. | Low future reconstruction error alone is not evidence of learning useful slow coordinates. |
| Zhang et al., [PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders](https://arxiv.org/abs/2408.08753), 2024 preprint version | A point-cloud decoder can reconstruct surprisingly well from supplied masked-patch centers without encoder information. | Audit every decoder input. Masking does not automatically eliminate the alternative information path observed in BCR. |
| Yan, Li & Zhang, [GeoRecon](https://arxiv.org/html/2506.13174v2), 2025 preprint, v2 | Graph-conditioned reconstruction combines a pooled clean graph code with noisy per-node features. | Relevant molecular evidence, but still a decoder information path outside the exported code. Not my preferred successor to BCR. |

## 1. Direct multiscale geometric reconstruction

Train the native coordinate-to-MACE encoder end to end. Define fixed smooth
observables G(X): radial density at several shells, weighted coordination/counts,
angular density contractions, and cross-shell correlations. Begin with existing
validated radial and moment producers; add SOAP only as a separately identified
target/baseline if its additional information justifies its cost. Avoid silently
combining old nearest-80 labels with full-radius input definitions.

For exported z = E(X), use:

    L_geometry = sum_b alpha_b mean((D_b(z) - standardized G_b(X))²)

Each target family has train-fitted normalization and an explicit block weight,
so a large radial or TDA block cannot win simply by having more channels. The
initial training readouts should be linear, with controlled weight norms; matched
nonlinear probes remain evaluation tools. Otherwise a very sensitive decoder can
hide physical changes in tiny latent movements. No sample-specific coordinates,
atom features, or descriptors enter D outside z. A constant-code baseline must
fail to recover within-condition structural variation.

Fit observed→observed and relaxed→relaxed variants. The latter is a valid encoder
for applications that already use relaxed configurations; the former establishes
what the same training recipe retains in instantaneous observations. Their
reconstruction MSEs have different target domains and must not be compared as a
single ranking. Evaluate both on common downstream outcomes as well.

This is an intentionally clean ablation of an objective family already present
in the repository. Remove JEPA and BCR terms for these fits. Start without a rank
target or slowness loss; inspect collapse and physical retention instead of
automatically requesting 128 independently varying directions. Keep the same
backbone and export across new arms, and probe both the intermediate pooled code
and final export. A separate readout-architecture change can follow if that audit
still finds a large gap.

**What it tests:** whether training can improve accessible structure without
competing objectives. It does not claim to discover physics beyond the supplied
descriptors. A train-fitted descriptor PCA with the same output dimension is the
necessary compression control.

## 2. Relaxed structure as a fixed teacher

Let X be the observed patch and Q the same-frame, same-center relaxed patch.
The first teacher should simply be the fixed structural vector G(Q), not another
network that must be trained and tuned. Train:

    z = E_observed(X)
    L_student = L_present(G(X) from z) + lambda L_relaxed(G(Q) from z)

Both outputs come from the exported embedding. Select lambda on development
retention and independent downstream measurements; do not assume that forcing
the entire observed and relaxed embeddings to be equal is desirable. Retaining
observed geometry leaves room for physically meaningful thermal distortion.
Velocity/history can be added later with a matched observation budget.

If the relaxed-input model in method 1 provides useful information beyond the
fixed descriptor/PCA baseline, freeze that model and test learned-teacher
distillation as a second step. Its target representation and normalization must
stay fixed throughout student training. This differs from the existing
joint-gradient JEPA target pathway. A teacher fitted with held-out sources would
invalidate the comparison even if it never sees their event labels.

The useful experiment is whether the observed-input student closes part of the
gap to the relaxed-input reference on **the same physical targets and future
outcomes**. Same-domain relaxed reconstruction is an oracle-like observation
reference, not a fair observed-input competitor. If downstream use already
permits quenching, directly using the relaxed encoder may be preferable to
distillation.

Quenching is a physical transformation, not an exact nuisance symmetry. Full-cell
relaxation can depend on atoms outside the student's local observation. Hence
even a perfect local student need not recover Q deterministically; squared error
estimates its conditional mean. A residual gap is not automatically a training
failure. Keep this experiment separate from a context-radius sweep.

**What it tests:** whether existing relaxations offer a useful fixed training
signal for the instantaneous exported state. This is the most promising
cross-domain extension, but not a claim that hot-to-cold learning has never been
attempted here.

## 3. Teach physical relationships between embeddings

Good reconstruction does not ensure that nearby z values represent similar
environments. After method 1 establishes retention, add a small relational loss
directly to z. Use a declared distance between train-standardized descriptor
blocks as the teacher relation, with pairs sampled within temperature and
independently defined structural strata. For example:

    L_relation = mean Huber(d_z(i,j)/s_z - d_G(i,j)/s_G)
    L = L_geometry + gamma L_relation

Here s_G is a fixed training-reference distance scale, and s_z is a declared
running training-only scale treated as constant in differentiation, with its
behavior near collapse specified. Use fixed sampled pairs, not an uncontrolled
all-pairs batch loss; preserve absolute structural amplitudes through
L_geometry. This is a proposed adaptation of relational distillation, not a
literal replication of the CVPR paper.

The first comparison should be relaxed geometry versus the same relaxed model
plus this term. Relaxed data give an appropriate structural distance target
without first solving uncertain observed→relaxed mapping. Cross-domain distance
distillation is a later combination if both independent ideas pass.

A descriptor metric imposes a scientific preference. It cannot demonstrate
discovery merely by reproducing its own neighborhoods. Evaluate nearest-neighbor
agreement in **original embedding space** using withheld angular/topological
measurements and future outcomes. Match temperature and current coarse order so
liquid–crystal or temperature separation cannot dominate the result. Avoid
forcing adjacent atoms or successive frames to share a state, and do not use
cluster separation in UMAP as a selection metric.

**What it tests:** whether the exported distance becomes useful while information
retention is maintained. Stop if it only copies descriptor distances while losing
held-out information or providing no practical compression/inference benefit.

## Alternatives to keep bounded

**Short-lag kinetic learning.** The Qiu paper is relevant, but our
[earlier literature review](../mace_context_clusters_20260915/LITERATURE_REVIEW.md)
already records a negative linear-VAMP versus static-PCA comparison on future-law
retrieval. First reuse the [linear implementation](../../src/temporal_vamp/linear_vamp.py)
on fixed physical descriptors, within a declared condition and temporal population.
Compare to equal-dimensional PCA and uncompressed descriptors. Only useful
independent downstream gains would justify end-to-end MACE with a structural
anchor plus a [VAMP-style objective](https://www.nature.com/articles/s41467-017-02388-1).
This would train the native encoder; the cheap diagnostic is not a proposed
post-processing replacement for it. Raw 0.75 ps trajectories support candidate
lags such as 0.75, 3 and 9 ps where coverage allows. Sparse paired quench anchors
are not a dense relaxed trajectory. Do not transfer reduced-unit lags from the
glass paper or interpret pooled nonstationary quench data as equilibrium kinetics.

**Masked geometric prediction.** Predict fixed missing-neighborhood structural
statistics from one exported code of a genuinely masked graph. Rebuild edges and
geometric features after masking; hidden coordinates, patch centers and hidden
edges must not reach the decoder. An invariant code should predict invariant
targets unless an explicit orientation convention is supplied. In monatomic Al,
species prediction is trivial. Random atom deletion changes density and is not
an invariance augmentation. This is a plausible later pretext task, but more
implementation work and ambiguity than the first three methods. Include a
mask/condition-only baseline before interpreting a reconstruction gain.

**Further denoising, a larger model, or stronger whitening.** These do not directly
resolve the demonstrated objective mismatch. Keep historical BCR as an external
reference. A new BCR hybrid is unnecessary for the first screen requested here.

## Proposed first screen: four encoder fits, one seed

Use current verified paired relaxed/observed data and original trajectories. No
new simulation is needed for this screen. Freeze ancestry-level train, tuning
and development roles before extracting examples. The existing 45-root full-radius
audit supplies a small reproducible mechanism cohort; the expanded collection
supplies a later scaling cohort. The former has only 1,600 fitting examples, so
success there is a gate for scaling, not a strong generalization claim.

| Arm | Input and objective | Primary comparison |
| --- | --- | --- |
| A | Observed input, observed fixed geometry | Retention versus same initialized untrained encoder and matched descriptor PCA |
| B | Relaxed input, relaxed fixed geometry | Same-domain retention; common downstream outcome versus relaxed descriptors |
| C | Observed input, observed plus fixed relaxed geometry | A versus C on common present, relaxed and future targets |
| D | Relaxed input, relaxed geometry plus relational loss | B versus D on retention and physical/future neighborhood quality |

Use identical native MACE architecture, fresh initialization, atom support,
export dimension, optimizer budget and seed across arms. Full radius 8 Å is the
current mechanism-audit reference; maintain the halo required by the chosen
message-passing support contract, and smooth pooling/counts. Do not silently use
nearest-80 truncation. A common initial ceiling of 2,048 updates follows an
existing pilot budget, but is only an engineering screen: inspect learning curves
and underfitting before interpreting a null result. Do not promise a wall time
from the previous decoder/probe queue. No hardware benchmark is part of the
scientific loss or scheduled training stage.

Before neural training, fit condition-only, descriptor, descriptor-PCA and
random-MACE controls on exactly the same support and splits. Use one predeclared
moderate coefficient per added term for the first screen; a broad coefficient,
width and lag grid would defeat the purpose. A genuinely learned frozen teacher,
masked task and end-to-end kinetic arm are conditional follow-ups, not additional
mandatory fits.

## Selection and stopping rules

1. **Retention:** matched linear and residual nonlinear probes from exported and
   pooled features; radial/angular blocks and within-temperature errors reported
   separately. Keep at least one structural family, such as independently
   defined topology or higher-order local structure, entirely outside training
   targets. Train-fitted scales and block definitions remain fixed.
2. **Useful distance:** neighbor errors for those withheld measurements and future
   physical outcomes, compared with descriptor/PCA neighbors. Report density and
   current-order matched results, not only globally pooled distances.
3. **Dynamics:** frozen matched downstream predictors of physical evolution and
   sustained local onset. Compare NLL/Brier, AP and timing including misses;
   future frames define labels only. Readouts receive identical known conditions.
   Existing shooting branches can support an additional distributional assay,
   with every descendant of a parent kept in the same split.
4. **Robustness:** rotation/permutation, support-boundary and precision checks;
   response to true structural changes alongside temporal jumps. Float16 archived
   coordinates do not support arbitrarily small perturbation experiments.
5. **Decision:** keep the geometry-only arm if additions fail. Promote a variant
   only with improved held-out utility and an explicitly tolerated retention
   tradeoff. Declare that tolerance before running; do not recycle BCR's 5%
   denoising gate for these different tasks. Rank and smoothness are diagnostics.

The current q6<0.35 mask selected every sample in the BCR transfer assay. It must
not be presented as independent liquid-specific validation. Use validated
structure/phase labels, distinguishing disordered bulk, interface and crystal;
PTM-Other alone can contain defects and interfaces. Historical test cohorts have
already informed this proposal and must be called reused evaluation, not a fresh
confirmatory test. Paired root bootstrap intervals with one seed describe source
uncertainty conditional on that fit, not optimization-seed uncertainty.

The first useful outcome is an encoder that retains and organizes verifiable
structure at least as well as cheap alternatives. A persistent failure against
matched relaxed descriptors would argue for using those descriptors while
reconsidering the representation requirement, rather than escalating model size.
