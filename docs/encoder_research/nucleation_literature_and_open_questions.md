# Nucleation and prestructured liquid: literature, open questions and evidence ledger

Created and literature checked: **26 September 2026**. Status: research synthesis
and proposed validation; no new experiment or changed label definition.

Use this document when interpreting new birth-harvest results, choosing encoder
comparisons, or assessing a possible publication claim. The central question is:

> What information in a local atomic observation and its surroundings predicts
> a new crystal's establishment, beyond recognizable crystal order and proximity
> to an existing crystal?

There is substantial prior work on order parameters, precursor regions and
learned structural scores. A contribution must identify what additional physical
information we measure, under what observation constraints, and how we rule out
recognition, front arrival and label artifacts. **The hypotheses below are not
established novelty claims.**

Related references:

- [Current detector: implementation audit and ambiguities](nuclei_and_prestructured_liquid_20260926.md)
- [Spatial warning literature: closest interface and GNN precedents](spatial_warning_literature_20260926.md)
- [Broader representation-learning literature](representation_literature_20260926.md)
- [Completed origin audit](../../experiments/crystallization_origin_20260925/RESULTS.md)
- [Regional birth-harvest proposal](../../experiments/crystallization_origin_20260925/HARVEST_PROPOSAL.md)

## 1. Keep the physical claims separate

| Claim | Required evidence | Insufficient evidence |
| --- | --- | --- |
| An atom is locally crystal-like | Declared structural classifier and calibration | A future event alone |
| A liquid region has correlated order | Order/coherence relative to reference liquid, including spatial extent | High scalar q6 alone, or PTM's unclassified category |
| An embryo establishes | Connected cluster, ancestry and a declared persistence rule | One crystalline frame |
| An embryo is dynamically critical | Growth/dissolution commitment under defined basins and ensemble | Crossing our 64-atom threshold |
| A feature is an advance predictor | Prospective performance on held-out sources at declared lead times | A retrospective average aligned to successful births |
| A feature participates causally | Appropriate intervention or other mechanistic evidence | Predictor accuracy or feature attribution alone |

Do not assume these are consecutive stages of a universal two-step mechanism.
A coherently ordered shell, a transient embryo and a distinct metastable liquid
phase are different claims. Spatial association with a growing crystal also
does not determine whether liquid order preceded or was induced by that crystal.

## 2. Primary literature and what each source establishes

The following are bounded summaries of the linked primary sources. Differences
between systems and measurement protocols are part of the evidence, not reasons
to select only papers supporting a precursor hypothesis.

### R01 — Bond-orientational order

**Steinhardt, Nelson and Ronchetti (1983), “Bond-orientational order in liquids
and glasses.”** [Physical Review B](https://doi.org/10.1103/PhysRevB.28.784).

Rotational invariants of local bond directions and their correlations describe
order beyond radial structure. The study includes supercooled Lennard-Jones
liquid and glass models. This is foundational prior art for using orientational
order; it does not make every highly ordered liquid environment a productive
crystallization precursor. Our learned representation must be compared with
such structural information, not only with density or a binary PTM flag.

### R02 — Nucleation barriers and dynamical crossing

**ten Wolde, Ruiz-Montero and Frenkel (1996), “Numerical calculation of the rate
of crystal nucleation in a Lennard-Jones system at moderate undercooling.”**
[Authors' institutional record](https://amolf.artudis.com/pub/4667/).

Combining barrier sampling with dynamical crossing calculations supplies a
physical nucleation-rate analysis. This is a methodological precedent for
separating structural cluster detection from transition kinetics. A catalogue
of persistent clusters alone does not estimate a critical nucleus or a
nucleation rate with equivalent physical validation. Numerical parameters from
Lennard-Jones simulations are not transferable Al thresholds.

### R03 — Averaged bond order and coherent neighbors

**Lechner and Dellago (2008), “Accurate determination of crystal structures
based on averaged local bond order parameters.”**
[Paper](https://arxiv.org/abs/0806.3345),
[full text](https://arxiv.org/pdf/0806.3345).

Averaging complex bond-order coefficients over an atom and its neighbors before
forming invariants improves structural discrimination. The paper also discusses
solid-like neighbor connections based on correlated bond order. These are strong
controls for our encoders. Local and averaged descriptors observe different
spatial supports; a fair comparison must include the neighbors-of-neighbors
needed to compute them. No single coherence threshold is universal.

### R04 — Robust local crystal templates

**Larsen, Schmidt and Schiøtz (2016), “Robust Structural Identification via
Polyhedral Template Matching.”** [Paper](https://arxiv.org/abs/1603.05143).

PTM identifies crystalline environments robustly under thermal disorder and can
recover lattice orientation. It is a structural classifier, not a growth
probability. Our use is well motivated, but the RMSD cutoff requires calibration
for small embryos and interfaces. The [OVITO documentation](https://www.ovito.org/manual/reference/pipelines/modifiers/polyhedral_template_matching.html)
describes 0.1 as useful for defect identification in crystalline solids; that is
not a universal nucleation calibration.

### R05 — A complementary radial fingerprint

**Piaggi and Parrinello (2017), “Entropy based fingerprint for local crystalline
order.”** [Paper](https://arxiv.org/abs/1707.09892).

An approximate local entropy fingerprint distinguishes liquid-like and
solid-like environments without specifying a particular crystal template.
Combining it with local enthalpy improves structural discrimination in the
paper. For our geometry-only comparisons, the radial fingerprint is the relevant
candidate control; adding enthalpy would change the input contract. Template
independence does not make its thresholds or neighborhood choices arbitrary-free.

### R06 — Prestructured surroundings can improve a reaction coordinate

**Díaz Leines and Rogal (2018), “Maximum Likelihood Analysis of Reaction
Coordinates during Solidification in Ni.”**
[Paper](https://arxiv.org/abs/1810.04782).

Path-ensemble analysis finds that the prestructured region surrounding a
crystalline cluster improves the nucleation reaction-coordinate description
relative to the crystalline core alone. This is close prior art for our context
hypothesis. It does not establish that Al has the same pathway, that a shell is
detectable many picoseconds before establishment, or that a learned encoder
adds information beyond conventional order descriptors.

### R07 — Cluster size can be an incomplete reaction coordinate

**Liang et al. (2020), “Identification of a Multi-Dimensional Reaction
Coordinate for Crystal Nucleation in Ni3Al.”**
[Paper](https://arxiv.org/abs/2004.01473).

The study identifies size, crystallinity and chemical order as relevant to
nucleation in this alloy. It motivates checking growth versus dissolution at
matched cluster sizes. The alloy-specific chemical-order result does not justify
adding species or material channels to our geometry-only encoder. Nor does it
provide a critical cluster size for our monatomic Al potential.

### R08 — Topological metal analysis and alternative pathways

**Becker, Devijver, Molinier and Jakse (2022), “Unsupervised topological
learning approach of crystal nucleation.”**
[Published paper](https://doi.org/10.1038/s41598-022-06963-5),
[author version](https://arxiv.org/abs/2109.06797).

Persistent-homology descriptions and unsupervised classification examine Al,
Ta and Mg, using inherent structures. The work connects emergence with low
fivefold symmetry and reports coupled positional/orientational ordering. This
is close prior art for learning liquid environments and identifying nuclei.
Relaxation is part of its observation procedure; an apparent precursor after
minimization needs a separate interpretation from one in instantaneous MD.
Structural classes alone do not establish prospective predictive skill.

### R09 — Intervention on liquid preordering

**Hu and Tanaka (2022), “Revealing the role of liquid preordering in
crystallisation of supercooled liquids.”**
[Paper](https://doi.org/10.1038/s41467-022-32241-z),
[accessible article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9352720/).

Suppressing crystal-like liquid order through a biasing strategy strongly
reduces crystallization rates in the studied systems, including NiAl. This
provides mechanistic evidence beyond correlation. It is not a direct validation
of our Al features: such interventions modify the dynamics, and their effects
require system-specific interpretation. Attention maps or feature importance
from our predictor would be much weaker evidence of causality.

### R10 — Continuous learned order already exists in Al

**Chapman et al. (2023), “Quantifying disorder one atom at a time using an
interpretable graph neural network paradigm.”**
[SODAS paper](https://doi.org/10.1038/s41467-023-39755-0),
[accessible article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10328988/).

A GNN supplies a continuous atomic-order score, including Al solid–liquid
interfaces. Learning an order field or detecting an interface is therefore
not sufficient novelty. The key additional evidence for our question would be
prospective event information under controlled visibility and observation
support. Their thermally calibrated score is not automatically a calibrated
event probability or the same training treatment as our encoders.

### R11 — A negative precursor result must remain in view

**de Jager, Smallenburg and Filion (2023), “In search of a precursor for crystal
nucleation of hard and charged colloids.”**
[Paper](https://arxiv.org/abs/2306.05886),
[published version](https://doi.org/10.1063/5.0161356).

Conventional and unsupervised structural analyses of spontaneous nucleation
find no separately detectable advance precursor in the studied systems:
crystalline signatures arise with nucleation. This is not evidence that our Al
data lack precursor information. It does mean that “a better representation
must find a precursor” is an invalid premise. A carefully bounded negative
result can be informative, provided sampling, sensitivity and statistical
power are reported.

### R12 — A recent, directly relevant Al comparison

**Tipeev and Zanotto (2025), “Crystal nucleation and growth dynamics of aluminum
via quantum-accurate MD simulations.”**
[Acta Materialia](https://doi.org/10.1016/j.actamat.2025.121245).

The study uses a liquid-trained ML potential, a pair-entropy fingerprint for
emergent clusters, and spontaneous/seeded Al crystallization to examine nucleation
and growth kinetics. It is a particularly relevant precedent for an alternative
Al cluster description. The generating potential and preparation differ from
ours; its temperature range, criticality and kinetic conclusions cannot be
transferred by material name alone. Review the full methods before adopting
any numerical settings.

## 3. What our existing evidence actually says

The [completed origin audit](../../experiments/crystallization_origin_20260925/RESULTS.md)
attributes **98.5% of positive fixed-Al64 onset windows at both 3 and 6 ps** to
arrival of an existing crystal. Therefore, historical onset prediction does not
by itself demonstrate forecasting of new isolated nuclei.

The same audit finds **206 isolated establishment candidates in 90 training
sources**, but zero regional-birth training windows on the original fixed
center-liquid grid. This is an observation/sampling mismatch, not evidence of
zero births or of absent precursors. The later regional harvest has a distinct
target and eligibility definition; see its
[proposal](../../experiments/crystallization_origin_20260925/HARVEST_PROPOSAL.md)
and [audit guide](../nucleus_harvest.md).

The [method audit](nuclei_and_prestructured_liquid_20260926.md) records our PTM,
connectivity and lineage implementation. Its existing training-source counts
are 253, 206 and 167 for minimum sizes 32, 64 and 128 atoms with 1.5 ps persistence;
64 atoms with 3 ps persistence gives 201. These are definition-sensitive event
counts, not four independent measurements of a true physical nucleus count.

Current birth-harvest descriptors include q4/q6, normalized w4/w6, averaged q6,
mean orientational coherence, coherent-neighbor counts and density. They are
sampled diagnostics, not a validated map of precursor shells. In this pathway
we do not yet export averaged q4 or use PTM orientation/RMSD to resolve those
regions. Older heuristic ordered-liquid classes must not be mistaken for
physical validation of the birth detector.

## 4. Candidate contributions and how to disprove them

These are our proposed research questions, inferred from the comparison above.
The distinction between birth and growth, the existence of bond order, a
learned atomic score, and a larger-context neural network are not themselves new.

| ID / possible contribution | Closest prior work | Decisive comparison | Evidence against the claim |
| --- | --- | --- | --- |
| **H1: Conditional precursor information** — surrounding liquid adds advance information about regional establishment | R06, R08–R11 | Core descriptors versus core + surrounding descriptors versus learned context; matched origins, support and likelihood heads; naturally crystal-free subgroup | Gain disappears after matching present order/proximity, or occurs only when visible crystal enters the input |
| **H2: Spatial organization matters beyond order amount** — arrangement and alignment distinguish productive from unproductive order | R01, R03, R06 | Scalar order/count summaries versus orientation-correlation/radial profiles versus shared-patch relational predictors | Conventional organization descriptors explain the gain; the learned architecture adds no detectable predictive information |
| **H3: Causal history distinguishes persistent precursors from fluctuations** | R06's path analysis and the broader representation review | Snapshot, real history and separately trained repeated-anchor control with matched capacity/support | Gain vanishes beyond shared-window overlap, under equal budgets, or after retaining the same current structure |
| **H4: Forecast conclusions survive defensible event definitions** | R02–R05, R12 | Matched events and prediction conclusions across PTM, bond coherence and pair-entropy descriptions, cadence and precision | Model ordering or claimed lead time depends on one arbitrary cutoff; disagreements are as large as the claimed effect |
| **H5: Birth and arrival have different observable information requirements** | R06/R07 versus the interface studies in the spatial review | Separately declared regional-establishment and existing-front outcomes, assessed with the same observation families | Apparent birth skill is explained by front visibility, unresolved ancestry, or unequal populations/support |

H1–H3 are hypotheses about physical information. H4 is initially a necessary
validation requirement; routine threshold sweeps alone are weak novelty. H5
could support a useful benchmark or mechanism-specific finding, but the physical
distinction is established. Before claiming priority, search forward citations
of the closest papers for the exact successful contribution.

## 5. Comparisons needed before trusting an apparent precursor

### A. Freeze the event, population and observation

Define region, lineage rule, establishment threshold, confirmation span,
eligibility and censoring before fitting. Distinguish the first size crossing,
its later confirmation and any future estimate of commitment. Future frames
may define labels but may not enter eligibility decisions or encoder inputs.
Allow a small ordered embryo at the origin if the declared regional risk rule
allows it; requiring an entirely liquid center can remove precisely the states
we want to study.

For birth versus arrival, specify whether outcomes may both occur or whether
the task concerns the first of competing events. Do not treat front capture as
an independent censoring event without justification. Report unresolved origin
and incomplete follow-up explicitly, rather than converting either to a negative.

The full periodic cell may establish the reference event. A local predictor
must see only its declared region, graph halo and history. Audit all input atoms
across every contextual patch and frame. A naturally crystal-free observation
is a useful subgroup under stated detectors, not proof of absence of all order.
Deleting crystalline atoms to make that subgroup would alter the observation.

### B. Build strong, matched structural controls

Compare radial geometry/density; local and neighbor-averaged q4/q6; w4/w6 and
fivefold motifs; coherent-bond counts; PTM fraction/RMSD/orientation; and the
geometry-derived pair-entropy candidate. Keep crystal composition and orientation
explicit in diagnostics: an HCP patch in Al may be a stacking fault, and a
twinned nucleus need not be two births.

Use identical outer support and comparable readout capacity when comparing
these descriptors with MACE. Count the extra support of neighbor averaging and
message passing. Calibrate structural thresholds on training-side references,
independently of forecast AP. A PTM-derived feature predicting a PTM-derived
label is a recognition control, not independent physical validation.

Use native and relaxed observations as explicitly separate treatments when
investigating relaxation. Relaxation may reveal hidden order or change the
structure; original-MD outcomes remain the reference. No physical-reconstruction
encoder pretraining is implied.

### C. Measure additional predictive information

For the same outcome and held-out population, compare the baseline descriptors
with descriptors plus the learned embedding using matched predictor families.
A useful report is

\[
\Delta\mathrm{NLL}
=\mathrm{NLL}(Y\mid d_{\mathrm{baseline}})
-\mathrm{NLL}(Y\mid d_{\mathrm{baseline}},z).
\]

Positive values mean better held-out log score from adding the embedding.
This finite-model comparison is **not automatically a conditional mutual
information estimate**: limited capacity, optimization and calibration matter.
Evaluate simple and stronger heads and, where feasible, the raw history/context
alongside the embedding to audit information lost in compression.

Report 3 ps as primary and 6 ps as secondary, using proper predictive scores,
calibration and source/ancestry-level uncertainty. AP remains diagnostic.
Event-centered enriched samples can support discovery or fitting, but probability
and false-alarm claims require representative sampling or declared sampling
corrections. Multiple windows from one nucleus are not independent births.

Match or stratify by current order and available proximity information in the
analysis; keep full-cell distance out of local predictor inputs. Retain both
population-wide and matched comparisons because conditioning can remove part
of the physical signal as well as a confounder. No temperature, simulation age,
absolute time, species/material channels or other prohibited covariates are
introduced. Horizons organize targets, and conditions may remain audit metadata.

### D. Quantify ambiguity before claiming lead time or mechanism

Match events between definitions and report unmatched births, atom-membership
agreement, timing shifts, splits/merges and near-interface disagreements.
Count agreement alone is insufficient. Keep primary results on each definition's
full eligible population; a stable-event intersection is a secondary diagnostic
because it preferentially retains easier events. An ensemble of label rules
measures rule disagreement, not a calibrated probability of physical truth.

Compare native cadence with downsampling at the same physical persistence span;
1.5 ps requires 11 observations at 0.15 ps, versus three at 0.75 ps. Compare paired
coordinate precisions and perturbation sensitivity. Normalize coordinate-noise
RMS by a declared mean local neighbor distance, and inspect label response as
well as embedding response. Preserve historical metric definitions.

Later, a representative small commitment study could separate growth from
dissolution. Define basins, ensemble, follow-up and how external-front capture
is handled. Identical deterministic phase-space restarts are not independent
trials; resampling velocities defines a position-conditioned experiment, which
differs from forecasting given observed velocities/history. A well-predicted
finite-horizon event is not automatically a committor.

## 6. Figures that would make the claims reviewable

| Figure | What it should show | Main safeguard |
| --- | --- | --- |
| Periodic core/shell/lineage snapshots | PTM motifs, continuous coherence, candidate shell and established neighbors | Shared color scales; examples include failures and uncertainty |
| Birth-aligned trajectories | Core size, shell order, competing motifs and prediction scores | Include failed embryos and controls; retrospective alignment is labeled |
| Order-space distributions | q4/q6, averaged order, entropy and growth/remelting outcomes | Show the liquid population, not only successful nuclei |
| Definition-agreement plot | Matched-event timing shifts and unmatched events | State cadence, thresholds and denominator |
| Conditional predictive-gain curves | Held-out log-score gain versus support radius and forecast horizon | Equal observations/readout budget; source-level intervals |
| Visibility and calibration panels | Risk calibration and false alarms with/without visible crystal | Natural subgroups; no atom deletion or future-conditioned inputs |

## 7. Revisit checklist and claim ledger

Before promoting a result into a conclusion:

- [ ] Exact claim, closest prior paper and material/potential differences recorded.
- [ ] Physical event and operational label distinguished; uncertainty retained.
- [ ] Current-input eligibility and future label confirmation audited separately.
- [ ] Complete spatial/history support and crystal visibility recorded.
- [ ] Conventional descriptors and learned features compared on matched inputs.
- [ ] Failed embryos, representative controls and censored records included.
- [ ] Source splits and shared preparation ancestry preserved.
- [ ] Proper-score selection, calibration and predictive uncertainty reported.
- [ ] Event identity/timing sensitivity and coordinate/cadence effects reviewed.
- [ ] Alternative explanations and the result that would falsify the claim stated.
- [ ] Forward/backward citation search refreshed for the precise proposed novelty.

Update this table with links to frozen result artifacts; do not replace an
earlier interpretation without retaining its dated record.

| Claim ID | Status on 2026-09-26 | Evidence still required | Result / decision record |
| --- | --- | --- | --- |
| H1 | Open; current onset cohort mainly measures arrival | Representative regional forecast evaluation and visibility controls | Not yet available |
| H2 | Open; context predictors exist, but this physical interpretation is unvalidated | Matched order-organization baselines | Not yet available |
| H3 | Open for isolated establishment | Matched snapshot/history/repeated-anchor study on the new target | Earlier mixed-onset history results do not resolve this claim |
| H4 | Size/persistence count sensitivity demonstrated | Event matching, complementary classifiers, cadence/precision comparison | [Existing audit](nuclei_and_prestructured_liquid_20260926.md) |
| H5 | Target mixture demonstrated | Separate predictive comparisons with defensible origin attribution | [Origin results](../../experiments/crystallization_origin_20260925/RESULTS.md) |

## 8. Search scope and remaining reading

This is a targeted review, not a systematic review or proof of priority. Searches
covered bond-orientational order, PTM, prestructured liquid, negative precursor
findings, metallic nucleation reaction coordinates, learned Al order fields,
and recent Al nucleation simulations. Exact-title searches checked primary
papers and author/institutional records. The linked spatial and representation
reviews extend coverage to interface kinetics and predictive representations.

Evidence depth: R03's full paper was inspected for the descriptor construction;
R04 is supplemented by implementation documentation. For other entries,
author abstracts and accessible publisher/article passages support the limited
claims recorded here. Numerical protocols and reproductions have not been
independently validated. In particular, R12's full methods and the cited
commitment/intervention protocols require deeper reading before implementation.

Still needed before a novelty claim: forward citations on core/shell reaction
coordinates and negative precursor studies; work on lineage-aware local event
benchmarks and competing outcomes; sensitivity of nucleus identity to structural
classifiers; and prospective learned predictors with explicit crystal-visibility
controls. Record contrary or overlapping work here as it is found.
