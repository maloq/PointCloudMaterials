# Does spatial VICReg create an apparent interfacial shell?

Status: all nine 24-pass fits, all 63 checkpoint assays and six classical controls
are complete. [Findings and interpretation](RESULTS.md) ·
[Summary plots and tables](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/completed-review-v1/README.md).
The [archived-coordinate findings](COORDINATE_FINDINGS.md) remain a separate
descriptive reference. [Execution receipts](../../docs/spatial_vicreg_bias.md).
[Detached descriptor islands](PACMAP_ISLANDS.md) are audited against PTM,
individual descriptor families and a frozen scaling-sensitivity control.
The question is whether neighbor-view
alignment produces a smooth crystal-to-liquid representation that clustering
then partitions into apparent intermediate states, and how much structural or
predictive information those states contain beyond that construction.

## What the repository establishes

- [The paper](../../paper/main.tex), view construction, explicitly motivates
  spatial smoothness through overlapping, neighbor-centered views. The
  invariance loss acts on the projector output, not directly on the raw encoder.
- The paper's main views use 128 candidate points and 80 model points. Its Al
  extraction also has a grid-overlap parameter. Overlap between extracted
  training samples and overlap between a positive pair are different quantities.
  Neither proves that the positive-pair graph connects every atom in one epoch.
- [The current producer](../../src/training_methods/shared/vicreg.py),
  `_resolve_neighbor_flags` and `_augment`, shifts only when enabled, then crops
  around the shifted origin. [The shift helper](../../src/utils/pointcloud_ops.py)
  selects among configured neighbors; its optional relative-distance setting
  expands the candidate set. Measure actual distances rather than inferring
  them from a configuration name.
- The actual archived GeoFormerV2 epoch-34 checkpoint was read on 29 September:
  `vicreg_neighbor_view=False`, `factor_vae_enabled=True`, gamma 0.1,
  jitter 0.01, mirror probability 0.5, no rotation, no point dropout,
  `representation_source=vicreg_projector`, batch 16,384, epoch 34/global step
  3,965. Its identity is recorded in
  [campaign.json](../../configs/geoframe_evolution/campaign.json); the
  [reproduction recipe](../../configs/geoframe_evolution/epoch34.yaml) agrees.
  The current static loader returns one cached patch; with shifting disabled,
  the current VICReg module makes both views from the same nearest-80 crop.
  This verifies saved settings and current reproduction semantics, not every
  historical code revision. The picture is therefore not established evidence
  for direct alignment of different centers. Do not relabel that run as plain
  spatial-neighbor VICReg.
- The [completed MACE mechanisms study](../encoder_mechanisms_20260926/COMPLETED-RESULTS.md)
  changed observed/relaxed alignment. It did not isolate spatial-neighbor
  alignment and cannot settle this question.

## Mechanism and competing explanations

For fixed view outputs Y and symmetric positive-pair weights W, with
L = diag(W 1) - W,

    sum_ij W_ij ||Y_i - Y_j||² = 2 trace(Y^T L Y).

The invariance term penalizes variation across positive-pair edges. Variance and
covariance regularization oppose trivial collapse but do not certify a physical
meaning for the retained directions. This graph interpretation has a direct
precedent in [Balestriero & LeCun (2022), section 3](https://arxiv.org/html/2205.11508v3).
Their spectral results use specified objective/model assumptions; they are not
a proof that this nonlinear GeoFormer will learn a particular spatial field.
[HaoChen et al. (2021)](https://arxiv.org/abs/2106.04156) likewise analyze an
augmentation graph for a different, spectral contrastive objective.

Separate four explanations:

1. **Input overlap/coarse graining:** nearby patches share atoms. Even a frozen
   descriptor can interpolate as the fraction of crystal in a patch changes.
2. **Learned neighbor smoothing:** alignment suppresses differences between
   distinct centers, including meaningful differences across a boundary.
3. **Clustering a continuum:** K-means divides a smooth one-dimensional trend
   into colored shells without establishing distinct metastable states.
4. **Physical local variation:** ordering, defects or liquid motifs really vary
   near crystal and remain distinguishable beyond proximity and mixed support.

These explanations can coexist. A smooth interface descriptor can be useful
without defining a new liquid state. A deterministic local encoder in evaluation
mode does not diffuse information through a whole MD cell at inference: its
output depends on its actual input support. Identical local input must yield
identical output regardless of a remote crystal. Statistical correlations,
overlapping observations and training-set memorization are separate issues.

## Stage 0: audit views and frozen representations

Retain source/frame/center and atom identities for both views. Measure center
separation, intersection-over-union of atom IDs, shared-atom fraction, actual
support and union support, pair degrees and connected components. Do not equate
an atom-center graph with an exact graph of stochastic crops: different crops
at one center can have different outputs. Audit how frequently a positive pair
crosses an independently identified motif/phase boundary; these labels are
diagnostics only and do not choose positive pairs.

Evaluate initialization, archived exports, and both raw encoder z and once-applied
projector y. Use deterministic grouping and inference normalization; audit
GeoFormer's known frame-switch sensitivity separately from spatial smoothing.
The published epoch-34 image and paper-like neighbor training are separate
references with their true objective and input records.

## Stage 1: change the alignment relation while holding views fixed

Use GeoFormerV2, geometry only, a 128-dimensional encoder and an MLP projector.
Disable FactorVAE, JEPA, temporal views and physical reconstruction. Keep the
architecture, initial weights within seed, input preprocessing, optimizer,
schedule, presentations and variance/covariance calculations identical.

For each parent patch choose center A and a nearby center B using the audited
spatial sampler. Construct two independently perturbed versions of each crop:
A1, A2, B1, B2. Replay exactly these four tensors in all treatments. Keep the
variance/covariance terms on the same four fixed minibatch groups, averaging
their terms independently of how alignment pairs are assigned.

Define, with MSE averaged over examples and latent coordinates:

    L_same  = [MSE(y_A1, y_A2) + MSE(y_B1, y_B2)] / 2
    L_cross = [MSE(y_A1, y_B2) + MSE(y_B1, y_A2)] / 2
    loss(alpha) = 25 [(1-alpha) L_same + alpha L_cross] + 25 V + C

| Treatment | alpha | What differs |
| --- | ---: | --- |
| S0 | 0 | Alignment only between perturbations of the same local environment |
| S05 | 0.5 | Half the alignment weight joins neighboring environments |
| S1 | 1 | Alignment joins neighboring environments throughout |

This gives a dose response without changing the observed crop distribution or
overall alignment coefficient. Ordinary `neighbor_view=false` versus `true`
also changes which crops the model sees; keep that simpler reproduction as a
separate comparison if needed. The four-view estimator is an explicitly matched
mechanism experiment, not a bitwise reproduction of historical two-view training.

Use three paired seeds and 24 complete passes: **nine scientific fits**. Default
batch and microbatch are 256 parent examples; each contributes four views, so
record 1,024 view presentations per full batch and the actual execution splits.
Define an epoch by a fixed parent population. Record updates and presentations
as well as epochs. Save initialization and every epoch; full assays at completed
passes 0, 1, 4, 8, 12, 18 and 24, with passes 12 and 24 the primary endpoints.
Freeze one common label-free schedule before examining treatment outcomes.
No checkpoint selection by cluster appearance, physical labels or AP.

Use the source and anchor roles in [fixed Al64 v1](../../configs/fixed_cohort/al64_v1.json),
reviewed via [DATASETS.md](../../DATASETS.md). Structural gradients use train
ancestors only. The fixed cache has 80 candidates, so it cannot silently stand
in for a larger parent crop shifted and truncated to 80. A versioned view cache
must re-extract candidates from full cells at the frozen anchors, record the
parent-candidate policy and all support changes, and retain all evaluation rows.
Choose this extraction policy before fitting; use identical views across arms.
Record observed versus relaxed geometry explicitly. The first matched comparison
uses current observed Al geometry; transfer to the old relaxed static assay is
reported separately and is not an exact reproduction of that data distribution.

The encoder sees relative coordinates only: no absolute center position, atom
identity, material ID, temperature, simulation age, time, motion, or teacher
labels. No context halo or history unless explicitly added in a separate study.
Current physical descriptors/proximity are supplied only to declared diagnostic
readouts. A frozen forecast sees current z or y, with each added control input
recorded separately. Scientific fits use the existing online W&B project and
stable IDs; frozen diagnostics remain local. Execution is through Slurm.

## Stage 2: show what smoothing alone can produce

On the same evaluation cells, calculate an independent crystalline indicator
from PTM and average it over the actual encoder support. Also compute local
unaveraged order, existing neighbor-averaged order, and initialization features.
Build a deliberately simple spatial-diffusion control by repeatedly averaging
the crystal indicator or frozen features over a declared atom-neighbor graph.
Use a prespecified panel of diffusion steps; do not tune it to a held-out image.

Cluster these fields with the same fitting-only procedure and fixed K panel
(3, 6, 7, 10). If they produce similar shells, the visual arrangement is not
specific evidence for a learned intermediate state. This is a sufficiency
demonstration, not proof that it explains the trained encoder quantitatively.
The crystal-indicator control is explicitly privileged: it sees external
classical labels and potentially a larger support after diffusion. Report its
effective support and do not rank it as an equal-input learned encoder.

A secondary stress assay can join independently generated bulk liquid and
crystal without adding a specially prepared intermediate liquid population.
Such a synthetic interface has joining artifacts and lacks equilibrium
provenance; it is only a known-construction null, not physical ground truth.
The same-center versus neighbor contrast must also be measured on real held-out
interfaces before drawing a scientific conclusion.

## Stage 3: structural information and field measurements

**Native-coordinate transitions (requested follow-up).** Check individual z/y
coordinates as well as a bulk phase direction. Rank coordinates on one fitting
region, freeze their identity and sign, then examine other spatial regions and
snapshots. Plot both population medians with actual observation spread and
individual geometrically selected crystal-to-liquid transects. A smooth pooled
mean can hide abrupt local jumps, and absence of one smooth native coordinate
does not exclude a smooth linear combination. Coordinate identities can rotate
between independently trained encoders. The implemented descriptive assay and
its exact distance-coordinate definition are recorded in
[the metric contract](../../docs/metrics/spatial_vicreg_coordinates.md) and
[execution guide](../../docs/spatial_vicreg_bias.md). It re-extracts the actual
archived epoch-34 model; it is not the proposed S0/S05/S1 experiment.

**Pair contraction.** On fixed physical neighbor pairs, compare squared latent
distance divided by uniform training-population variance, separately within crystal,
within liquid and across independent boundaries. Keep the reference population
and normalizer definition fixed; report raw variance/effective rank alongside
ratios. Stronger cross-boundary contraction with alpha supports smoothing, but
low rank or loss of bulk separation is a degeneration, not a meaningful shell.

**Shell location and width.** Fit a bulk crystal-versus-liquid linear score on
training reference data for each export. On identical held-out centers, plot
that continuous score against signed distance to the independent interface and
alongside actual crystalline fraction in the input. Normalize by training bulk
endpoints and report 10–90% transition width where both crossings are supported;
otherwise report undefined, not an extrapolated width. The score is a diagnostic
phase readout, not a training target. Compare physical order profiles and
unsmoothed/explicitly smoothed controls on the same cells. Changing shell width
with alpha while physical profiles remain fixed is evidence of an induced
representation effect. Dependence on input radius alone is also expected from
legitimate coarse graining and is not decisive.

**Liquid structure beyond mixed support.** Evaluate both all noncrystalline
centers and the strict subset with no independently crystalline atom anywhere
in the actual inference support. Report population counts, uncertainty and
threshold sensitivity. Within matched proximity, support crystal fraction and
density, measure incremental held-out readout of unaveraged q4/q6/normalized
w4/w6, bond topology, template mismatch and independent fault indicators from
z or y. Retain averaged-order diagnostics, but do not make a spatially averaged
reference the sole judge of spatial smoothing. Add embeddings to the same
controls, with matched readout capacity and redundant-feature controls. Shared
geometric origin makes these operational references, not definitive liquid-state
labels; residual gains establish information beyond the specified controls only.

**Future information beyond an approaching front.** Compare proper held-out
predictive scores from current physical descriptors/proximity versus the same
inputs plus z or y, with frozen encoders and likelihood-based readout selection.
Use 3/6 ps outcomes and calibration; AP remains diagnostic. Keep a separate
crystal-clear cohort and distinguish existing-front arrival from new regional
establishment. Count independent eligible birth events before fitting that
subtask. Existing mechanism results had inadequate strict birth coverage; newly
registered collections require their own ancestry and eligibility audit.
Predicting proximity or front arrival does not by itself validate a precursor.
If a candidate survives the static controls, future persistence or repeated
trajectory/shooting outcomes are the stronger follow-up for physical relevance.

All primary distances/readouts use native z/y, not UMAP or t-SNE coordinates.
Fit normalization, probes and cluster centers on training ancestors; selection
uses validation only, calibration is separate, test remains fixed. Add dense
spatial evaluation windows only as a declared supplemental track, preserving
the all64/legacy16 rows. Bootstrap sources/ancestries, report paired seed effects,
and never count neighboring atoms or checkpoints as independent replicates.
The old static images remain descriptive because their training ancestry is not
an independent test. Ta/Zr transfer requires its own known provenance and
material-specific structural references; do not transfer Al class meanings.

## Interpretation and staged follow-up

| Observation | Warranted inference |
| --- | --- |
| S1 broadens the shell and contracts physical boundary differences relative to S0, with a consistent S05 trend | Neighbor alignment causally changes the field under this matched protocol |
| A frozen support-average/diffusion field gives similar colored shells | Shell appearance alone is insufficient evidence of a separate state |
| Shells exist at initialization or S0 too | Neighbor alignment is not necessary; quantify shared-input and clustering effects |
| Effects concentrate in y while z retains physical information | Projector geometry, rather than complete encoder information loss, explains part of the observation |
| Matched/strict-clear liquid retains independent structural and future information across held-out sources | Evidence against the strong claim that the representation is only proximity/mixed-support smoothing |
| No measurable residual gain | No gain with these controls/readouts and sample size; not proof of information-theoretic absence |

Only after this comparison, vary neighbor separation at fixed inference support
and alignment strength at fixed pair sampler. Measure atom overlap rather than
claiming separation and overlap vary independently. A useful additional
augmentation is two same-center subsamples with reduced shared atom identity;
it changes geometric fidelity and needs matching/sensitivity checks, so it is
not the first clean causal contrast. Random global positive pairs change the
semantic task and can destroy the representation; they are a failure control,
not the main alternative.

Proposed results: `output/spatial_vicreg_bias/<run>/analyses/`, with grouped
plots/tables and saved predictions, input contracts and provenance. New numerical
metrics need their own reviewed `docs/metrics/` contract and frozen definitions
at export. The implemented [training recipe](../../configs/spatial_vicreg_bias/al64_20260929.json)
and [metric contract](../../docs/metrics/spatial_vicreg_bias.md) freeze the concrete
choices. The initial study runs the three stages above; synthetic joining, new
future-prediction fits, template/fault readouts requiring additional references,
and Ta/Zr transfer remain follow-ups. The numerical producer uses a core/liquid
distance contrast, not reconstructed interface distance. PTM detection-count
sensitivity is reported; a new full-cell PTM RMSD-threshold sweep is not run.


## Concrete cluster-to-structure assay

From the exact consumed 80 atoms, compute 412 independent classical descriptors:
258 alpha-complex TDA, 40 bond-order, 45 CNA and 69 geometry summaries. Fit cluster
centers only on uniform training sources and freeze assignments on all64, legacy16
and the all-phase uniform supplemental track. Membership predicts descriptor
means, and a fixed Gaussian-mean ridge readout measures what membership adds
beyond current crystal proximity, support crystal fraction and density. Compare
continuous native embeddings and nuisance-stratified permuted cluster labels.
Report all phases, noncrystalline centers and strict-clear support separately;
the annotations never enter encoder optimization or remove interface inputs.

Seeds 17/29/43 share initialization, sample order, four tensors and regularizers
across alpha. The mechanism comparison uses 256-parent batches (1,024 views),
AdamW 1e−4, one-pass warmup/cosine decay, 0.01 Å jitter and 0.5 x-reflection; this
is not an exact replay of the archived 16,384-batch FactorVAE experiment. Positive
conditional gains establish access to these descriptors beyond these controls,
not unique physical phases, metastability, or precursor prediction. Repeated
epochs and correlated atoms are not independent replications.

A positive control-adjusted ridge gain can arise because cluster one-hot features
provide a nonlinear basis for variables the controls already observe. It is an
improvement over this fitted readout, not an estimate or proof of conditional
mutual information. Apply the identical readouts to the scalar phase/diffusion
controls; compare those capacity effects before attributing an encoder gain to
additional physical structure. All-phase/interface results remain central;
strict-clear results address a separate, narrower liquid-information question.
