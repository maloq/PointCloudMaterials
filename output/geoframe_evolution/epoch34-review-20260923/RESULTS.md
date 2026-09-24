# GeoFrame: interfaces improve while the projector loses liquid detail

**Completed:** a fresh 35-pass reproduction of the requested VICReg + grouped
FactorVAE GeoFrameV2 recipe, numerical evaluation at every epoch, the random
initialization, and the archived epoch-34 reference. Every checkpoint is retained.
This is one training seed, with the original batch size, static sample cache and
160-epoch learning-rate clock. It reaches 2,135 model updates and 3,965 Lightning
optimizer steps, matching the archived epoch-34 update counts.

[Training curves and all UMAP/spatial plots](../epoch34-reproduction-20260923/index.html)
· [Checkpoint metrics](../epoch34-reproduction-20260923/tables/checkpoint-summary.csv)
· [Literature and protocol](../../../experiments/geoframe_evolution_20260923/README.md)
· [Ta/Zr dense candidate gallery](../structured-liquid-regions-20260923/README.md)

## A correction to the original comparison

The archived epoch-34 model used **VICReg**, whereas the archived epoch-159 model
used **VISReg** (lambda 0.4, 4096 projections). Both used FactorVAE, but they were
not early and late checkpoints of one unchanged objective. Their pictures alone
cannot establish deterioration with training duration. The fresh trajectory here
holds the objective fixed. It reproduces the recipe on current software, not a
bitwise historical execution or the original grid-centered figure sampling.

## What the new trajectory tells us

The table reports held-out spatial-region R² of six continuous bond-order
descriptors within noncrystalline environments, averaged across three snapshots
per material. Epoch index 11 means 12 completed passes; index 34 means 35 passes.

| Material | Encoder, 12 passes | Encoder, 35 passes | Projector, 12 passes | Projector, 35 passes |
|---|---:|---:|---:|---:|
| Al | 0.180 | 0.206 | 0.191 | 0.159 |
| Ta | 0.237 | 0.287 | 0.219 | 0.176 |
| Zr | 0.173 | 0.205 | 0.175 | 0.141 |

**The encoder improves on this readout while the projector worsens for all three
materials.** Thus the evidence is more specific than “the whole encoder loses
all nuance.” The representation used for the original epoch-34 picture was the
projector. Treating its output as interchangeable with the encoder hides this
difference. R² measures this declared linear-readout task; it is not a complete
measure of information in either representation.

Meanwhile, Al projector average precision improves from **0.550 to 0.605** for
the mixed-boundary proxy and from **0.662 to 0.768** for planar faults. The final
raw encoder reaches **0.875** for planar faults. This is the behavior the assay
was designed to expose: improved boundary/defect resolution can coexist with
reduced liquid descriptor fidelity. K=7 cluster/context matrices show whether
that information is also organized into separate unsupervised groups:

- [Archived epoch 34](plots/archived-epoch34-cluster-context.png)
- [Fresh run after 12 passes](plots/epoch-011-cluster-context.png)
- [Fresh run after 35 passes](plots/epoch-034-cluster-context.png)

The final Al 177 ps contingency is especially clear: one embedding cluster
contains approximately **98% of unclassified-liquid references, 98% of five-fold
liquid proxies, and 89% of ordered-liquid candidates**. Another cluster captures
64% of planar-fault references. Thus accessible fault information and poor
unsupervised separation of different liquid environments can occur together.
Those liquid categories are declared descriptor-based proxies, not proven phases.

Al projector liquid participation rank changes from **2.41 to 2.06**, while the
encoder changes from **6.64 to 7.29**. This is concentration of variance, not a
claim that only two exact dimensions or two physical states remain. A global
variance/covariance loss does not guarantee variance within each liquid regime:

`Var(z) = E[Var(z | material, phase)] + Var(E[z | material, phase])`.

Between-material or between-phase variation can satisfy part of the global
objective while local distinctions become weak. This is a plausible explanation,
not a causal attribution to VICReg or FactorVAE: separating their effects needs
a matched loss ablation.

The proposed boundary-aware spatial AUROC stays near chance (roughly 0.49–0.52
at the final endpoint). Attractive large-scale domains therefore do not establish
this finer local-order coherence. It is a new proxy-sensitive diagnostic, with
shuffled and constant-embedding controls; it does not replace inspection of
continuous fields or establish that all liquid subtypes should be discrete.

The small-displacement assay still detects sensitivity: final Al 95th-percentile
changes under per-coordinate noise of 0.0001 Å are about **6.7% of population
pair RMS for the encoder** and **2.7% for the projector**. The projector becomes
smoother along training while losing some liquid readout fidelity. GeoFrame's
discrete grouping/frame choices are a separate architectural issue; smoothing
alone is not a sufficient objective. Perturbations are not MD time.

## Interface and precursor reference labels

We compute full-snapshot PTM with FCC/HCP/BCC/ICO templates, template RMSD,
explicit Al stacking-fault/twin labels, local and averaged bond order, five-fold
order proxies, and local crystalline-neighbor fractions. Readouts use separated
spatial halves with a gap larger than two input radii. This is a **transductive**
encoder assay because the original recipe trains on these static snapshots.

The mixed-boundary class is an operational neighborhood proxy, not a unique
solid–liquid dividing surface. It can include grain boundaries and isolated
crystalline motifs. The Al planar-fault label is more specific; HCP Zr is not
automatically an Al-style defect. Neither the pictured colors nor seven learned
clusters are classical ground truth.

Some classes, notably the conservative non-template crystal-interior category,
have no held-out examples; their scores are explicitly undefined. Fine fault
subtypes also need sufficient examples. This experiment does not validate a
complete defect taxonomy just because the aggregated planar-fault score is high.
In these Al samples, most identified planar-fault atoms are coherent twins. The
specific twin readout reaches AP 0.678 at 174 ps and 0.709 at 177 ps for the final
projector; intrinsic and multilayer fault subtypes lack enough examples for a
separate reliable score. This does not identify the original image's colors
without matching their original atom-level assignments.

The dense Ta/Zr analysis finds connected high-order liquid regions outside
established crystal. It preserves atom coordinates, order values, motif
affinities, component membership, nearest-crystal distance and truncation flags.
For example, Ta region 4 contains a complete 17-atom component at least 25.7 Å
from a PTM crystal; Zr region 4 contains a complete 16-atom component at least
23.0 Å away. These are candidates to track, not demonstrated nuclei. The largest
displayed Zr component has 109 atoms but reaches the inspection boundary, so its
reported size is incomplete. Do not infer a Ta/Zr nucleation-rate difference
from deliberately selected high-order examples.

The Ta/Zr distinction in [Hu & Tanaka (2022)](https://www.nature.com/articles/s41467-022-32241-z)
motivates distinguishing crystal-compatible ordering from competing five-fold
order. [Becker et al.'s Ta/Al/Mg study](https://www.nature.com/articles/s41598-022-06963-5)
and [Zr study](https://arxiv.org/abs/2109.08126) motivate checking topology and
spatial morphology. We do not transfer their numerical thresholds or assign
their potentials to data with unknown provenance. Ordered-liquid candidates
need persistence and future-fate validation before being called prenuclei;
accepted Zr dynamics are absent from the available collection.

## Do these metrics predict crystallization performance?

We reuse the latest Al assay's 45 independent roots: 25 fit, 5 tuning, 15
development. Inputs use the checkpoint's 80-atom/radius convention. Future
labels remain original MD; the encoder sees relaxed present geometry. The
conditional onset readout includes current physical descriptors, making its AP
different from earlier temperature-only MACE readouts.

The final encoder's 12 ps onset AP is **0.203**, while the projector selects the
constant-risk checkpoint with AP **0.0325**. AP alone would make the encoder look
decisively better. Its source-averaged Brier improvement beyond current physics
is only **−0.000378**, with paired-source 95% interval **[−0.00816, +0.00591]**.
That does not establish a reliable gain across sources. The interval conditions
on this one encoder seed and this readout/data split.

For 9 ps future-order residuals beyond present physics, the final encoder's MSE
is **0.0391 worse** than the baseline (interval **[+0.0215, +0.0558]**); the
projector is **0.00543 worse** (interval **[−0.00719, +0.0168]**). Neither shows a
validated improvement on this conditional future-order test.

Across all 35 correlated checkpoints, encoder liquid-order R² correlates with
onset AP (Spearman **0.614**), but only weakly with Brier (**−0.165**). For the
projector those correlations are **0.208** and **−0.004**. These descriptive
associations are confounded by training age and readout selection; they are not
independent-sample significance tests or proof that optimizing a structural
metric will improve forecasting. [All endpoint contrasts](tables/endpoint-comparisons.csv)
and [their definitions](tables/METRICS.md) are retained.

## What to train and measure next

Use the **encoder output as the primary structural representation**, keeping
projector evaluation separate. A useful next matched ablation would retain this
recipe and add continuous instantaneous structure targets on the encoder
(local/averaged order and independent topology), with explicit liquid/interfacial
coverage. Compare that with removing FactorVAE under the same data, seed and
epoch schedule. This identifies a mechanism more cleanly than changing the
architecture, loss and data simultaneously.

Preserve similar physical environments while allowing real interfaces and
structural changes to separate. Spatial proximity alone must not imply an
invariance target. Pair smoothness with physical fidelity, noncollapse, and
response to measured change. Future prediction remains a separate endpoint,
conditioned on present structure. A geometry reconstruction gain does not by
itself establish usefulness for nucleation prediction.

## Validation and limits

Three targeted scientific tests pass: ideal FCC/BCC values and rotation
invariance, material-specific defect semantics, and meaningful shuffled/
collapsed coherence controls. Exact repeated checkpoint inference and all 37
evaluations completed. Final weights were verified bitwise against the last
periodic checkpoint. Metric definitions, input identities and implementation
hashes accompany exports. Broader existing layout tests have three failures in
the shared drift checker/stale family inventory; those utilities were not
changed here. Their details and the corrected post-fit epoch-number bookkeeping
audit are retained in the run's `technical/validation.json` and training receipt.
