# Matched spatial VICReg mechanism and cluster information

Version 1, fixed before the nine 24-pass fits. Producer:
`src/research/spatial_vicreg_bias/{data,train,evaluate,report}.py`.
This is a structural mechanism study, not a crystallization forecast.

## Population and actual inputs

Use fixed Al64 v1 identity
`e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d`.
Roles remain 90 training, 15 selection, 15 calibration, 30 test independent melt
ancestors. These are previously examined test sources, not a fresh blind test.
An epoch contains all 1,157,760 training-source anchor/frame observations.
No phase, future outcome or context-clearance filtering enters encoder training.
Analysis includes 64 frozen centers at frames 0,64,...,768, plus every original
benchmark observation. The uniform track has 124,800 rows; original all64 and
legacy16 rows retain their exact source/frame/atom mapping and order. Geometry
uses observed float16 MD decoded in float64 with periodic minimum images. No
minimization, history, velocity, temperature, absolute time, material ID, atom
ID, absolute position or physical teacher is an encoder input. Frames and atom
IDs only identify observations; coordinates are relative and divided by the
fixed Al scale 9.192189 Å. No future target is consumed anywhere in this study.

Each parent preserves the original nearest 80 atoms as view A and adds the
remaining nearest atoms to form a 128-atom candidate pool. B is one of the eight
nearest noncentral centers; its nearest 80 are chosen within that parent pool.
The eight possible membership lists are frozen. B is a parent-truncated crop,
not a claim of globally nearest 80 about B. Distances, crop radii, shared-atom
fraction and support-union visibility are retained. Two independent jitter
(0.01 Å) and x-reflection (probability 0.5) versions per view give A1,A2,B1,B2.
All atoms in each 80-point crop are consumed, with no hidden 8 Å filter. Batch
and microbatch are 256 parents/1,024 view presentations, except the final 128
parents. Shared weights, batches, per-step RNG and four-view distributions are
identical across alpha; optimizer updates differ through the alignment loss.

## Training intervention and exports

GeoFormerV2 128 with the archived architecture, plain VICReg, FactorVAE off.
Projector: the repository's 128→128→128 MLP with BN/ReLU after the first two
layers. Raw encoder and once-applied projector are both evaluated.
`same=(MSE(A1,A2)+MSE(B1,B2))/2`,
`cross=(MSE(A1,B2)+MSE(B1,A2))/2`;
`loss=25*((1-alpha)*same+alpha*cross)+25*V+C`.
V is mean ReLU(1−sqrt(population variance+1e−4)) over the same four batches;
C is mean squared off-diagonal sample covariance sum / 128 over those batches.
Alpha=0,0.5,1; paired seeds=17,29,43. AdamW, lr=1e−4, decay=1e−4, one-pass
linear warmup from 0.05×lr, then cosine to 1e−6 at pass 24; gradient clip=1.
BF16 forward with FP32 loss reductions; compiled training uses default mode,
not reduce-overhead. Inference is eager FP32 with TF32 off and frozen BN.
Initialization/every full pass is saved. Full assays: 0,1,4,8,12,18,24; primary
endpoints 12 and 24. No outcome/appearance/AP-based checkpoint selection.
Same-shaped inference repeats must be bitwise equal. Scientific training alone
logs online W&B; frozen fits/evaluations are local and update the existing run.

## Physical descriptors and visibility

The exact A crop supplies 412 descriptors: geometry 69; local/neighbor-averaged
bond order 40; CNA 45 at 3.2 Å, 3.6 Å and adaptive first-shell cutoffs; topology
258. Topology uses Gudhi safe alpha complexes for nearest32 and all80, finite
H0/H1/H2 intervals, square-root conversion of squared filtration radii to Å,
log lifetime summaries, entropy, soft Betti curves, H0 death features and H1/H2
6×6 persistence images. Essential H0 is excluded; large finite intervals are
retained. This reuses the mathematical kernels in liquid_predictability but
**does not** call its radius-8-truncating patch wrapper. Descriptor calculations
are operational structure references; they are not definitive liquid-state
labels or evidence of dynamics.

Full-cell independent PTM: cutoff 0.1, crystalline types FCC/HCP/BCC (1/2/3),
frozen original extraction receipts and chunk hashes. `clear_input` requires
zero detected crystalline atoms in A, not merely a noncrystalline center.
Pair-union and all-eight-augmentation-union masks cover exactly the respective
consumed atoms. The crystalline fraction and whole-cell crystal absence are
retained separately. All/noncrystalline-center/strict-clear cohorts have
readouts; K=7 also reports at most one or three detected atoms out of 80 as
sensitivity analyses. These vary allowed detection counts, not the PTM cutoff.
Visibility annotations stratify evaluation and **do not filter encoder training**.
Clear inputs can still be physically ordered; mixed support can be meaningful.

## Clusters and descriptor readouts

MiniBatchKMeans, K=3,6,7,10, n_init=3, batch=4096, max_iter=200, fixed seed
20260929; fit only uniform training observations. All other roles/tracks receive
frozen assignments. Euclidean native coordinates are used without per-axis
rescaling or a UMAP/t-SNE transform. No K is chosen using held-out metrics.

For each readout cohort, fit only uniform training ancestors. Target centering
and SD use the uniform training population. Exclude effectively constant targets
(SD <= 1e−8 max(1,abs(mean))); retain names/scales. Readouts are:

- Cluster membership → source-weighted per-cluster descriptor means, with 20
  global-mean pseudo-observations. Empty clusters predict the training mean.
- Fixed ridge (alpha=10, intercept unpenalized, training feature standardization)
  with phase/proximity/density controls, then the same inputs plus cluster
  one-hot. This is a penalized Gaussian-mean fit, not an AP objective or search.
- At K=7, continuous native embedding → the same descriptor targets with the
  same fixed ridge recipe, to measure information lost by hard clustering.
- Cluster assignments permuted within source × input-crystal-presence × distance
  bins [3,6,12,24,48 Å] × support-fraction bins [0,.05,.25,.5,.75,1]. Report
  movable fraction. This is a coarse nuisance-preserving negative control,
  not a p-value or exact conditional-independence test.

Controls contain support crystal fraction and its square, first14-neighbor
crystal fraction and square, nearest-crystal distance clipped at 60 Å (linear,
quadratic and hinge terms at 4/8/16/32 Å), whole-cell crystal absence, central
FCC/HCP/BCC flags, and local density12/density80. They are privileged diagnostic
inputs derived from current geometry/full-cell PTM, not encoder or forecast
inputs. Density identity targets are excluded from family averages. Per-feature
scores remain available and identify those tautological targets explicitly.

For every role/track, compute per-source mean squared error on training-SD
standardized targets, then average sources equally. `train_mean_skill` is
1−model MSE/training-mean-predictor MSE on that held-out population; it is **not**
ordinary R² relative to a fitted test mean. `conditional_cluster_gain` is
MSE(controls)−MSE(controls+cluster), positive for improvement. Family scores
average active features equally, excluding density identity targets. Uniform
test per-feature scores, source error arrays, readout coefficients, target
scales and every observation's cluster assignment are retained. Source bootstrap
95% intervals use 1,000 resamples of whole source IDs, paired for conditional
gain. Sources and seeds, not atoms or checkpoints, are replication units.
The summary shows paired-seed curves; it does not claim nine independent datasets.
Minimum readout support: 128 training observations from >=3 sources; evaluation
>=30 observations from >=3 sources. Unsupported cases are explicit, never zero.

## Spatial-field measurements and controls

Pair squared latent distance is mean squared coordinate difference divided by
mean per-coordinate **uniform training variance**. Effective rank is exp entropy
of the held-out covariance eigenvalue proportions. Also report variance relative
to training, neighbor cluster agreement, agreement after permuting B labels
within the same source/frame, PTM adjusted mutual information, and pair distance
when actual A/B crystalline fractions differ by >=0.25. This crossing definition
uses support fractions, not a claim that both center PTM labels were measured.
Masks include mixed/noncrystalline centers, clear A/B unions and crystal-absent
cells. Low distance without retained variance/rank is collapse, not good structure.

The phase direction joins mean train embeddings of >=90%-crystalline and <=5%-
crystalline supports, giving those endpoint means scores 0 and 1. A distance
contrast is (distance to crystal core − distance to bulk liquid)/2, with core
PTM-crystalline and >=80% crystalline first14 neighbors, liquid noncrystalline
and <=10% crystalline neighbors. It is **not** signed distance to a reconstructed
interface. Coordinate selection uses absolute training Spearman correlation;
coordinates need not align across seeds. Held-out profiles show raw scatter and
2 Å-bin medians (>=30 rows), selected native coordinates and the bulk direction.
The 10–90% width uses uniquely supported upward crossings in adjacent populated
bins only; missing/ambiguous crossings give undefined. Reversal counts accompany
widths; pooled profiles do not establish smooth individual paths.

Reference controls use the full-cell PTM indicator averaged over actual A80,
then 1,2,4 diffusion steps h←0.5h+0.5 mean(first14 neighbors) before A80 averaging.
Their crystal exposure is propagated over exactly the same graph. These are
privileged labels/expanded support controls and cannot be ranked as equal-input
encoders. Local q6 and averaged q6 controls use the same 80-atom geometry; their
own feature readouts are tautological. Three untrained GeoFormers (epoch0) supply
geometry-only initialization controls. Similar shells from these controls show
sufficiency, not that diffusion uniquely explains a trained encoder.

No future-prediction metric is introduced. Earlier success with crystal visible
in prediction context does not establish recognition of a crystal-free precursor.
This study evaluates static information and spatial effects. Temporal persistence,
new-nucleus prediction and Ta/Zr transfer remain distinct follow-ups.
