# Encoder, TDA and relaxation: problems, evidence and next decisions

Research review — 10 September 2026

This document consolidates the problems found in our MACE structural encoders,
topological supervision and relaxed-target preparation through September 10.
It distinguishes demonstrated failures, corrections already implemented, and
questions that remain open. Older experiments used different inputs, targets,
losses and splits; their absolute scores are not one common leaderboard.

The latest available-data comparison has 36 completed training runs. The
larger uniform-MEAM dataset now has all 90 relaxed target shards and all 90
frozen-feature shards prepared. Its preparation completed at 16:09 CEST; no
full-dataset training result exists at the time of this review.

## 1. Main conclusion

There is no single explanation such as “MACE cannot learn topology” or
“transformers do not work.” We have found several distinct problems:

1. Some earlier encoder inputs could not observe all atoms that determined
   their TDA targets. This was a concrete architectural mismatch, now corrected.
2. Several objectives rewarded changes in feature scale, material separation
   or regularization more strongly than useful local structural information.
3. Compression and whitening could assign excessive influence to very small
   target directions. Better loss design helped substantially in the Al pilot.
4. Additional snapshots contain some useful information, but the strongest
   controlled comparison shows only a small, uncertain temporal advantage.
5. A relaxed target depends on the potential and relaxation protocol. Different
   potentials give different labels for exactly the same observed geometry.
6. Average descriptor prediction can look successful while neighborhood-specific
   topology remains poorly predicted. Independent-source evidence is still limited.

The current best-supported direction is a consistent target definition, complete
spatial coverage, modest supervised training, strong single-frame controls and
evaluation of local variation. Choosing a more complicated temporal module is
not yet the highest-priority decision.

## 2. What problem are we actually asking the encoder to solve?

The current scientific pipeline is:

```text
Observed past positions, atom identities and times
    -> MACE features -> temporal/spatial fusion -> embedding -> TDA prediction

Observed full anchor cell + specified potential + minimization settings
    -> relaxed full cell -> same 80 anchor-selected atoms -> TDA target
```

These are different tasks and must remain explicit:

| Task | Target | Main limitation |
| --- | --- | --- |
| Instantaneous geometry encoding | TDA of the observed cloud | May mostly describe thermal distortion or density |
| Relaxed-topology prediction | TDA after a specified full-cell quench | Requires inferring hidden surroundings and the selected minimum |
| Finite-temperature stable-state learning | A thermodynamic or dynamical state | Not defined by one zero-temperature energy minimization |
| Defect or grain-boundary recognition | A physically validated structural distinction | TDA reconstruction alone does not validate this ability |
| Future-state prediction | A specified future structural outcome | Cannot be inferred from good current-TDA reconstruction alone |

A local noisy snapshot need not determine its relaxed target uniquely. However,
we have not demonstrated a universal impossibility of single-frame prediction.
Full-cell geometry and the potential define a much more specific minimization
problem than an isolated 80-atom observation does.

Under squared-error training, the ideal predictor is the conditional mean of
the target given its observations. Several frames can reduce uncertainty if
they reveal relevant motion or persistent structure. They do not guarantee
identification of one basin, and a conditional mean persistence image need not
correspond to the persistence image of any single actual relaxed configuration.

“Most probable at temperature T,” “a local energy minimum,” “a global energy
minimum,” and “structurally stable over a future interval” are not interchangeable
target definitions. The present relaxation labels represent fixed-cell local
energy minima reached by the recorded procedure.

## 3. Issue register

“Corrected” means that the specific implementation/protocol problem has a
documented remedy. It does not imply that the resulting encoder meets the
scientific goal.

| ID | Problem | Evidence status | Current disposition |
| --- | --- | --- | --- |
| E1 | Encoder excluded atoms used by TDA | Demonstrated in older compact adapter | Corrected in the full 80-atom graph |
| E2 | Spatial pooling discards useful local detail | Strong Al ablation evidence | Atom-level fusion helps; finite context remains |
| E3 | Chemistry and length scale dominate geometry features | Demonstrated in mixed-material audit | Separate geometry-only and physical protocols |
| E4 | Regularization overwhelms topology supervision | Demonstrated by gradients and ablation | TDA-only Al controls implemented |
| E5 | Loss reduction masks rank loss or feature shrinkage | Demonstrated in earlier VICReg runs | Requires task and within-group evaluation |
| E6 | Temporal benefit over a strong anchor control is small | Demonstrated in current pilot | Larger independent-source test remains |
| T1 | PCA whitening strongly amplifies small directions | Demonstrated in failed temporal pilot | Balanced full-target Al objective implemented |
| T2 | Mean prediction conceals weak local information | Demonstrated in material and potential audits | Local baselines and centered diagnostics required |
| T3 | TDA is not a direct symmetry/defect label | Target-definition limitation | Physical defect validation remains open |
| R1 | Relaxed targets differ between potentials | Demonstrated on nine matched cells | Use consistent targets or explicit potential identity |
| R2 | CG can stall before force convergence | Observed preparation failure | FIRE recovery completed |
| R3 | Converged minima need not be unique or thermodynamic states | Open scientific limitation | Basin/protocol sensitivity needs more measurement |
| D1 | Many patches but few independent source histories | Demonstrated in existing splits | Larger 30-source dataset prepared |
| N1 | Numerical approximation can exceed subtle signals | Ordinary BF16 failed an earlier screen | Qualified compensated path and precision checks |
| O1 | Export, analysis and restart semantics can differ | Several historical failures recovered | Preserve exact protocol/checkpoint provenance |

## 4. Encoder problems

### E1. The old spatial support did not contain the target

The earlier compact adapter received 80 atoms but excluded geometric messages
outside 6.5 Å from the center, with tapering already beginning at 5 Å. Its TDA
target used the center and 64 neighbors regardless of that center-distance limit.
The 3.5 Å TDA death cutoff was incorrectly insufficient as a support argument:
it limits filtration radii, not distance of input atoms from the patch center.

An audit found excluded target atoms in 64.3% of Al, 100% of Mg and 99.0% of Ta
training views. Moving only excluded atoms changed the target while leaving
every retained encoder edge and endpoint position exactly unchanged. No choice
of weights could distinguish those constructed input pairs.

The current plain80 and denoising encoders use all 80 target atoms in the finite
MACE graph. That removes this specific blindness. It does not supply atoms
outside the patch: the native 5 Å edge cutoff and artificial patch boundary
still limit environmental context. Increasing temporal depth cannot recover
spatial information that is never observed without relying on correlations.

Evidence: [support audit](../output/mace_joint_properties_20260908/INTERIM_REVIEW_20260908.md),
[full-80 protocol and checks](../experiments/mace_plain80_20260909/README.md).

### E2. Pooling before temporal fusion can discard useful structure

The first temporal model reduced each frame to 256 pooled scalar MACE features
before attention. Once spatial information is lost at that point, the temporal
module cannot directly recover which atom carried it.

The latest atom variant follows each atom identity through time and then learns
spatial pooling. Its matched anchor-only control repeats the current atom
features in all temporal slots, preserving module capacity. On the same Al
pilot, the anchor MLP has balanced MSE 0.6875, while the atom-anchor model has
0.2315. This supports retaining atom-level structure and changing the learned
readout. The comparison changes architecture and training details together;
it does not isolate one pooling operation as the sole cause of the improvement.

MACE remains frozen in this experiment. This is a useful control, not a claim
that the pretrained representation is optimal. It limits training cost and
removes backbone drift while testing the target, readout and temporal fusion.

### E3. Geometry-only and physical-material encoders are different protocols

The successful earlier GeoFrame setup used source-radius-normalized coordinates
without explicit element labels. A subsequent MACE setup used physical Å and
actual Al/Mg/Ta species. These are not matched inputs.

In the frozen-feature audit, between-material means explained 99.51% of the
standardized variance with physical coordinates and actual species. Normalizing
distance alone left 99.45%; fixing species alone left 94.60%; combining a common
normalized scale and fixed channel reduced it to 20.69%. This is diagnostic
variance decomposition, not a downstream accuracy comparison.

The geometry-only replacement now has its own normalized, fixed-channel API.
The physical temporal/denoising models retain physical coordinates and material
identity. They answer different questions and their caches/checkpoints cannot
be interchanged casually. In Al-only training, cross-element separation is
removed, but temperature, density and source/potential shortcuts remain possible.

Evidence: [VICReg/input audit](../output/mace_vicreg_audit_20260909/REPORT.md),
[encoder APIs](mace_temporal_encoder.md).

### E4–E5. The optimization objective can improve while the representation worsens

The failed September 9 temporal model optimized TDA MSE plus 25 times the
variance penalty plus covariance. Across 16 completed epochs, 99.70% of total
loss reduction came from regularization. On the fixed diagnostic batch, its
initial encoder regularization-gradient norm was 613.1 times the TDA-gradient
norm; even at selection it was 42.0 times larger. These gradient ratios were
measured before Adam preconditioning and are not an average over training.

Frozen ridge decoding of the temporal embedding worsened from MSE 0.9577 to
1.1227. Pre-fusion MACE features retained nearly the same predictive quality.
The observed damage was therefore concentrated downstream of the backbone,
not a demonstrated loss of all structural information inside MACE.

An earlier plain80 VICReg run also lost within-Al rank before TDA was active:
effective rank fell from 4.78 initially to 1.41 at epoch two. A single scalar
shrinkage of frozen features reduced validation loss from 67.45 to 18.73,
almost matching the actual first epoch's 18.41, without learning new geometry.
Thus low total loss, low covariance or global spread do not establish useful
within-material structure. This was not attributable to an active TDA head in
those pre-TDA epochs.

The Al ablation directly helps separate this issue: otherwise matched pooled
transformers scored 0.4424 with regularization versus 0.2693 with TDA alone.
Balanced full-target supervision improved this to 0.2349. Gradient balancing
in older runs also did not guarantee joint improvement: equal magnitudes do
not remove conflicts between objective directions or define the correct target.

Spatial/temporal invariance introduces an additional unresolved scientific
tradeoff: nearby atoms and nearby times may genuinely straddle a defect or
transition. Pulling their embeddings together can suppress desired differences.
The relevant tolerance needs empirical validation; proximity alone is not a
proof that two neighborhoods should share a representation.

Evidence: [temporal failure diagnosis](../experiments/mace_temporal_transformer_20260909/DIAGNOSIS.md),
[VICReg audit](../output/mace_vicreg_audit_20260909/REPORT.md),
[balanced-objective experiment](../experiments/mace_balanced_representation_20260908/README.md).

### E6. Learned temporal fusion is not yet clearly better than a strong anchor model

| Same Al pilot; three training seeds | Test balanced MSE |
| --- | ---: |
| Mean pooled frame features with learned head | 0.4073 |
| Pooled temporal transformer, balanced TDA | 0.2349 |
| Atom-anchor control | 0.2315 |
| Atom-temporal model | 0.2257 |
| Relaxed-input neural reference | 0.1246 |
| Relaxed-input ridge reference | 0.0698 |

The source-averaged temporal gain over atom-anchor is 3.07%, with exploratory
95% interval [-0.09%, 5.42%]. There are only two held-out trajectories, so this
does not establish a robust temporal advantage. The relaxed-input reference
shows that the target is quite decodable when the relaxed geometry is available;
it does not prove that hot observations contain enough information to attain it.

For the trained atom-temporal model, repeating the anchor at inference worsens
MSE to 0.2677, while reversing the past gives 0.2263. The model uses extra
observations, but little sensitivity to their ordering is visible in this test.
Replacing history at inference is not equivalent to training an anchor-only
model; the latter is the stronger architecture control. Attention weights,
nonzero historical gradients and changed embeddings also do not by themselves
demonstrate useful predictive contribution.

There is no matched GRU-versus-transformer result. There is also no completed
velocity-input ablation. The original histories spanned 0.4 ps at 0.1 ps spacing;
the independent MEAM histories span 3 ps at 0.75 ps spacing. Five observations
are not five independent noisy samples, and equal frame counts are not equal
physical windows. Actual per-sample times are now passed in the mixed-cadence
encoder. Extending the horizon could reveal more motion or cross a real basin
change; the appropriate window remains an empirical question.

Evidence: [current comparison and interventions](../output/mace_al_denoising_20260910/available_mixed/analysis/RESULTS.md).

## 5. TDA and supervision problems

### T1. Descriptor compression and weighting can hide the intended signal

Our current target is a 144-dimensional alpha-complex persistence summary:
16 H0 death-radius features, 64 H1 birth/lifetime features and 64 H2 features.
Filtration values are converted from squared radii to radii in Å. H1/H2 surfaces
are lifetime weighted; finite deaths above 3.5 Å are excluded. This is a lossy
descriptor of a finite point set, not the complete persistence diagram.

The failed mixed-material temporal model compressed 144 coordinates to 32 PCA
coordinates and whitened them. The first four components contained 99.6407%
of global raw variance, and the first 16 contained 99.9935%. First/last retained
standard deviations differed by 1,454 times, giving roughly 2.11 million times
different squared-error weights per unit raw change.

This can put large training pressure on very small directions. Small variance
does not prove irrelevance: subtle defects may also be low-variance signals.
Their physical meaning, numerical repeatability and within-condition variance
must be checked rather than simply discarded.

The newer Al experiment retains all 144 coordinates and balances H0/H1/H2 by
training-only block scales, including a floor. It improves this particular
comparison but still defines a chosen metric. Equal block weights are not a
physical theorem, and changing the training distribution changes its scale.

Evidence: [target implementation](../src/analysis/liquid_structure.py),
[scaling diagnosis](../experiments/mace_temporal_transformer_20260909/DIAGNOSIS.md).

### T2. Average accuracy is not local topology learning

The failed temporal experiment had aggregate raw-target R² 0.809 but mean
within-material raw-target R² -0.159. Differences between element means could
conceal weak local prediction. Mg alone contributed about 55% of validation
error while representing 25% of validation anchors.

The potential audit exposes a similar issue within Al. Fixed-model error was
0.0936 against MEAM labels and 0.8492 against EAM labels. However, 81% of the
squared label difference is a global per-pixel mean shift. A correction fitted
on eight sources reduced EAM error on the ninth to 0.1316. A constant EAM
descriptor fitted on those same eight sources scored 0.1306. Calibration
removed average bias without demonstrating recovery of local variation.

After removing each source's descriptor mean, paired MEAM/EAM correlations were
only 0.195, 0.178 and 0.189 for H0/H1/H2. These are diagnostic correlations for
these early frames, not a proof that all nonlinear transfer is impossible.

Always distinguish trained-head performance from information recoverable by
a newly fitted frozen probe. Report constant baselines, within-material and
within-source/frame variation, each homology block, and both raw and scaled
errors. Do not compare absolute MSE across changed PCA bases, scales or datasets.

Evidence: [potential comparison](../experiments/mace_al_denoising_20260910/POTENTIAL_DIFFERENCES.md).

### T3. Topology, symmetry breaking and defects are related but not identical

Distance-based persistence is invariant to rigid translation and rotation.
Rotating a perfect lattice as a whole therefore does not produce a distinct
orientation label. Local distortions, packing changes, interfaces and defects
can change persistence, but no current result establishes that our descriptor
uniquely identifies defect type or grain misorientation.

For two colliding grains, a patch spanning the interface can contain changed
relative geometry. A patch inside either perfect grain may look equivalent
under rotation. Detecting relative orientation or a larger interface can require
orientation-sensitive auxiliary measurements or spatial context beyond 80 atoms.
No controlled collision, dislocation, vacancy or grain-boundary benchmark has
been completed for this denoising encoder.

Finite-cloud boundaries and the selected atoms influence persistence. Keeping
the same anchor-selected identities is necessary for the matched denoising
question, but those identities need not remain the relaxed center's nearest
80 neighbors. Re-selecting them would define a different target. The finite
death cutoff suppresses some large boundary features; it does not remove all
boundary sensitivity or make the descriptor periodic.

In our potential comparison, H0 counts remain 79 and H1 counts approximately
276 under both potentials, while H2 lifetime distributions change. EAM has
more H2 intervals above 0.05 Å lifetime but fewer above 0.10 Å. “More cavities”
is therefore incomplete without a persistence threshold, and does not count
vacancies. Similarly, persistence images with similar values need not imply
identical structures: compression loses information.

Distance normalization also removes or changes physical scale information.
It should be chosen according to whether density/strain is signal or nuisance,
not treated as an automatic repair. In the matched potential audit, simple
first-shell distance normalization did not remove the descriptor discrepancy.

## 6. Relaxation problems

### R1. Potential identity is part of the target definition

Lee2003 MEAM and Mendelev Al1 EAM/FS agree on a very similar reference FCC
spacing, but use different fitted interactions and functional forms. The local
Al1 file was verified against the official LAMMPS Al_mm file beyond its comments.

Nine matched 70,304-atom cells were minimized with both potentials using the
same full periodic starting geometry, box, FIRE settings and force threshold.
The same centers and atom identities were used for both labels. This isolates
target-potential sensitivity on MEAM-generated observations from the earlier
mixed-data temperature/cadence confounding.

| Matched-cell result | Measurement |
| --- | ---: |
| Mean displacement between relaxed endpoints | 0.956 Å |
| First-12-neighbor identity overlap | 82.2% |
| First-shell distance spread, MEAM / EAM | 0.285 / 0.192 Å |
| Local FCC+HCP fraction, MEAM / EAM | 0.62% / 4.37% |
| Mean hot-state force-direction cosine | 0.882 |
| EAM RMS force on MEAM minimum | 0.227 eV/Å |
| MEAM RMS force on EAM minimum | 0.181 eV/Å |

Own-potential RMS forces are approximately 0.0003 eV/Å. Each potential prefers
its own endpoint energetically, in all nine sources. Both configurations remain
mostly unclassified/disordered under the specified PTM templates and threshold;
these fractions are not evidence of bulk crystallization.

The prespecified paired model-error test gives p=0.00390625, using nine source
units and 512 swaps. It establishes that this error difference is consistent
in the selected sample. It does not establish physical superiority of one
potential, nine times more disorder, or a comparison of native dynamics under
both potentials. The post-hoc calibration above is a separate explanatory test.

Mixing labels is possible as an explicit conditional task, for example using
potential identity and separate heads. Treating them as interchangeable labels
without a target-potential input is scientifically ambiguous. A shared
potential-independent structural representation may still be useful, but
equivalence of its two target heads must be tested rather than assumed.

### R2. Force convergence can fail, and minimizer changes alter the protocol

The original CG preparation stalled on source 001 at 400 K, frame 640. Energy
repeated at about -231793.0675 eV while maximum force stayed at 0.08909 eV/Å,
above the required 0.01 eV/Å. Five target frames had completed. The stalled
attempt was stopped and recorded as a failure; its unconverged target was not
silently accepted.

The older GPU binary did not support the needed FIRE path. A validated newer
build converged on the identical input in 74.22 seconds, reaching maximum force
0.00912 eV/Å and energy -231795.4530 eV. A separate FIRE output preserved the
scientific distinction and the original CG artifacts.

Five contexts have converged CG and FIRE labels, from only two independent
sources. Their balanced descriptor distances range 0.00416–0.00824. This is
much smaller than the 0.956 mean potential difference in the separate nine-source
audit, but the cohorts differ and two sources do not establish general minimizer
equivalence. The available-data pilot retains five CG labels and four additional
FIRE MEAM labels by provenance. The new 90-frame dataset uses FIRE consistently.

Evidence: [relaxation/recovery record](../experiments/mace_al_denoising_20260910/README.md),
[CG/FIRE comparison](../output/mace_al_denoising_20260910/available_mixed/analysis/cg_fire_target_sensitivity.json).

### R3. Convergence is necessary, but basin stability remains open

A force threshold checks stationarity to a specified tolerance. It does not
prove uniqueness, a global minimum, robustness to perturbations, or equality of
labels under different minimizers. More tolerance and repeated-start checks
are needed to quantify the label variability relevant to subtle topology.

The full periodic cell is relaxed before extracting patches, which avoids the
incorrect isolated-cluster relaxation problem. However, the encoder sees only
local histories while the target depends on the surrounding cell. How much
unobserved context limits prediction is not yet quantified.

The box is fixed. Own-potential relaxed virial pressures in the matched audit
average -1.91 GPa for MEAM and -1.44 GPa for EAM: these are not zero-pressure
equilibria. Cell relaxation would change the target and should be a separate
protocol, not an unrecorded preprocessing change.

Large hot-to-relaxed displacements also show that this is not just subtracting
independent small Gaussian noise. In the matched audit, mean atom movements
are 0.417 Å under MEAM and 0.931 Å under EAM. The causal history may describe
vibrations, rearrangements or an evolving basin. Which component is predictable
requires controlled history/horizon and basin-sensitivity tests.

## 7. Data, evaluation and numerical problems

### D1. Source count, leakage and confounding limit conclusions

The failed temporal pilot had 26,624 training anchors from only nine source
trajectories and one target time per source. Patches overlap and share history;
more epochs do not create new independent trajectories. Ta validation also
shared a source. That validation selected the checkpoint, so it was not an
untouched test set.

The latest mixed Al pilot has 13,056 training, 4,864 validation and 1,280 test
neighborhoods, but only two held-out trajectories. EAM continuations share an
ancestor; its test continuation was validation data in earlier experiments.
Potential covaries with temperature (650 versus 400 K), history cadence
(0.1 versus 0.75 ps) and campaign. Per-potential score differences in that pilot
alone do not isolate potential effects. The matched potential audit was designed
to address that specific ambiguity, not all remaining generalization questions.

The prepared larger dataset contains 30 independent sources at 400/450/510 K,
three target frames per source and 256 neighborhoods per frame. Its split is
18 training, six validation and six test sources, totaling 13,824/4,608/4,608
neighborhoods. This improves source/time coverage; it still needs training and
analysis before supplying new evidence.

Bootstrap units must be independent sources, not neighborhoods or training
seeds. A source-bootstrap interval with one source is not estimable; an older
zero-width temporal interval was corrected for this reason. Small-sample
intervals, including the current two-trajectory temporal interval, remain
exploratory. Per-temperature and per-phase/context effects should be visible.

Static plots also include ancestors of some training continuations. Imposed
clusters, attractive spatial maps and reduced neighbor/random distance ratios
are descriptive diagnostics, not independent evidence of physical phases,
defect identities or future crystallization. Compare against constant,
single-frame and physical-descriptor baselines on held-out lineages.

### N1. Numerical approximations must be judged against the signal of interest

Ordinary BF16 was rejected in the earlier numerical screen: its radial-only
embedding error was 5.52 times the response MSE to a 0.005 Å perturbation.
A first compensated implementation also failed because compilation removed
the precision casts defining its residual. Enabling the required cast semantics
and checking forward/backward computations corrected that implementation.

The qualified compensated path passed subsequent gradient and perturbation
checks. These observations do not show that the current temporal failure was
caused by BF16. Similarly, full-cell CPU/GPU forces agreed near machine precision
for the validated MEAM and EAM backends.

In the latest audit, centered float16 coordinate storage contributed about
5.9e-6 balanced descriptor error, versus 0.956 from changing the potential.
This comparison used exact relaxed-dump coordinates at 16 centers per source.
It isolates local cache quantization, not the effect of quantizing the shared
original full trajectory before either relaxation. Quantization of frozen atom
features is a separate step and is mirrored by the exported encoder.

The storage policy remains verified float16 positions, float32 boxes, exact
integer identities/timelines, and unchanged integration/restart precision.
Finite-difference velocity estimates from stored positions would need their own
cadence and quantization assessment; the current encoder has no velocity input.

Evidence: [BF16 screen](../experiments/mace_bf16_20260908/README.md),
[potential precision controls](../experiments/mace_al_denoising_20260910/POTENTIAL_DIFFERENCES.md),
[storage protocol](trajectory_conversion.md).

### O1. Correct execution and correct science are separate requirements

Verified implementation properties include identity-aligned histories, exact
agreement of history anchors with paired caches, genuine past-only windows,
physical time inputs, cached/direct gradient agreement and exported-inference
checks. Historical gradients are nonzero. These checks argue against specific
indexing or disconnected-attention explanations; they do not guarantee useful
denoising or cover every possible implementation defect.

The history axis is not the earlier cache's augmentation/view axis. A hot view,
its relaxed partner and a future view must not be treated as three past frames.
The single-frame static analysis API cannot substitute for temporal inference.
Exports need their input convention, feature scaling, time offsets and decoder
target scaling, not just an unlabelled weight tensor.

Recovered operational issues include a missing analysis-config default,
Dynamo recompilation limits during post-training inference, an overly strict
bitwise pooling/round-trip assertion, and a Lightning last-checkpoint file that
did not represent the actual final optimizer state. Compiled/eager inference
was checked when disabling compilation for analysis, and future saves explicitly
preserve the final optimizer state. These were operational problems with
documented fixes, not evidence that TDA is intrinsically unlearnable.

Detached execution survives closing the IDE, but work inside an existing Slurm
allocation still depends on that allocation remaining alive. Preparation,
training and analysis have distinct completion states. Completed target/feature
caches must not be reported as completed model training. Scientific changes
require explicit configurations and separate output provenance; retired commands
in historical experiment records may need the recorded source snapshot.

Evidence: [analysis recovery](../experiments/mace_original_vicreg_20260909/README.md),
[current encoder contract](mace_temporal_encoder.md),
[experiment status and reproduction](../experiments/mace_al_denoising_20260910/README.md).

## 8. What we should test next, in order

| Priority | Question and comparison | What the result would resolve |
| --- | --- | --- |
| 1 | Train anchor and temporal models on the prepared uniform-MEAM dataset, with identical source splits and balanced TDA | Whether history helps beyond spatial representation on more independent sources |
| 2 | Keep relaxed-input, constant-descriptor, anchor and learned/mean-history controls | Separate target decodability, mean prediction and actual temporal benefit |
| 3 | Compare repeated quenches, small input perturbations and tighter convergence on selected held-out contexts | Estimate protocol/basin variability before demanding finer prediction accuracy |
| 4 | Evaluate defect/interface examples with known construction or independent structural measurements | Establish whether TDA and the embedding preserve the symmetry breaking we care about |
| 5 | Test explicit potential identity or separate heads, then transfer in both generating-potential directions | Separate calibratable mean shifts from local structural transfer |
| 6 | Once target and evaluation are stable, compare history lengths, velocities, GRU and transformer with matched budgets | Decide which temporal information and architecture are actually useful |

Success should mean reproducible improvement over a strong trained anchor
control on independent sources, preservation of meaningful local variation,
and demonstrated usefulness for the selected physical task. A falling total
loss, high global R², increased rank, or attractive clusters alone is insufficient.

Do not read the current evidence as proving that noisy input cannot predict
relaxed topology, that a transformer is worse than a GRU, that EAM or MEAM is
physically superior, or that TDA cannot detect defects. Those conclusions exceed
the experiments completed so far.

## 9. Evidence map and file roles

| Topic | Primary repository record |
| --- | --- |
| Missing spatial support | [September 8 architecture audit](../output/mace_joint_properties_20260908/INTERIM_REVIEW_20260908.md) |
| Geometry/species shortcuts, rank loss, loss shrinkage | [VICReg audit](../output/mace_vicreg_audit_20260909/REPORT.md) |
| Failed joint temporal encoder | [September 9 diagnosis](../experiments/mace_temporal_transformer_20260909/DIAGNOSIS.md) |
| Completed loss and fusion comparisons | [36-run Al results](../output/mace_al_denoising_20260910/available_mixed/analysis/RESULTS.md) |
| Potential, force, structure and calibration differences | [Detailed September 10 report](../experiments/mace_al_denoising_20260910/POTENTIAL_DIFFERENCES.md) |
| Relaxation failure, recovery and larger data | [Al experiment record](../experiments/mace_al_denoising_20260910/README.md) |
| Numerical precision | [BF16 audit](../experiments/mace_bf16_20260908/README.md) |
| Current input/export interfaces | [Temporal MACE documentation](mace_temporal_encoder.md) |

This Markdown file is a versioned research review, indexed from the repository
README, encoder documentation and Al experiment record. No encoder, target
producer, relaxation settings or training jobs were changed to produce this
review.
