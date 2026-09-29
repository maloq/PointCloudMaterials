# Encoder mechanisms: evidence audit and execution

**Completed28 September:** all18 scientific fits,54 milestone quality and54
displacement assays, plus18 adaptation endpoint evaluations finished. See the
[completed results](COMPLETED-RESULTS.md) for three-seed contrasts, training
trajectories and paired-source uncertainty. Earlier updates below retain their
historical cutoffs.

Updated28 September 2026. **All nine encoders trained24 epochs; controls and birth
audits completed. Milestone evaluation failed partway through and prevented the
nine adaptation fits from starting.** See the [results audit](RESULTS-20260928.md)
for the completed comparisons and the limits on interpretation.
The [later progress report](PROGRESS-20260928.md) now includes all three Epi
seeds at epoch24 and the observed structure/prediction trade-off through training.
Question: which parts of the encoder pipeline preserve useful local structure,
organize it into meaningful distances, and provide information about future
crystallization? This proposal follows a repository-history audit, rather than
starting a new architecture or loss sweep from the literature alone.

## What we already tried

These are separate experiments, not rows of a common leaderboard. Historical
time/temperature inputs, retired reconstruction treatments, AP objectives,
different observation supports and reused test sets keep their actual labels.

| Evidence | Completed finding | Consequence for the next experiment |
| --- | --- | --- |
| [Predictive atlas and VAMP, September 1–3](../../docs/predictive_atlas_current_progress_20260903.md) | On the harder matched future-law assay, linear VAMP retrieval distance was 0.47130 versus static PCA 0.45715, a 3.095% deterioration. Temporal atlas improvement was 0.437%; only two matched held-out sources supported that comparison. | VAMP is already tested. A new kinetic treatment needs a specific changed assumption and a bounded diagnostic first. |
| Same atlas: ordinary-MD temporal training and encoder fine-tuning | Ordinary realized-future prediction improved 0.95%, while shooting-law retrieval worsened 0.175%. Three-seed fine-tuning again selected the unchanged encoder. | Predicting one future and representing a conditional future distribution are different targets. More updates alone were not enough. |
| [Liquid SRO benchmark](../../docs/liquid_sro_benchmark_20260905.md) | Corrected MACE+VICReg was close to random MACE; smooth-density and SOAP controls were stronger on several future observables. Three seeds, six held-out Al sources; historical baseline included temperature. | Retain an initialized-encoder control and simple geometry baselines. Historical numerical gains cannot be copied into a condition-free table. |
| [Frozen local-state maps](../mace_local_state_20260915/RESULTS.md) and [motion study](../mace_local_motion_20260916/RESULTS.md) | Smoothing improved some group metrics but lost instantaneous topology; the physical map increased TDA error 2.55-fold. All 44 motion fits failed the stated joint acceptance gate. | Do not revive discarded replacement maps or optimize smoothness as the goal. |
| [Geoformer 35-pass audit](../../docs/encoder_research/results.md#geoframe-through-35-passes) | Liquid readout improves in the encoder and declines in its projector. Epoch34 VICReg and epoch159 VISReg were different objectives. | Always audit the export stage. The original pictures do not isolate training duration. |
| [Parameter queue review](../../output/encoder_research/mace-vicreg-epi-20260923/RESULTS-20260926.md) | Of 28 planned fits, 11 training fits and 10 fit-and-analysis tasks were documented complete. Matched VISReg reduced liquid-neighbor error about 14% in two seeds; FactorVAE removal was inconsistent. | Do not claim a completed parameter optimum or duplicate unfinished projector/covariance comparisons without checking receipts. |
| [Paired temporal MACE/Epi](../../output/encoder_research/mace-vicreg-epi-20260923/RESULTS-20260926.md) | Six fits, 24 epochs, two seeds: Epi liquid-neighbor error 1.410 versus VICReg 1.679. Epi+variance had better predictive proper-score point estimates, without established benefit beyond current physics. | Structure retention and predictive utility differ. Historical conditional probes included temperature and age-related inputs. |
| [Relaxation/history follow-up](../crystallization_followup_20260922/RESULTS.md) | Relaxed observations lowered integrated Brier by 13.3%; dense original history worsened it against repeated-current control. Other additions had uncertain gains. | Full-cell relaxation changes observations; history needs an input-matched control and a declared question. |
| [Distance/future factorial](../structural_state_future_20260923/README.md), [findings](../../docs/encoder_research/results.md) | Eight fits, two seeds; no predeclared mechanism success. Small distance benefit; future-residual improvement did not replicate. | Do not restart the retired physical-reconstruction/future-residual pipeline. |
| [Latest eight MACE checkpoints](../encoder_quality_20260926/RESULTS.md) | Epi improves liquid diagnostics modestly; fine-tuning trades some structural/temporal measures for prediction. Descriptor add-back still helps. | Investigate accessibility, paired-view alignment and adaptation separately. |
| [Al64 batch256 context results](/work/PERSO/vmorozov/analysis/encoder_context/al64-epochs-20260925/RESULTS.md) | All 16 historical context fits completed, including now-retired physical initialization. Epi/VICReg did not uniformly improve event NLL against scratch. | Existing long training is evidence, not something still awaiting its first result. |
| [Batch1024 scratch/VICReg](/work/PERSO/vmorozov/analysis/encoder_context/multimaterial256-b1024-20260925/RESULTS.md) and [Epi recovery](/work/PERSO/vmorozov/analysis/encoder_context/multimaterial256-b1024-epi-recovery-20260925/RESULTS.md) | 8+4 completed context fits cover the 12 retained treatments. Observed harmonic NLL: scratch 0.12619, VICReg 0.12657, Epi 0.12768; relaxed: 0.12255, 0.12157, 0.12125. One seed; point estimates. | Effects depend on input/readout. All 12 retained comparisons are complete across the two roots; generated reports retain obsolete planned denominators. |
| [Continuous crystal-distance results](../spatial_distance_20260926/RESULTS.md) | Five frozen-encoder readouts completed 16 epochs. Context helps proximity estimation, but reference crystal is visible at 97.5% of vector alarms at probability >0.5. | Separate existing-crystal recognition from precursor information. |
| [Birth harvest](/work/PERSO/vmorozov/analysis/nucleus_harvest/train-audit-20260926/RESULTS.md) | All 90 training sources processed; 206 candidate births, 123,626 causally eligible sampled controls. Not a released prediction benchmark. | There is a concrete starting population for a separate birth study; eligible held-out events remain to be established. |

Some older scientific collections are unavailable under their historical paths.
The [publication record](../../docs/research_results_system.md) documents this.
This audit uses available reports/receipts; it does not claim to have recomputed
all historical numbers or recovered absent raw arrays.

## What changed in the recommendation

The [literature review](../../docs/encoder_research/representation_literature_20260926.md)
correctly identifies relevant methods, but its original priority for nonlinear
kinetic learning did not sufficiently account for our earlier negative VAMP and
future-law experiments. The first priority is now a controlled attribution study.

The present code supplies a particularly useful intervention. In
[`mace_epi/objective.py`](../../src/research/mace_epi/objective.py), all three
paired objectives contain direct export alignment, `25/51 * invariance`.
Epi+variance also contains `25/51 * variance` and
`-0.1 * E / E_initial`. In the current
[`stream_pretrain.py`](../../src/research/encoder_context/stream_pretrain.py),
the two views are **same-time observed and relaxed patches**. Its Epi reference
is the initial width128 encoder, frozen and projected 128→64.
The older September23 experiment instead used **t and t+0.75 ps**, width64 and
a different width16 random reference. These are not the same experiment.

Thus Epi may be useful because it preserves informative random geometric
features, because pairing teaches useful invariance, or because of both. The
completed Epi/VICReg comparisons do not separate these explanations.

## Stage 0 — isolate readout and distance effects without new encoder training

**Question:** is apparently missing information absent, hard to decode, or poorly
organized by the chosen distance?

Reuse the frozen observed-input VICReg/Epi pretrained and fine-tuned exports,
scratch likelihood-trained export, and matched initialized MACE controls.
Historical checkpoints remain external controls when their budgets differ.
Recover an initialization only from its retained state or verified replay of
the frozen producer, including normalization; an arbitrary fresh random model
must be labeled a separate initialization rather than a matched one.
Read existing quality and context tables first; do not rerun the completed
eight-model evaluation just to create a new report.

Only add the missing comparisons:

1. For the two pretrain/fine-tune pairs, compare a small declared grid of
   likelihood-trained readout capacities with identical optimization/selection
   budgets. Include the existing head and existing linear/MLP results. A head
   that stays at constant risk needs an optimization/data diagnostic, not an
   automatic conclusion that the representation contains no information.
2. Compare `z + d32` with `z + P32(z)`, where `d32` is the existing same-patch
   geometry descriptor and `P32` a fixed train-only projection. Both inputs
   have 160 dimensions and the same downstream parameter count. This extends
   the existing add-back test with a no-new-information capacity control.
   An add-back benefit still does not prove information-theoretic absence from z.
3. Compare raw, train-standardized and matched-dimensional PCA distances on
   fitting-set neighbor forecasts and withheld physical observables. This is a
   diagnostic of metric choice, not a new learned replacement encoder. Do not
   retrain the discontinued frozen physical-map approach.
4. Include same-support smooth-density/SOAP evaluation controls if compatible
   features can be produced. Compare their actual receptive fields; do not
   import historical temperature-conditioned results as matched baselines.

The 32 descriptors' producer is
[`physical_targets`](../../src/research/encoder_context/geometry.py): radial,
count and angular powers from the supplied 80-candidate coordinates. The
[quality runner](../../src/research/encoder_quality/run.py) already exports
z-only, descriptor-only and joint readouts. Retain that exact input provenance.

**Decision:** if readout or scaling repairs the apparent deficit, prioritize
that explanation before attributing the improvement to another encoder loss.
If gains remain limited across controlled readouts, proceed to Stage 1; this
is evidence within the tested model classes, not proof of intrinsic sufficiency.

## Stage 1 — does alignment help beyond the geometric reference?

**Primary new training experiment: three treatments × three seeds = nine
encoder fits.** All use the exact same 345,600 Al observed/relaxed pairs, initial
weights, train-fitted normalization and sample order within each seed.

| Arm | Objective on raw 128D export | Isolated comparison |
| --- | --- | --- |
| R0 | Epi + variance, **alignment coefficient 0** | Geometric-reference/variance learning without forcing the two views together |
| R1 | Epi + variance, existing alignment coefficient 25/51 | R1−R0 isolates the alignment term |
| R2 | Existing paired VICReg: `(25 I + 25 V + C)/51` | R2 versus R1 compares regularization recipes at the same paired alignment |

R0 still evaluates each observed and relaxed view against its own fixed random
reference and applies the same variance calculation. It receives the same data,
not a smaller observed-only training population. Keep Epi weight 0.1 and its
initial-scale calculation fixed. Save the initialized encoder for every seed.
This controls for architecture and random geometric features without adding a
physical-reconstruction target.

Use native geometry-only MACE, width/export128, one constant atom channel,
batch/microbatch256; same nearest80 candidates, radius8, cutoff5, two spatial
blocks, no halo. Train 24 complete epochs, retaining 0/4/8/12/18/24 exports.
Use the existing learning-rate family and a common declared 24-epoch schedule;
the primary endpoint is fixed epoch24, epoch12 is a prespecified secondary
endpoint. Earlier epochs diagnose trajectories and are not promoted by onset
metrics. Validation monitors the label-free objective. Run scientific tracking
online with stable W&B IDs. Wall time must be measured before scheduling;
equal examples/updates do not imply equal FLOPs because Epi computes a reference.

The objective now accepts an explicit alignment weight. Streaming pretraining
saves configured fixed endpoints, milestone receipts, shared per-seed initial
weights/normalization and the reference projection. Historical recipes retain
their recorded12-epoch settings; the new recipe declares24 epochs.

Primary scientific contrasts are R1−R0 and R1−R2, with per-seed results for:

- Liquid-neighbor discrepancy and withheld physical-information readouts.
- Nonbulk classwise discrimination and distance-matched boundary quality.
- Noise response versus real rearrangement response at comparable displacement.
- Frozen predictive likelihood, 3/6 ps log loss/Brier and calibration.

The stronger rearrangement assay is **new**: compare same-center MD changes
with nuisance perturbations matched on displacement magnitude, and stratify by
independently measured neighbor/bond changes. Existing descriptor-change
correlation is insufficient. Specify its producer, matching, coverage and
metric contract before calculating it. Avoid dividing noise and motion scores
from different populations and calling that a signal-to-noise ratio.
Existing static Al/Ta/Zr frames remain exploratory wherever encoder ancestry or
generating potential is uncertain; spatial probe separation alone does not make
them independent encoder tests. Dynamic Al comparisons retain source-held-out
roles, and no material-transfer claim follows from a static plot.

**Decision:** R0 matching or exceeding R1 would argue that pairing is not the
source of the gain. R1 suppressing nuisance response while retaining meaningful
changes would support useful learned invariance. Improved structural readout
without improved likelihood supports a structural benefit only. Predeclare
practical margins from training/selection measurement repeatability before
examining final contrasts; lack of significance does not establish equivalence.

## Stage 2 — does supervised adaptation improve access or discard structure?

Use the **predeclared R1 epoch24** from each of the three seeds, not the treatment
with the highest test score. Compare:

| Pipeline | Encoder during likelihood training | Predictor input |
| --- | --- | --- |
| Frozen R1 | Frozen | Its current local z128 |
| Adapted R1 | Fully trainable | Same architecture and local z128 |
| Scratch | Same-seed random initialization | Same architecture and local z128 |

This is **nine likelihood fits**, six with encoder updates and three head-only.
Use the same fresh predictor initialization/order within seed, 24 epochs,
likelihood checkpoint selection from epoch12 onward, and a separate fixed
epoch24 report. After adaptation, apply identical fresh frozen probes to every
export. Hold observation support and domain fixed: observed geometry first.
Context heads and relaxed-input inference are subsequent comparisons, not
additional factors in this attribution experiment.

**Decision:** a gain confined to the joint head suggests an access/optimization
effect. Gains with fresh probes suggest useful representational adaptation.
Structural loss accompanying a predictive gain defines a measurable trade-off;
it does not make the model universally better. The existing one-seed quality
comparison motivates this experiment but does not replace its matched training.

## Stage 3 — separate interface recognition from birth information

This is a separate scientific target, not a relabeling of Al64 atom-onset rows.
The completed training harvest is the starting audit, not yet a fitting release.
Freeze operational establishment rules, ancestry handling, ambiguity, competing
arrival events and sampling probabilities before building selection/calibration/
test data. Preserve all original source roles; no split changes based on births.

Use outcome-independent origin regions/centers and retain zero-event exposure.
Future nucleus locations may assign outcomes but must not choose predictor
inputs or become coordinates. Event-enriched training needs recorded sampling
weights and evaluation on the declared natural-risk population. Keep unresolved
and censored outcomes explicit rather than silently counting them as negatives.

First count distinct events and independent sources for three observation
conditions: no confirmed crystal in the local patch; none in the complete
context footprint; no PTM-crystalline atom anywhere in that footprint. These
are different populations. If a strict group has no supported positives, report
that limitation; do not manufacture a precursor conclusion from an easier group.
The existing harvest's 8 Å exclusions do not certify a 32 Å context is clear.

Start with frozen encoders and likelihood probes: same-support geometric
descriptors, Epi and scratch representations; local and context observations
reported separately. Use a label-side crystal-visibility control only as an
explicit privileged diagnostic, never as a matched geometry-only competitor.
Compare bulk liquid, ordered-liquid candidates, competing order, liquid-side
interface and defective solid through continuous reference measurements and
uncertain labels. Do not equate the image colors to distinct phases.

**Decision:** conditional predictive benefit on crystal-clear regional births
would support precursor information. Improvement limited to visible fronts
establishes interface/arrival utility. Lack of detectable information is an
allowed result; a critical-nucleus/committor claim still needs different evidence.

## Stage 4 — a conditional kinetic diagnostic, not a new VAMP queue

Only if Stage 0 leaves a specific distance/persistence deficit, reuse
[`LinearVAMP`](../../src/temporal_vamp/linear_vamp.py) on the current continuous
MACE features as a bounded diagnostic. Compare against dimension-matched PCA
at 8/16 coordinates and lags 0.75/3/6 ps where accepted exact pairs exist.
Fit on training ancestors; select covariance regularization with label-free
held-out lagged scores. Do not select encoders or fitting rows with onset labels.

Evaluate within-liquid and interface strata separately, but preserve the
unfiltered sampling contract for fitting. Check actual future changes and
independent physical observables; retrieving similar future outputs of the same
encoder alone risks a circular conclusion. Ordinary-MD tests describe realized
future prediction. A conditional-law claim requires accepted repeated-shooting
data, parent/source grouping and branch-sampling uncertainty.

Only reproducible benefit over PCA without structural loss would motivate a
new nonlinear kinetic objective. Pooled cooling data and incomplete local
observations do not automatically justify equilibrium/reversible timescales.
The older negative result remains a baseline rather than being erased by a
new method name.

## Scientific populations and limits

[DATASETS.md](../../DATASETS.md) and the source registry were consulted; an
inventory refresh was requested for this audit. The fixed prediction contract is
Al64 release `e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d`:
90/15/15/30 train/selection/calibration/test sources and 45,291 test windows.
Keep all64 and legacy16 tracks separate. Historical test sources are reused;
neither repeated evaluation nor changing a model makes them untouched.

Stage 1 uses the paired Al subset of structural release
`0d605bdbde89f27ee40135292633325a8ffa52ba3fede61542545a4d8b7441a3`
([contract](../../docs/datasets/structural_multimaterial_256.md)). It does not
quietly claim mixed-material pretraining from that release's name. No event or
phase label chooses its samples. Calibration/test ancestors never train it.

The existing [joint distance-encoder study](../distance_encoder_20260926/README.md)
had not completed its12-epoch budget at inspection. Await that experiment's
result rather than duplicating it; its batch4096, data and target differ from
Stage1. Its eventual checkpoint is an external task-trained control, not an
isolated regularizer ablation.

The [new Al sources](../../docs/simulations/al_birth_uniform_20260926.md) are
assigned development/train, so they cannot be relabeled an untouched test.
The [Ta shooting pilot](../../docs/simulations/ta_shooting_20260926.md) represents
one conservative preparation lineage, with four shots per parent and uncertain
original generating potential; it cannot establish independent-material
replication or precise commitment probabilities. Zr's retained static structures
support structural analysis, not prospective prediction. Consult the dataset
registry for current availability rather than treating planned campaigns as
completed evidence. A confirmatory release needs separately declared independent
ancestry; simulation design and inventories remain in their linked documentation.

For every proposed run, record encoder inputs separately from predictor inputs.
Stage1's encoder sees observed/relaxed geometry during fitting; inference here
uses observed geometry. Its fixed random encoder is a training-only teacher.
Stage2's predictor sees local z128; Stage3 context extends support and must audit
the complete footprint. No stage adds temperature, simulation age, explicit
time, material/species channels or scale features. Physical descriptors are
evaluation inputs only. No physical-reconstruction pretraining or AP objective,
selection, promotion or fitted mixture weights are proposed.

## Deliverables and order

1. Publish Stage0's missing controls alongside existing evidence; no new encoder
   fits yet. In parallel, freeze Stage3's birth eligibility and exposure contract.
2. If Stage0 does not resolve the mechanism, run Stage1's nine encoder fits.
3. Run Stage2's nine likelihood fits if the adaptation question remains useful.
4. Evaluate birth information once its release has supported held-out outcomes.
5. Keep Stage4 conditional; no broad architecture, width, batch or loss-weight
   search is part of this proposal.

All new scientific fits use online `teshbek/PointCloudMaterials` tracking and
resumable local receipts. Seed uncertainty and whole-source uncertainty are
reported separately, with paired contrasts and retained unsuccessful arms.
Primary predictive selectors are likelihood-based; AP3/AP6 are diagnostic.
Structural and dynamic properties remain separate criteria, with no weighted
universal score. Select practical margins on training/selection evidence before
reading final results; more than one useful encoder may remain.

Publish each scientific comparison as a named analysis bundle with frozen metric
definitions and implementation hashes, following the
[result conventions](../../docs/research_results_system.md). New calculations
need metric contracts before implementation; historical definitions remain intact.

## Execution specification

See [active recipe](../../configs/encoder_mechanisms/alignment_readout_20260926.json),
[operations](../../docs/encoder_mechanisms.md), and [metric definitions](../../docs/metrics/encoder_mechanisms.md).
The three seeds are20260926/27/28. All three treatments use the same saved
per-seed initial weights and train-only normalization, not independently rounded
normalization estimates. Scratch adaptation uses that retained epoch0 state.

The readout grid is linear/128-unit/256-unit, with equal24-epoch NLL budgets.
Existing exact-budget fits are reused. New density26 and same-support SOAP50
controls are included. The historical initial control is the exact retained
Epi reference; no unverified VICReg initialization is claimed.

The displacement assay has960 same-center pairs over30 held-out sources, with
origin-neighbor identities retained and three equal-RMS synthetic controls per
pair. Bond crossings and local affine residuals supply independent change axes.
Synthetic Gaussian displacements are not identified with a thermal ensemble.

Stage1 and Stage2 are fixed, approved contrasts queued after Stage0 finishes;
Stage0 results will inform interpretation, not select a test-winning arm. No
practical-equivalence claim will be made without independently justified margins.
Birth-prediction fitting and the small VAMP follow-up remain evidence-gated.
The birth job audits existing held-out roles with frozen training definitions;
it does not yet create a new prediction release or start simulations.

## Submitted execution

- Real-data preflight1009653 completed successfully on node58, RTX PRO6000:
  all three256-row losses gave finite encoder gradients and634,496 parameters.
- Readout controls1009654 started on node58 with verified online W&B receipts.
- Training array1009660 queues nine seed/treatment pipelines, at most three
  GPUs, after controls. Each R1 pipeline includes the three24-epoch adaptation
  fits and independently fitted frozen probes.
- CPU audit array1009656 processes15 selection,15 calibration and30 test
  sources with the sealed training definition.

The pending original array1009655 was replaced before any encoder training:
a locked immutable identity writer prevents concurrent workers racing on one
receipt. The revised training snapshot and replacement receipt are retained
under `technical/training-revision-v2`; running controls/audits retain the
original snapshot. Scientific recipes and objective implementations did not
change. The output root is
`${storage:analysis}/encoder_mechanisms/alignment-readout-20260926`; its README
and `technical/launch.json` link execution and result bundles.
