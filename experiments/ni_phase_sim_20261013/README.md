# Can local phase-space prediction learn useful simulator sensitivity?

The earlier measurement-only Ni forecasts had poor100fs sensitivity despite
decreasing training response loss. This experiment changes the predicted object
to actual fixed-identity atomic Q,V and tests learned latent transport separately.

For start snapshot i, physical continuation b and center j, retain the same80
atom IDs at every observation tau_k=kh, h=5fs. H_k contains their center-relative
positions and velocities. A per-atom rotation/permutation-equivariant encoder
phi maps H_k to eight vector channels per atom. A learned autonomous map G
advances that state; a learned equivariant linear decoder returns Q,V changes
anchored at the actual initial observation. No global invariant vector is asked
to determine an absolute atomic orientation. The first study uses present Q,V,
not an observed history or explicit time/temperature/box/species inputs.

Four arms, each with three initialization seeds:

1. Direct Q,V evolution (identity encoder/decoder).
2. Learned latent evolution, supervised by decoded Q,V at every5fs step.
3. The same latent model plus decoded physical response JVP supervision.
4. Arm3 plus J_G(phi(H_k))[J_phi(H_k)u_k] matching J_phi(H_(k+1))u_(k+1).

The latent map also has ordinary value consistency and joint dynamic-state
decoding supervision. Exact PyTorch JVPs and their parameter gradients are used;
simulator force tangents are analytic CUDA derivatives. Every arm includes
position/velocity kinematic consistency and autonomous multistep fitting.
The derivative ablations share architecture. The smaller direct baseline is a
useful practical control, not an equal-parameter bottleneck experiment.

The pinned diverse320 anchor bank supplies320 training states/28 source
ancestors, and all310 former nontraining anchors/62 ancestors form test.
Snapshots and centers were selected before inspecting future outcomes. The
original roles remain provenance. Test is never used for model selection or
early stopping; final checkpoint at8,000 updates is fixed in advance. Each
update uses256 examples, AdamW maxLR0.001,400 warmup updates and cosine decay.
Rollout lengths1/4/10/20 have probabilities0.2/0.2/0.3/0.3; supervision covers
every step in each rollout. Train transforms use training paths only.

Teachers are **new deterministic full-cell Ni EAM/FS NVE continuations** from
saved original Q,V and the original instantaneous box:1fs velocity Verlet,
float64 states/tangents, float32 force arithmetic. Save Q,V,acceleration and
two initial perturbation tangents at0,5,...,100fs. The teacher evolves all32k
atoms, while the network sees80. They are neither replay of native NPT nor
relabeling of previous stochastic BAOAB continuations. Force engine validation
is reused; integration energy drift is recorded per path. Perturbation RMS is
0.01Å in Q and0.1Å/ps in V, localized initially to the selected80 atoms.

Main outcomes: source-balanced Q,V errors and sensitivity versus ballistic,
persistence and zero-response baselines at5/20/50/100fs; inner16 versus all79
noncentral atoms; latent transport; BO/TDA/density/velocity measurement errors;
pair-overlap stability; paired ancestor-bootstrap differences and fixed example
plots. This tests dynamics learning, not yet long-horizon NPT nucleation skill.
Missing environmental information around crop boundary atoms remains a limitation.

Recipe: [phase_space_20261013.json](../../configs/ni_latent_sim/phase_space_20261013.json).
Execution: [workflow](../../docs/ni_phase_sim.md).
Exact scores: [metric definitions](../../docs/metrics/ni_phase_sim.md).
Results: [run](../../output/ni_phase_sim/dense-20261013-v2/).

```bash
python -m src.research.ni_phase_sim launch --config configs/ni_latent_sim/phase_space_20261013.json
```

## Completed results, October13

All12 fits reached8,000 updates and all12 evaluations completed at10:46:48 CEST,
8h39m after worker submission. All630 teachers and online W&B evaluation updates
completed, with no final collection/training/evaluation failure receipts. No
training remains queued for this campaign.

The following are source-balanced test results at100fs, averaged over three
training seeds. Test contains310 anchors from62 independent source ancestors;
all former nontraining roles remain pooled in test. Position and velocity RMSE
are the square root of the seed-averaged MSE, per Cartesian component, across
all79 noncentral atoms.

| Model | Position MSE (Å²) | Position RMSE (Å) | Velocity MSE (Å²/ps²) | Velocity RMSE (Å/ps) |
|---|---:|---:|---:|---:|
| Ballistic |0.368476|0.60702|84.049880|9.16787|
| Persistence |0.132813|0.36444|84.049880|9.16787|
| Direct Q,V |0.164145|0.40515|41.650596|6.45373|
| Latent, values only |0.080392|0.28354|29.900354|5.46812|
| Latent + physical response |0.070012|0.26460|29.339182|5.41657|
| Latent + physical response + latent transport |0.067088|0.25901|28.500430|5.33858|

Both derivative terms together reduce100fs position MSE16.55% and velocity
MSE4.68% versus the matched latent value-only arm. Physical response alone
reduces position MSE12.91%; adding latent transport reduces it another4.18%.
The position gains have negative paired ancestor-bootstrap intervals in all
three seeds for both successive additions. Velocity improvements are smaller
and not individually significant for every seed/comparison. The direct model
has fewer parameters and even loses to position persistence at100fs, so the
matched latent ablations are the cleaner evidence about derivative supervision.

For a readable sensitivity score, divide each exported normalized response MSE
by its explicit zero-response control on the same test rows, then average the
four Q/V perturbation × Q/V output ratios equally. This is a derived presentation
of `prediction.csv`; zero response is exactly1 for this presentation, unlike the
original train-normalized table columns.

| Model |50fs sensitivity error / zero|100fs sensitivity error / zero|
|---|---:|---:|
| Direct Q,V |1.5528|1.9099|
| Latent, values only |1.0141|2.0256|
| Latent + physical response |0.6282|1.0617|
| Latent + physical response + latent transport |0.5637|0.9702|

This supports substantial sensitivity learning at50fs. At100fs the full model
removes about52% of the value-only sensitivity error, but is only3% better than
zero response. Its three100fs seed scores are0.9259/0.9884/0.9963. The response
to a Q perturbation measured in future Q is still worse than zero; an overall
score slightly below1 does not establish accurate sensitivity in every channel.

Structural gains are mixed. At100fs, normalized bond-order error improves
0.6137→0.5887 and density error2.3785→1.5369 from value-only to full response.
Velocity-shell error worsens3.4174→3.8262 and flow error2.5486→3.3400. TDA
standardized errors are large (260.77→201.30, while physical-only is357.10);
these need channel/scale inspection before a broad topology-recovery claim.
No learned model has a pair below1Å at the four reported horizons. That is a
limited overlap diagnostic, not proof of physical stability beyond100fs.

Retained conclusion: dense atomic supervision plus simulator derivatives gives
measurable held-out gains, with an additional benefit from latent tangent
transport.100fs sensitivity remains weak and velocity-structure recovery is
not uniformly improved. No NPT equivalence, longer-horizon prediction or
nucleation-prediction improvement has been established by this experiment.

Frozen metrics: `analyses/<arm>/tables/{prediction,measurements,learning}.csv`;
paired intervals: `analyses/comparison/tables/paired.csv`; fixed examples:
`analyses/<arm>/plots/trajectories/example-{0,1,2,3}.png` under the run linked above.

## Frozen nucleus readouts on all ninety Ni sources

Requested October 13 after the short-trajectory results. Does the improved
short-time physical response add information about nucleus formation at longer
times? Reuse all twelve final checkpoints without representation fine-tuning.
Use all ninety source trajectories, including the completed October 12 birth
catalogue. Preserve the phase model's declared 28 train and 62 test ancestors;
do not create a new validation set. Preserve older catalogue roles separately.

The existing assay samples four founder atoms and sixteen controls for each
liquid-origin root, at exact 1/3/6 ps leads before the start of persistent 32-atom
or default 64-atom nuclei. Persistence is 1.5 ps. No neighborhood filter is
added. These are event-enriched classification assays, not the natural incidence
of nucleation, and nearby sampled atoms are not independent nuclei. Labels from
the saved NPT histories are the outcomes; the new physical NVE teachers are not
used as birth labels. Different leads are separate onset-relative tasks.

Compare initial invariant latent summaries, their neural 20/50/100 fs changes,
and those changes appended to initial BO/TDA/density/velocity measurements.
Controls are the position measurements, position plus velocity measurements,
and the train prevalence. All features use only the input observation or a
neural forecast generated from it; no saved future values enter the probe.
CatBoost settings are frozen at 600 trees, depth 4, learning rate 0.04 and L2 10,
without early stopping, test selection or AP optimization. Report AP/AUROC,
NLL/Brier, reliability bins, and paired whole-source bootstrap intervals.

Recipe: [phase_births_20261013.json](../../configs/ni_latent_sim/phase_births_20261013.json).
Exact definitions: [Ni phase birth metrics](../../docs/metrics/ni_phase_births.md).
Results: [ninety-source readouts](../../output/ni_phase_births/90sources-20261013-v1/).

```bash
python -m src.research.ni_phase_sim.births launch --config configs/ni_latent_sim/phase_births_20261013.json
```

The 216 readouts completed successfully in 5m30s. Full derivative training
improves the initial-latent 64-atom AP from 0.431/0.381/0.307 to
0.501/0.428/0.348 at 1/3/6 ps, with paired intervals excluding zero.
The structural+velocity measurement baseline is much stronger at
0.795/0.767/0.710. Adding the full-derivative forecast gives
0.796/0.782/0.711; none of these AP gains has a paired interval excluding zero.
The 32-atom assay has the same latent improvement but no gain over measurements.
See the linked result README for NLL, controls, counts and interpretation.

## Encoder and decoder with observed history

Question: does attention to actual recent Q,V improve physical trajectories,
simulator sensitivity and nucleus information beyond a larger single-frame
encoder/decoder? Four native observations at -0.3/-0.2/-0.1/0 ps follow the same
central atom and current eighty atom IDs. Both encoder and decoder have two
four-head temporal-attention blocks with rotation-invariant weights and vector
values. The nonlinear decoder attends to the observed past on every5-fs step;
future rollouts remain autonomous. Derivatives perturb the current Q,V while
holding observed history fixed, exactly matching the existing physical teacher.

Seven treatments, each with seeds20261023/24/25: original architecture with
value/full-derivative objectives; attention with one frame and value/full
objectives; attention with four frames and value/physical-response/full
objectives. Full adds latent tangent transport to physical response. The
attention variants have80304 parameters; original controls have31264.

Reuse all630 physical teachers and all19800 nucleus assay rows, retaining28
train/62 test ancestors. Exact native histories were recovered without dropping
examples; no new MD or birth filtering. Budgets remain8000 updates, batch256,
LR0.001,400-update warmup, cosine decay and final checkpoint. Microbatch128 with
two accumulation passes fits alternative GPUs. Starts are now fixed at the
anchor, so all arms have exact observed history; original architecture controls
are rerun with this same sampling. Earlier random-start results are preserved.

Evaluate physical values, measurements, overlaps and derivatives at5/20/50/100fs,
then frozen nonlinear nucleus probes for32/default64 atoms at1/3/6ps. Add a
four-frame measurement baseline alongside the unchanged present measurements.
All checkpoint/hyperparameter choices precede test evaluation. Paired source
bootstrap contrasts isolate history, capacity and derivative objectives.

Recipe: [phase_history_20261013.json](../../configs/ni_latent_sim/phase_history_20261013.json).
Definitions: [history metrics](../../docs/metrics/ni_phase_history.md).
Results: [attention comparison](../../output/ni_phase_history/attention-20261013-v1/).

```bash
python -m src.research.ni_phase_sim.history launch --config configs/ni_latent_sim/phase_history_20261013.json
```

No scientific results at protocol freeze. The real-data integration benchmark
verified nonzero gradients through both attention modules and the new readout,
rotation equivariance and agreement of exact JVPs with finite differences.


### History comparison interim, October13 19:48

Twelve of21 fits and both evaluations are complete. Across the first two seeds,
full-derivative four-frame attention improves default64-atom latent-path AP
from original-model0.443/0.394/0.333 to0.534/0.490/0.365 at1/3/6ps. Source-paired
intervals support1/3ps gains;6ps remains uncertain. Four-frame direct measurements
remain much stronger at0.803/0.813/0.719. Adding the neural path has no established
AP gain over that baseline, but small1ps NLL/Brier improvements. At100fs,
position MSE improves0.06295 to0.05901 and Q-to-Q response error1.6901 to1.0118;
the zero-response reference is1.0516, so that absolute sensitivity skill is weak.
One-frame attention/full has only one completed seed, so capacity versus history
attribution remains preliminary. All budgets/recipes remain unchanged. See
[the frozen interim tables and interpretation](../../output/ni_phase_history/attention-20261013-v1/analyses/interim-20261013-1948/README.md).
