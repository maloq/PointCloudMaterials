# Does measurement supervision make learned Ni dynamics structurally useful?

Mathematical formulation, including the proposed evolving-memory extension:
[LaTeX](method.tex) ·
[PDF](../../output/ni_phase_measurements/observable-20261014-v1/analyses/formalization-20261018/data/method.pdf).
The frozen experiments and results below retain their original architecture.

The previous history model improved short atomic forecasts, but actual structural
and velocity measurements remained much stronger nucleus predictors than its
pooled latent features. This study separates inadequate readout from inadequate
representation, and tests whether measured values and their physical derivatives
make latent trajectories more useful.

For snapshot i, continuation b and central atom j, write H_ibj(tau) for the
positions and velocities of the same 80 atoms, relative to atom j. The elapsed
time tau is measured from the input snapshot; it is not an additional model
input. The observed history contains H at -0.3, -0.2, -0.1 and 0 ps. The model
encodes this history into atom-resolved vector features z, evolves z in 5 fs
steps, and decodes Q,V changes anchored at the actual input. Both encoder and
Q,V decoder attend to the fixed observed past throughout the autonomous rollout.

Define Y_ibj(tau) as the 244 measured values: bond order (40), TDA (113), smooth
density (74), and velocity measurements (17). A new invariant attention decoder
g(z) predicts Y from the atom-resolved latent state. Its pair correlations retain
information that a global average can discard. Actual Y is a training target;
it is never fed into the neural trajectory predictor. Direct-Y nucleus probes
remain explicit evaluation controls. No temperature, absolute time, age, box,
species or material label enters a neural model or probe.

## Frozen comparisons

Four treatments, each with seeds 20261023, 20261024 and 20261025:

| Treatment | Measurement supervision | Latent derivative supervision |
|---|---|---|
| present | Present Y only | First 5 fs transition |
| future | Present and every future Y, plus connected 5 fs increments | First transition |
| observable | Same as future, plus the physical responses of density and velocity measurements | First transition |
| transport | Same values/increments as future | Four separated transitions per update, cycling across all twenty |

All arms retain physical Q,V trajectory, reconstruction, kinematic, first-step
latent-value and full-rollout physical-response losses. Only the listed terms
differ. Observable response compares the neural JVP with
D_Y(H(tau)) u(tau), where u is the simulator tangent resulting from the initial
position or velocity perturbation. BO/TDA receive value supervision only.
Transport compares one learned transition applied to the encoded physical
tangent with the encoded next physical tangent. Past observations stay fixed
during these derivative queries, matching the teacher perturbation protocol.

Each model has 205,869 parameters and trains from scratch for 8,000 updates.
Batch 256 is accumulated from two microbatches of 128. AdamW uses maximum LR
0.001, 400 warmup updates, cosine decay, weight decay 1e-5 and gradient norm
clipping at 10. Sample rollout lengths 1/4/10/20 with probabilities
0.2/0.2/0.3/0.3. Select the final fixed-budget checkpoint only.

Present/future/increment/observable-response weights are 0.25/0.25/0.1/0.25.
Measurement losses balance the four groups equally, after training-only
standardization; response loss balances the two differentiable groups.
All exact normalizers, physical-loss weights and metric definitions are in
[the frozen recipe](../../configs/ni_latent_sim/phase_measurements_20261014.json)
and [metric contract](../../docs/metrics/ni_phase_measurements.md).

## Data and evaluation

Reuse all 630 sealed Ni EAM/FS NVE teachers: 320 train anchors from 28 ancestors,
310 test anchors from 62 ancestors. Each physical teacher uses the original
full cell and box, float64 integration/tangents, float32 force arithmetic,
1 fs integration and 5 fs saved states through 100 fs. Measurement targets and
their analytic smooth responses are calculated in float64 from these states;
neural computation remains FP32. No new MD is generated. The four-frame input
history uses exact native 0.1 ps observations from the original saved MD.

All former nontraining roles are test, with original roles retained as
provenance. There is no inner selection set, test tuning, calibration, early
stopping or AP objective. The broad spatial dataset is not used.

Report physical Q,V accuracy and response, decoded measurements and increments,
measurement-response error relative to predicting zero, overlaps and paired
whole-ancestor intervals. Keep measurements calculated from predicted atomic
coordinates separate from the learned measurement-head outputs: agreement
between these two decoders is not guaranteed by this objective.

Frozen CatBoost readouts retain the same 19,800 nucleus-assay rows (19,652 unique
states), 1/3/6 ps leads, 32-atom and default 64-atom nuclei with 1.5 ps persistence.
No new neighborhood filtering. Compare initial latent summaries, latent paths,
decoded Y, decoded Y paths, and their additions to actual measured history.
Report NLL, Brier, calibration, AP and AUROC; AP does not select any model.
The assay has 20% case-control prevalence, not natural nucleation incidence.

A separate frozen diagnostic uses the three parent full-history models. Fit
present-Y readouts from pooled latent features, full atom-resolved features and
raw Q,V using a fixed 2,000-update budget. These heads receive no birth labels;
their predictions then receive the same frozen birth probes. Success of the
full readout would support a pooling bottleneck. A failure is informative only
if the raw-state control establishes that the readout can learn these targets.

## Execution and interpretation

The queue has one absolute 12-hour deadline, six GPU lanes, checkpoints every
50 updates, bounded recovery and independent evaluation stages. A shorter
allocation may hand off within that same deadline. Partial fits remain partial;
they do not replace final checkpoints in the matched comparison. Scientific
training uses online W&B; frozen diagnostics and probes stay local.

Local integration checks cover all four losses, head invariances, finite
differences versus JVPs, feature extraction, evaluation output and checkpoint
resume. On RTX PRO 6000, full 100 fs updates take approximately 2.5–4.3 seconds
and peak allocated memory is 19–38.5 GiB. These are timing estimates, not outcome
evidence. Queue delays and GPU type affect completion within the cap.

Results: [observable-20261014-v1](../../output/ni_phase_measurements/observable-20261014-v1/).
Operations and reproduction: [workflow](../../docs/ni_phase_sim.md#measurement-supervision-workflow).

## Completed results, reviewed October 15

All twelve models reached 8,000 updates; all twelve physical/measurement and
nucleus evaluations and all three frozen diagnostics completed. The queue
finished October 14 at 10:26:59 CEST, 10h39m after launch and before the 12-hour
deadline. One training segment paused at its allocation boundary and resumed;
there were no failed stages. Original launch verification remains provenance.

Default 64-atom nucleus AP below is averaged over three model seeds. The assay
contains 81 held-out nucleus roots from 38 ancestors, not four test examples.
Latent-path features contain the current invariant summary and neural changes
at 20/50/100 fs; the 1/3/6 ps columns are nucleus prediction leads, not simulated
trajectory lengths. Former validation/calibration roles are included in test.

| Features / treatment | 1 ps | 3 ps | 6 ps |
|---|---:|---:|---:|
| Previous full-history latent path | 0.530 | 0.486 | 0.376 |
| Present-measurement treatment, latent path | 0.519 | 0.482 | 0.398 |
| Future-measurement treatment, latent path | 0.502 | 0.490 | 0.394 |
| Observable-response treatment, latent path | 0.561 | 0.514 | 0.434 |
| Multitransition transport treatment, latent path | 0.472 | 0.422 | 0.354 |
| Actual measured input history | 0.803 | 0.813 | 0.719 |
| Actual measured history + observable-response latent path | 0.814 | 0.818 | 0.721 |

Observable response improves over its matched future-value treatment: AP
differences are +0.0596/+0.0240/+0.0398 with paired whole-source 95% intervals
[0.0295,0.0880]/[0.0032,0.0439]/[0.0173,0.0621]. NLL and Brier also improve at
all three leads. These are the frozen descriptive intervals, without a
multiple-comparison correction. The comparison with the previous campaign is
descriptive; its matched within-campaign control is the future treatment.

Adding the observable-response latent path to actual measured history gives
only +0.0103/+0.0051/+0.0014 AP, and all three intervals include zero. NLL falls
from 0.2210/0.2454/0.3188 to 0.2149/0.2396/0.3123; only the 3 ps interval
excludes zero for that treatment. Some other treatments also improve proper
scores, so this is not evidence that the new measurement-derivative term is
necessary for those small complementary gains. For 32-atom nuclei, adding this
path to history reduces 6 ps AP from 0.7100 to 0.6929 (paired difference
[-0.0324,-0.0014]); the gain is not consistent across tasks.

The learned measurement decoder improves substantially with observable-response
supervision, but generalization remains weak. At 100 fs its group-standardized
errors change from 1.005 to 0.644 (BO), 1.311 to 1.084 (TDA), 1.151 to 0.918
(density), and 1.180 to 0.931 (velocity), relative to future values alone.
All four paired error differences exclude zero in every seed. However, the
mean smooth-response error/zero-response error is still 0.987 at 100 fs,
versus 2.337 without observable-response training (equal mean of the four
Q/V-perturbation × density/velocity ratios). Removing excessive sensitivity
does not establish accurate response prediction when the ratio remains near 1.

For the present-only treatment, the final 1,000-update mean present-measurement
training loss is about 0.051, while the source-balanced test group-mean error
is 1.651. This supports a serious generalization problem on the 320-anchor
training bank; a low training loss is not evidence of structural understanding.

Atomic forecasting does not improve uniformly. At 100 fs the previous model's
Q/V MSE is 0.06087 Å² / 33.53 Å² ps^-2; observable-response gives
0.05969 / 35.92, and multitransition transport gives 0.05831 / 30.80. The latter
improves atomic velocity prediction while weakening nucleus readouts. Its
Q-error improvement over the matched future treatment varies across seeds.
Physical response error/zero averages 0.762 previously, 0.771 for observable
response and 0.787 for transport: no improved 100 fs physical sensitivity here.

The frozen representation diagnostic does not support pooling as the sole
bottleneck. At 1 ps its reconstructed-measurement nucleus AP is 0.645 from
pooled latents, 0.612 from full atom-resolved latents and 0.409 from raw Q,V.
The raw-state head is itself weak, so failure of the full latent head cannot
establish that the encoder discarded the information. This diagnostic used
6,804 unique training input states, versus 320 physical teacher anchors for
the end-to-end motion models; its scores are not a matched training-size
comparison with the twelve motion fits.

Retained conclusion: smooth observable-response supervision helps the learned
representation under this recipe, but does not establish better AP than actual
measurement history or accurate 100 fs measurement sensitivity. Generalization
of measurement decoding and useful long-time information remain unresolved.

Sources: the completed run's `analyses/comparison/tables/birth_summary.csv`,
`birth_paired.csv`, `physical_paired.csv`, `measurement_paired.csv`, individual
`measurement_decoder.csv`, `prediction.csv`, `learning.csv`, and
`diagnostics/seed-*/tables/reconstruction.csv`. Definitions and implementation
hashes remain frozen alongside those exports.
