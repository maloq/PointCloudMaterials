# Shooting futures and response learning

One observed configuration can generate many future trajectories when velocities
and thermostat draws change. The original shooting campaigns let us estimate
this conditional spread. A response label asks a second question: how does that
future distribution change after a small change in the initial coordinates?
Individual path sensitivity can be large even when the distribution is unchanged;
the analytic cancellation experiment demonstrates why branch-noise correction matters.

## Completed pilot

![Pilot evidence](analyses/pilot-summary-v1/plots/pilot-evidence.png)

- **A: Existing Al shots.** Current geometry predicts future structural features
  better than a training prior. The clear-liquid corrected error falls from
  0.0552 to 0.0322 (42%). Lines are the six historical held-out sources.
  This does not establish birth-time prediction or prospective generalization.
- **B: Toy supervised learning.** Response labels improved feature and derivative
  prediction with the same eight-shot data. The original comparison used only
  250 updates and was not equal cost; stronger follow-up controls are below.
- **C: Cancellation.** The true law response is zero although individual paths
  respond. Naive squared branch derivatives report a false positive average;
  cross-branch correction removes the variance contribution in expectation.
- **D: Atomistic feasibility.** Actual fixed-MACE coordinate and path derivatives
  passed numerical gates. Four shots can be too noisy near small responses.
  These 16 perturbed-FCC states are development configurations, not independent melts.

The interactive shooting examples use actual crystalline fractions of up to 80
neighbors within 8 angstrom around the same central atom at future observations.
Training observations at minimum/median/maximum 12-ps empirical variance were
chosen for illustration. These examples are deliberately selected, not typical
states or calibrated probabilities. Original values and selection provenance are
in `analyses/pilot-summary-v1/technical/shooting-examples.json`.

## Completed follow-up and recovered comparison

![Completed comparisons](analyses/pilot-summary-v1/plots/completed-followups.png)

The stronger toy comparison has five paired initializations, shared 64-shot
selection labels and 4096 fresh analytic evaluation points. At 45 seconds of CPU
label-generation plus training cost:

| Labels | Future-feature MSE | Derivative MSE |
|---|---:|---:|
| 8 value shots | 0.027705 | 0.218733 |
| 32 value shots | 0.011129 | 0.134789 |
| 8 value + response shots | 0.006711 | 0.050591 |

Responses improve both errors in all five seeds against the strengthened 32-shot
baseline: 40% lower value error and 62% lower derivative error on average. The
15-second checkpoints select the same models. Initialization spread is not
independent-dataset uncertainty. Short CPU timings include contention/implementation
costs and do not establish atomistic cost effectiveness or an active-query benefit.

All 27 original shooting readout fits and their comparison are complete. In clear
liquid observations the frozen MACE VICReg path NLL is 30.849 versus prior 31.718;
paired source-bootstrap difference is -0.869, interval [-1.528,-0.266]. MACE Epi,
bond-order and topology controls also improve the prior. These are separate
proper-density readouts from the pilot's ridge/RFF diagnostic. Their uncertainty
against the prior does not establish superiority over each other. No AP selection.

The ongoing atomistic follow-up uses four deliberately selected existing states,
32 new AD branches per state and 16 fresh CRN pairs per direction, at 20/100 fs.
It has passed its numerical gates. No atomistic learner is fitted by this study.

Plots render saved tables; original metric definitions and file hashes are linked
in the rendering receipts. They do not replace the historical scientific exports.
