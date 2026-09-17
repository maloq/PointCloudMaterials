# Does early geometry–velocity–history interaction improve a local predictive state?

**Results update, 17 September:** the three-seed pilot and longer H100 comparison
are complete. The [consolidated report](../../output/predictive_memory/research-summary-20260917/RESULTS.md)
also incorporates user-reported H200 width results, Gaussian-head diagnostics and
the separate newer partial-observation protocol. The older status statements
below describe the original launch; use the dated report for completed findings.

Train one native tensor MACE encoder whose spatial messages receive temporally
informed atom features before a single multiscale pooling stage. Reconstruct
fixed present physical targets and predict fixed physical futures from the
exported state. This tests information retained about local evolution, rather
than defining representation quality by small latent increments.

The architecture is initialized from scratch and retains scalar, vector and
rank-two channels through both MACE product blocks. It does not reuse the old
scalar MLIP weights or a frozen-feature map. Positions and velocities are relative
to the tracked atom; history is causal and preserves producer atom IDs. Exact
space-time ancestors supply spatial computation context. The current-state
output is 128-dimensional in the main recipe.

Use [`configs/mace_causal/pilot.json`](../../configs/mace_causal/pilot.json) with
the existing `src.research.mace_velocity` entry point:

```bash
conda run -n pointnet python -m src.research.mace_velocity causal-prepare --config configs/mace_causal/pilot.json
conda run -n pointnet python -m src.research.mace_velocity causal-train --config configs/mace_causal/pilot.json --variant D --device cuda:0
conda run -n pointnet python -m src.research.mace_velocity causal-probe --config configs/mace_causal/pilot.json --variant D --device cuda:0
```

| Variant | Training question |
| --- | --- |
| A | Snapshot geometry, present-information loss only |
| B | A plus fixed multi-horizon future physical supervision |
| C | B plus center-relative velocities inside spatial messages |
| D | C plus interleaved causal atom attention |
| repeated_anchor | D trained separately with repeated current observations |
| E | Continue D with slowness, subject to per-task validation information constraints |

Primary comparisons A–D and the repeated-anchor control hold source splits,
anchors, parameter shapes, initialization and update budgets fixed. E is an
additional training phase, not an equal-total-budget substitute. Context radii,
hidden tensor width and deterministic/Gaussian future heads are separate config
ablations. Checkpoint selection never uses test sources.

The present target has 169 columns: group structure, H0/H1/H2 topology and motion.
Future targets include those physical endpoints, changes and intervening qbar6
path summaries at 0.75, 3 and 9 ps. Local sustained-onset hazards account for
confirmation and right censoring; labels cannot enter the encoder. The explicit
6.75 ps follow-up extension reuses existing trajectory frames and the same label
producer, retaining original labels unchanged. There is no transition oversampling.

Evaluation includes within-low-order physical errors, source-resampled uncertainty,
physical persistence baselines, declared-lag jump distributions, Gaussian NLL and
coverage, and validation-calibrated local alarms with missed-event timing penalties.
Linear/nonlinear frozen readouts test accessibility. A matched diagnostic predictor
with access to raw history alongside z tests whether predictive information was
discarded. Its constant-history control has the same trainable architecture.

Status: implementation and small end-to-end checks completed; no comparative
scientific advantage has been established. The six-update, six-source smoke is
execution evidence only. A budgeted multi-seed study, a broader selection of
anchors around transitions, responsiveness/decoder-sensitivity assays, and a
separate audit of the unchanged hard topology boundaries remain research work.
The 0.10 jump criterion is reported at 0.75 ps within the declared low-order
population; no committor interpretation or automatic scientific success is assumed.

See the [workflow and validation record](../../docs/mace_causal.md),
[metric definitions](../../docs/metrics/mace_causal.md), and
[causal-input terminology](../../docs/research_glossary.md#observed-history-and-causal-input).


## Three-seed pilot, 16 September 2026

The active pilot uses all 150 retained independent Al preparations: 90 training,
30 validation and 30 test sources, across 400/450/500/510/520 K. Four centers and
three anchors per source produce 1,800 examples. Each initialization
(20260916/20260917/20260918) fits A–D, the repeated-anchor control, and E with
16 channels per irrep and a 128-dimensional exported state. Primary fits use
1,000 updates of two source-sampled windows; E adds the same update budget to
its selected D initialization. Every encoder gets the same four diagnostic
readout fits, each with 500 updates. This is an exploratory training budget,
not an assumption of convergence.

The comparison recipe is [`comparison.json`](../../configs/mace_causal/comparison.json).
It uses fixed physical errors and exact held-out target pairing. Seed means are
formed within source before 2,000 whole-source bootstrap draws. These intervals
quantify source uncertainty conditional on the fitted seeds, not training-seed
uncertainty. The raw per-seed/source rows are retained. Joint A future/hazard
outputs are excluded because those heads received no predictive supervision;
matched frozen probes provide the A forecasting comparison.

Run state and logs: [`pilot-20260916`](../../output/mace_causal/pilot-20260916/).
Comparative conclusions remain pending completion of the full suite.

A separate D Gaussian-head ablation uses the same data and 1,000-update budget
for all three seeds (`pilot-gaussian-seed*.json`). Its endpoint MSE, NLL and
one-standard-deviation coverage are exported per source; the Gaussian objective
and additional variance outputs are kept separate from the deterministic suite.


The completed data audit contains 1,080/360/360 train/validation/test windows.
The low-order subset contains 373/105/190 windows from 35/10/17 sources.
Sustained future onset labels cover 10/6/1 distinct centers from 10/5/1 sources;
the two positive test windows overlap around the same local event. Consequently,
this pilot cannot establish general onset-detection or timing skill. Event scores
are descriptive only; a broader fixed-time/transition-aware anchor protocol is
needed for that separate question. The data population was not changed after
this audit. See [audit counts](../../output/mace_causal/pilot-20260916/technical/data-audit.json).

## Longer matched history comparison

The next fixed-data cohort asks whether history helps after more optimization:
C, D and repeated_anchor at width 16, 5,000 updates and eight source-sampled
examples per update (40,000 sampled windows per fit). It starts with seed
20260916; a single seed does not establish replication. See
`configs/mace_causal/h100-packed/width16-seed20260916.json`. All three use the same
numerically verified packed implementation. The H200 recipes extend this design
to widths 16 and 32 and three seeds, retaining output dimension 128.

Linear/nonlinear frozen readouts run for every encoder with 2,000 updates. The
more expensive matched history-access diagnostic is predeclared only for D;
no state-sufficiency claim is made for the other variants. The comparison recipe
requires all common readouts and both members of D's diagnostic pair. Physical
scores, validation-only selection and whole-source uncertainty remain unchanged.
Training-budget and effective-batch differences from the original pilot must not
be attributed to history or hardware. Compare C/D/repeated_anchor within cohort.
This cohort is running; scientific outcome comparisons remain pending.
