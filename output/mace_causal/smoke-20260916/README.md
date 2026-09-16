# Causal MACE implementation validation — 16 September 2026

Baseline before implementation: commit `69500e5`. This is a correctness and execution
smoke, not a trained-model benchmark or evidence of improved physical prediction.

The final `pointnet` regression run passed **48 tests**. Coverage includes rotation,
reflection, boost and atom-ID invariance; causal attention and exact graph pruning;
smooth spatial/age boundaries; physical-time matching; censored hazards and missed
alarms; train-only normalization; exact interrupted optimizer resume; encoder reload;
and frozen-state/history-sufficiency probes. A synthetic expansion/contraction test
learns opposite futures at identical current geometry.

The H100 run uses six independent Al preparations (two each train/validation/test),
four tracked centers each and three anchors each: **72 windows**. The requested
future lags are **0.75, 3 and 9 ps**, with a 2.25 ps causal history envelope. Additional
follow-up uses the retained source trajectories and unchanged physical producers.
All six variants completed six updates; E adds six updates after D. The four probe
modes completed two updates each. A separate Gaussian D also completed six updates.
Strict inference reload and exact optimizer resume passed in the regression suite.

| Fit | Run output | Completion |
| --- | --- | --- |
| A | [tables](../smoke-20260916-A/tables/test.csv) | complete |
| B | [tables](../smoke-20260916-B/tables/test.csv) | complete |
| C | [tables](../smoke-20260916-C/tables/test.csv) | complete |
| D | [tables](../smoke-20260916-D/tables/test.csv) | complete |
| repeated_anchor | [tables](../smoke-20260916-repeated_anchor/tables/test.csv) | complete |
| E | [tables](../smoke-20260916-E/tables/test.csv) | complete |

The smoke architecture has four channels per irrep and a 16-dimensional state.
A separate real-input check verified the default 16-channel/128-dimensional encoder
on four frames, 1,449 atoms per frame and 138,598 spatial edges. All tested spatial,
temporal and velocity modules received finite, nonzero gradients. The producer
replay matched all 676 retained physical target values exactly.

No fit reached the declared low-order normalized RMS jump threshold 0.10 at 0.75 ps.
D's test value was approximately 0.785; E's was 0.791. These tiny-budget values are
not grounds to choose a scientific model. A's direct future heads are untrained;
use matched frozen probes for a fair representation comparison. No advantage or
state-sufficiency conclusion is drawn from the probe smoke.

Reproduction and evidence:

- [Resolved run recipe](technical/config.json), [completion](technical/completion.json), and [test log](technical/tests.log).
- [128-dimensional check](technical/architecture-validation.json), [target replay](technical/label-overlap.json), and [environment](technical/environment.json).
- [Main workflow](../../../docs/mace_causal.md) and [research protocol](../../../experiments/mace_causal_20260916/README.md).

Training cache, paired predictions, checkpoints and exact source snapshots are
retained at their recorded locations. Metric CSVs carry frozen definitions and
implementation hashes. Earlier development smoke outputs are preserved separately.
