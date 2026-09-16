# Causal MACE matched pilot

Completed held-out comparison. Values below are seed means averaged equally across low-order sources;
low order means current group qbar6 < 0.30, not a phase assignment.

| Encoder | Present physical MSE | 9 ps future physical MSE | J at 0.75 ps |
| --- | ---: | ---: | ---: |
| D | 0.8797 | 1.0156 | 0.7846 |

Physical errors use matched frozen nonlinear readouts and equal weight for the six target blocks.
E is a second training phase initialized from D; its total training budget is larger.
These are pilot fits, not an assumption of optimization convergence.

![Physical comparison](plots/physical-comparison.png)

| Comparison | 9 ps physical MSE difference | 95% source interval |
| --- | ---: | --- |
| D minus D | 0.0000 | undefined |

Negative error differences favor the first model. Intervals resample whole sources after
averaging seeds; they do not quantify training-seed uncertainty.

The [source table](tables/sources.csv) retains every seed and source. See the
[paired differences](tables/paired-differences.csv), [event results](tables/events.csv),
[metric definitions](tables/METRICS.md), and [input identities](technical/inputs.json).

The [history-access diagnostic](tables/sufficiency.csv) compares matched predictors with
and without variable raw observed history alongside frozen z. A gain means accessible
predictive information was omitted from z; no gain does not prove sufficiency.
