# Causal MACE matched pilot

Completed held-out comparison. Values below are seed means averaged equally across low-order sources;
low order means current group qbar6 < 0.30, not a phase assignment.

| Encoder | Present physical MSE | 9 ps future physical MSE | J at 0.75 ps |
| --- | ---: | ---: | ---: |
| A | 0.4729 | 0.4607 | 0.8969 |
| B | 0.4777 | 0.4608 | 0.8698 |
| C | 0.5922 | 0.5734 | 0.9880 |
| D | 0.5472 | 0.5312 | 0.8805 |
| repeated_anchor | 0.5855 | 0.5667 | 1.0068 |
| E | 0.4904 | 0.4730 | 0.9000 |

Physical errors use matched frozen nonlinear readouts and equal weight for the six target blocks.
E is a second training phase initialized from D; its total training budget is larger.
These are pilot fits, not an assumption of optimization convergence.

![Physical comparison](plots/physical-comparison.png)

| Comparison | 9 ps physical MSE difference | 95% source interval |
| --- | ---: | --- |
| B minus A | 0.0001 | [-0.0092, 0.0096] |
| C minus B | 0.1126 | [0.0727, 0.1577] |
| D minus C | -0.0422 | [-0.0735, -0.0137] |
| D minus repeated_anchor | -0.0355 | [-0.0719, -0.0082] |
| E minus D | -0.0583 | [-0.0701, -0.0468] |

Negative error differences favor the first model. Intervals resample whole sources after
averaging seeds; they do not quantify training-seed uncertainty.

The [source table](tables/sources.csv) retains every seed and source. See the
[paired differences](tables/paired-differences.csv), [event results](tables/events.csv),
[metric definitions](tables/METRICS.md), and [input identities](technical/inputs.json).

The [history-access diagnostic](tables/sufficiency.csv) compares matched predictors with
and without variable raw observed history alongside frozen z. A gain means accessible
predictive information was omitted from z; no gain does not prove sufficiency.
