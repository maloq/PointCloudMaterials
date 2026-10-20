# Research guide

The [Ni validation follow-up](../../experiments/ni_latent_validation_20261018/README.md)
adds source expansion, longer budgets, scalar-support trajectories and native
temporal-objective tests to the October 18 comparison.

- [Dense atomic Q,V and latent tangent experiment](../../experiments/ni_phase_sim_20261013/README.md): October13,12 fixed-budget fits on new100fs full-cell Ni NVE paths;320 train and310 pooled test anchors.

The project studies what local geometric representations retain about structure, physical response and future crystallization. Read the evidence before choosing a model change, and keep the observation, target population and evaluation protocol explicit.

**Review:** supplied documentation/methods review of 11 October 2026, based on repository `05e3e8ca57f42ea95f63f7f11c0aa05426fa5896`, reconciled with the current workspace and completed diagnostic receipts on the same date. The supplied literature reading-depth labels retain their provenance; integration does not constitute a new paper review or external-code reproduction.

## Main reading path

| Question | Page |
| --- | --- |
| What have we learned across the many reports? | [Evidence summary](evidence.md) |
| How should we resolve the outstanding hypotheses? | [Methods and open hypotheses](methods.md) |
| What experiments come next, and when should a branch stop? | [Experiment plan](experiment_plan.md) |
| What does related research actually support? | [Literature and inspected implementations](literature.md) |
| Which recent methods fit the current Ni latent simulator? | [October 18 primary-source review and shortlist](latent_simulation_literature_20261018.md) |
| How are those methods being tested? | [October 18 frozen Ni trajectory comparison](../../experiments/ni_latent_flow_20261018/README.md) |
| Which comparison and input rules apply? | [Research principles](principles.md) |

The evidence summary distinguishes motif benchmarks, representation retention, memory, alignment/partition behavior, birth/fate targets, simulator response and Ni dynamics. The methods page keeps the H1–H8/S1–S6 identifiers; the plan maps them to E0–E9. These identifiers connect definitions and decisions, rather than creating a separate report for each question.

## Interpretation of status

| Status in these pages | Meaning |
| --- | --- |
| Recorded result | A result reported under the linked source's population, recipe and calculation |
| Code-verified behavior | An inspected producer, loss or evaluation definition; its effect on results may remain unknown |
| Analytical deduction | A stated mathematical consequence under explicit assumptions |
| Hypothesis or candidate | A possible explanation or method awaiting a discriminating test |
| Proposed experiment | A design; implementation and execution are not implied |
| Historical or retired | An earlier protocol, decision or treatment retained with its original evidence |

The bounded Ni manifest, path, history and response-loss audits have completed: see [current findings](evidence.md#ni-methods-diagnostics) and [execution coverage](experiment_plan.md#execution-status). Population/domain audits and the remaining response-compatibility questions are still open. Completion of those bounded audits does not mean the full E0–E9 plan has run, and their new metric revision does not alter historical scores.

The [October 12 continuation plan](experiment_plan.md#ni-followup-20261012) incorporates the completed transformer/readout follow-up and all six direct-measurement runs. It prioritizes response-change fitting, data dependence, matched physical teachers and common-origin history comparisons. All15 overnight fits and evaluations,78 response readouts and3 history readouts are now complete; see [results and limits](evidence.md#measurement-overnight-20261012).

## Specialist references

| Reference | Scope |
| --- | --- |
| [Encoder families](encoders.md) | Architecture history, trained/exported spaces and retained feature stages |
| [Prediction context](prediction_context.md) | Separate encoder and predictor inputs; support, history, motion, relaxation and teachers |
| [Evaluation reference](evaluation.md) | Existing structural, geometric, dynamic and predictive measurements |
| [Finding and using analyses](analysis.md) | Locate, compare and visualize saved evidence |
| [Results database](database.md) · [Study catalogue](studies.md) | Original result locations and searchable provenance |
| [Dataset registry](../../DATASETS.md) · [Metric contracts](../metrics/) | Exact data/calculation identities, availability and historical definitions |

Read specialist architecture and evaluation records with their stated dates. The maintained [methods page](methods.md) supplies the cross-study qualifications added in this review.

## Recent protocol entry points

[Ni latent dynamics](../../experiments/ni_latent_sim_20261010/README.md) · [Completed Ni methods diagnostics](../ni_motion_diagnostics.md) · [Direct measurement-space dynamics](../measurement_sim.md) · [Ni physical response](../../experiments/ni_response_20261009/README.md) · [Structural-response and appearance tasks](../../experiments/structural_response_20261007/README.md) · [Continuous Al motion](../../experiments/continuous_motion_20261006/README.md) · [All-Al scope](../al_research_pool.md)

For the broader scientific portfolio, use [experiments](../../experiments/README.md). Procedures and operations start in the [documentation guide](../README.md).

<details>
<summary>Older reports, dated literature and superseded plans</summary>

The [archive/source index](../archive/README.md) preserves the route to detailed records. Historical AP-directed or reconstruction-based treatments retain those labels; they do not become new recommendations through inclusion in a summary.

</details>

## Maintaining the guide

Put a new conclusion in the relevant evidence section and link its exact protocol and result. Put a new discriminating question in methods and its gated execution in the plan. Add literature only when its claim, evidence level or implementation detail changes that reasoning. Keep execution logs and machine-readable evidence in the established result layout. A new dated report needs a distinct protocol or provenance purpose.
