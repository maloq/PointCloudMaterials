# Does initial motion strengthen simulator-response representations?

## October 6 native-velocity encoders: pooled nucleus-birth evaluation

The user requested combining original validation, calibration and test sources.
The completed `all010-pooled-v3` diagnostic evaluates 11 eligible births from
9 melt ancestors (230 matched rows). Frozen readouts were refitted using
original training data only: four training ancestors select C/iteration count
by NLL, then final fitting uses all 18 training ancestors/19 births. No pooled
labels enter fitting or probability calibration. The encoders remain frozen
with their original label-free response checkpoint selection; this is an
exploratory evaluation after earlier results were inspected.

Embedding AP with nonlinear CatBoost readouts (mean of three encoder seeds):

| Encoder | 0.1 ps lead | 0.5 ps lead | 1.0 ps lead |
|---|---:|---:|---:|
| Q values | 0.9245 | 0.9229 | 0.7401 |
| Q response | 0.9303 | 0.9169 | 0.7401 |
| QV values | 0.9238 | 0.8912 | 0.7402 |
| QV response | 0.9344 | 0.8844 | 0.7351 |
| Rich descriptors plus full-cell component statistics | 0.9980 | 0.9995 | 0.9954 |

The descriptor control has stronger spatial context. The 1:4 matched prior is
0.2. Response training and velocities do not show a consistent AP improvement.
NLL, Brier, paired uncertainty, both embedding/head spaces and linear diagnostics
are retained alongside AP; no AP-based fitting choices were made.

The exact neighborhood audit reproduced every original center rejection.
Among 70 distinct candidate births in the pooled test, the filter excluded 59:
57 had blocking components sharing future-onset atoms only, one had both shared
and disjoint components, and one had only a disjoint component. All blocking
components contained 8–42 atoms; none had 64 atoms at the observed frame.
The two disjoint components contained 8 and 13 atoms. The 13-atom fragment's
family is unresolved in the largest-component trace, so its earlier establishment
status is unknown. Atom overlap does not prove a common lineage, and geometric
proximity does not establish causal interference.

In 56 of the 59 rejected events, at least one sampled center was itself inside
a blocking component. Only 14 of 70 events fail the gate at -0.1 ps alone;
requiring the SAME center to pass at -0.1, -0.5 and -1 ps leaves 11 events.
This gate mostly removes precursor structure and unnecessarily restricts the
short-lead assay. Evaluation eligibility remains unchanged in this revision;
the audit supports a future protocol that stratifies precursor/crystal context
and determines eligibility independently for each lead. All source views,
including overlapping partial archives, retain their original identities.

[Protocol and reproduction](../../docs/response_dense_birth.md#pooled-test-and-neighborhood-audit)
and [metric definitions](../../docs/metrics/response_birth_pooled.md).
Results: `${storage:training_storage}/response_velocity/native500-f64-20261006-v1/analyses/all010-pooled-v3/analyses/`.

## Historical sampled-velocity experiments

The new full comparison uses the faster CUDA MEAM teacher at **0.1 and 0.5 ps**,
the original90/15/30 ancestors, four fixed sampled velocity fields and three
training seeds for each of the four arms. The user reauthorized longer runs.
Forces, states, tangents and response features use float32; the unchanged
backend retains scalar energy accumulation in float64. Earlier mixed-precision
targets are preserved separately and are not reused. This tests velocities and derivative supervision with more
independent training and held-out structures than the small pilot.

The earlier run is a three-hour feasibility pilot: 12/4/4 original source
ancestors, two sampled velocity fields, two noise shots per bank, four arms,
one training seed and 200 updates. The earlier full 1 ps protocol below is stopped.
Horizon definitions and all64 clustering observations remain unchanged;
pilot results cannot establish a reliable ranking or training-seed stability.

Compare geometry-only and position-plus-velocity encoders on the same MEAM
future bank, with and without derivative supervision. Initial velocity is an
observed input for QV and is marginalized for Q. Source roles remain fixed.

One1ps trajectory supplies absolute-future-plus-change features at0.1,0.5 and1ps.
Four independently sampled velocity fields per geometry each have repeated
noise realizations. Two coordinate and two velocity tangents describe local
response. All models also receive distribution-distance supervision appropriate
to their available input. No crystal loss, physical reconstruction or history.

Primary questions: whether velocities improve held-out predictive scores beyond
force-free motion; whether derivatives add information; and whether native
embeddings improve many-to-one descriptor clustering. All64 static transfer
reports both a sampled velocity and velocity-averaged embedding for QV. These
views are not measured MD velocities. No birth-prediction conclusion follows
from this pilot alone. Match parameter counts and use three training seeds.

Use validation feature NLL for selection and report all predetermined variants.
No AP/cluster optimization. Hold out entire source ancestors, never split velocity
or stochastic replicas across roles. Preserve original unrelaxed observations.

[Reproduction](../../docs/simulations/response_velocity_20261004.md) ·
[exact definitions](../../docs/metrics/response_velocity.md)

## Future-outcome similarity

All twelve float32 fits are complete. The additional diagnostic compares native embeddings and their existing nonlinear heads with noise-corrected future-feature distances on the original thirty held-out ancestors. It measures distance ranks and top-1/top-5 retrieval, excludes candidates sharing the query ancestor, includes initial-descriptor controls, and compares independent eight-shot halves. A second view averages the four fixed velocity fields per geometry. No model or metric parameter is selected on test results. [Definitions](../../docs/metrics/response_future_similarity.md).

Completed future-similarity evaluation: all twelve models contain held-out future-similarity information. Native Q-response top-5 gain at0.5ps is43.7%, versus39.5% for Q-values and51.9% for task-aligned initial descriptors. The paired difference remains uncertain across the30 sources; velocities do not improve retrieval. Existing nonlinear heads improve global distance agreement. Full tables and figures: `output/response_velocity/cuda-meam-500fs-float32-20261004-v2/analyses/future-similarity-v1/`.
