# Predictive-memory optimization follow-up

Launched detached on 17 September 2026 under H100 allocation 995957.
All observations come from the existing immutable partial-observation release;
no trajectories or simulation data are generated.

The first 12 pilot fits did not show a reproducible history benefit. Frozen-head
interventions also showed almost no use of the state in the velocity-input
models, while current physical measurements supported better linear forecasts.
This follow-up tests whether training budget and present-information retention
limit the native encoder/predictor.

| Setting | Seed | Fits | Updates per fit | Results |
|---|---:|---:|---:|---|
| Original present weight 0.05 | 20260917 | 4 | 12,000 | [Comparison](../optimization-original-seed20260917/comparison/README.md), [diagnostics](../optimization-original-seed20260917/diagnostics/README.md) |
| Present weight 1.0 | 20260917 | 4 | 12,000 | [Comparison](../optimization-present1-seed20260917/comparison/README.md), [diagnostics](../optimization-present1-seed20260917/diagnostics/README.md) |
| Original present weight 0.05 | 20260918 | 4 | 12,000 | [Comparison](../optimization-original-seed20260918/comparison/README.md), [diagnostics](../optimization-original-seed20260918/diagnostics/README.md) |
| Present weight 1.0 | 20260918 | 4 | 12,000 | [Comparison](../optimization-present1-seed20260918/comparison/README.md), [diagnostics](../optimization-present1-seed20260918/diagnostics/README.md) |

Each group fits snapshot, 12 ps history, 48 ps history, and a separately trained
48 ps repeated-anchor control. Architecture, target packet, conditions, source
splits and checkpoint selection remain matched. Result links become available
when each group completes. Test sources remain exploratory.

The queue has 16 fits and eight collection steps. Initial estimate is 8–10 hours
from the earlier measured fit timings, subject to actual GPU throughput. The
controller stops admitting work near its 18 September 03:45 UTC deadline;
trainers retain exact-resume checkpoints before that deadline. The allocation
expires at 03:59:30 UTC.

Live status: `technical/allocation-status.json`; individual fit status and
training/validation curves live in each result's `technical/` directory.
The tracked execution stores source/configuration snapshots, Git revision and
patch, runtime environment, detached process identity, command and logs.

See the [scientific protocol](../../../experiments/predictive_memory_20260917/README.md)
and [metric definitions](../../../docs/metrics/predictive_memory.md).
