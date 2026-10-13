# Ni simulator methods diagnostics

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.ni_latent_sim.diagnostics prepare|submit|check --config configs/ni_latent_sim/methods_diagnostics_20261011.json`

This implements the first stage of the October 11 methods review: E0 immutable
comparison manifest, E2 full-path scores and scrambling control, E4 shot-resolved
response/loss diagnostics, and E3 cached-history sufficiency readouts. Source
models, sequences and MD remain unchanged. No new MD, relaxation, broad spatial
replay, AP selection, or temperature/time covariates are introduced.

The queue has a one-GPU path stage followed with `afterany` by a one-GPU response
stage, using alternative compatible partitions. The independent history stage
uses CPU. Each stage has two bounded attempts and preserves failure receipts;
path and physical re-encoding packets resume by checksum. The queue does not
depend on a chat review. Scientific neural training is not part of this first
diagnostic stage. Frozen local readouts are not new W&B scientific runs.

The frozen plan lists the selected rows/checkpoints, source roles, normalization,
native and physical time grids, kernels, perturbations and RNG policy. All K
values reuse prefixes of K=128, with three independent outer keys. Paths store
conditional means, log variances and emitted smooth24 observations; full280
endpoint scores are streamed. Physical response audits reuse the saved two-shot
300-anchor bank, re-encoding one anchor per ancestor. The history stage shifts
the origin by +1 ps to compare up to 14 exact observations without changing the
evaluation rows or requested future horizons.

Results: `output/ni_latent_sim/methods-diagnostics-20261011-v1/` (or the configured
training storage), grouped under `analyses/`; execution and checkpoint copies in
`technical/`. See [definitions](metrics/ni_motion_diagnostics.md). The launch and
stage progress files establish ownership; this workflow never stops other jobs.

E1 requires a separate full-lineage censoring ledger; the selected local panel
cannot recover missing catalogue outcomes. E5 requires a completed matched
relaxed checkpoint and separate fixed target domain. E6/E7 are partition and
seed-maturity controls. E8/E9 acquisition and altered scientific training should
address a discrepancy established by these diagnostics, with a retained baseline.
These followups are recorded in the configuration, not falsely marked running.

## October 11 launch

Launch receipt: `output/ni_latent_sim/methods-diagnostics-20261011-v1/technical/launch.json`.
Jobs 1034535 (paths), 1034536 (response) and 1034537 (history) all completed on
their first attempts. Recorded worker times were 8.28, 3.29 and 1.58 minutes,
respectively; history ran in parallel with the GPU chain. All nine path
model/key tables, 60 shot-resolved teacher packets, three response-loss tables,
and 28 fitted history readouts with held-out predictions are present. A one-anchor
GPU integration assay re-encoded both physical shots and checked the student
output shape; target/response replay RMSE was 2.99e-7/5.55e-9 on that anchor.
Those checks establish execution correctness, not final scientific conclusions.

The maintained [evidence summary](encoder_research/evidence.md#ni-methods-diagnostics)
records the results: physical response fails to beat zero; the measured variance/K
penalty is small; rare path-response excursions undermine a favorable scrambling
interpretation; history gains depend on representation and readout. The
[experiment plan](encoder_research/experiment_plan.md#execution-status) records
the unresolved support, kernel, rank and population questions.

## Six-ps and response-observation follow-up

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.ni_latent_sim.diagnostics submit --config configs/ni_latent_sim/methods_followup_20261011.json`

Five path audits use a Slurm array with at most two concurrent single-GPU workers
on alternative compatible partitions. Each model has independent two-attempt
receipts and resumable packets; a failure does not prevent other array elements
from running. A separate CPU job fits36 local response readouts: nine observation
representations, Q/V query families, and linear/state-conditioned kernels. No new
physical simulation or neural encoder training is required. Results go to
`output/ni_latent_sim/methods-followup-20261011-v1/`.

The path panel is identical to the first audit. Four 6-ps models test physical
response supervision across two training seeds. The unchanged-batch1-ps
values-only transformer seed20261014 provides the matched horizon control.
Response readouts use all300 existing teacher anchors, with original80/20/200
fit/selection/test roles, and preserve their actual Q/V perturbation queries.
They test whether information is accessible before versus after compression,
with both a linear and a state-conditioned response operator. A negative result
does not establish an information bound from only80 fitting anchors.

Follow-up launch: GPU array1034576 (five models, maximum two simultaneous GPUs)
and independent CPU job1034577. The first two workers started on node07 A100s
and saved trajectory packets. The CPU stage completed all36 readouts on its
first attempt. Receipts and frozen source are in
`output/ni_latent_sim/methods-followup-20261011-v1/technical/`.
