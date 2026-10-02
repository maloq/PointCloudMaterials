# Response-atlas feasibility execution

## Local80 response training on real MD neighborhoods

[Protocol and execution](simulations/local_response_20261002.md) extend accelerated MLIP response supervision to local80 geometry, with train-only environment convergence gates and a matched optimization-time control.


## Default for new Al256 response runs

The October 2 default is [this simulator profile](../configs/simulation/response_atlas_fast_default.json):
float32 cuEquivariance, exact GPU periodic neighbors, conditional derivative graphs
with the last HVP graph released, batches of four independent replicas, and AD
value reuse with one ordinary audit per training parent. The optional Verlet skin
is zero: its measured standalone gain was negligible. Responses remain analytic
AD; finite differences are a numerical comparison, not the production default.
The potential, BAOAB splitting, 450K thermostat, timestep and20/100fs observations
are unchanged. Each replica retains its own original float64 RNG stream before
the explicit float32 cast. This setting applies only to this Al256 response case.

The existing workflow now defaults to
`configs/response_atlas/atomistic_training.json`, which points to the versioned
[fast recipe](../configs/simulation/response_atlas_training_fast_20261002.json):

```
python -m src.research.response_training.queue preflight
python -m src.research.response_training.queue submit
```

An explicit `--config` still selects a declared recipe. Historical runs should be
resumed from their original frozen source/configuration. New collection uses new
paths and the70000000 seed namespace, preserves the same parent roles/ancestry,
and records actual float32 q/box/basis. Batch checkpoints publish to STORE before
continuation; the existing locked multi-GPU adapter also supports this protocol.
Changing the default itself launches no new full scientific collection or fits.

Acquisition costs now report the actual shared training/selection bank for every
arm, including response and audit calls. They do not invent value-only acquisition
times for reused values. See [the versioned definitions](metrics/response_training_fast.md).
The full preflight checks the actual default oracle against the original float64
AD calculation, independently audits value/response execution, and retains the
student gradient/geometry checks. Scientific fits remain online in W&B; checks
and collection remain local.

Acceptance completed October 2: full100-fs default-versus-float64 and student
gradient/invariance gates passed. A105-trajectory acquisition audit covered train
(33 executions, eight reused values), selection(32) and test(40, disjoint streams).
All three recovered from STORE after removing their disposable scratch copies,
with zero new model calls. The final
[acceptance receipt](../output/response_atlas/atomistic-training-fast-20261002/technical/default-acceptance.json)
links the numerical evidence and the corrected ancestry record. These checks
created no encoder fits or full scientific collection.

## October 2: oracle acceleration benchmark

`python -m src.research.response_performance.benchmark run --config configs/analysis/response_performance_20261002.json`
runs a separate local benchmark against the completed Al256 response bank. It
freezes its source and configuration in `response_atlas/oracle-performance-20261002`.
Historical numerical producers and labels remain unchanged. The variants measure
conditional derivative graphs, GPU periodic neighbors, a Verlet skin, float64
cuEquivariance, replica batches1/4/8/16, separate float32 arithmetic and three
common-random-number finite-difference step sizes. See
[measurement definitions](metrics/response_performance.md). Unsupported or
inaccurate variants are explicitly recorded as failures; timings do not imply
acceptance. Numerical checks and hardware benchmarks remain local, without W&B.

After the sweep, `python -m src.research.response_performance.validate --config
configs/analysis/response_performance_20261002.json` repeats the reference,
conditional-gradient and fastest accepted float64 AD measurements. It also gates
the saved energy errors and exercises the resumable acquisition API against
archived streams. Repeats must run without concurrent GPU workloads. The first
October 2 reference overlapped the previously launched PaCMAP inference; its raw
measurement is retained, and isolated repeats supersede it for speedup estimates.
`python -m src.research.response_performance.publish --root RUN_ROOT` plots saved
measurements, using the isolated repeat values when available. Execution bundles,
per-variant errors, rejected variants and timings reside in the run's `technical/`
directories; these measurements concern the fixed Al256, 20/100-fs protocol only.

The opt-in next-protocol acquisition API is
`src.research.response_performance.collection.acquire_parent`. It requires a new
empty destination and a caller-supplied identity binding teacher, arithmetic,
backend, source hashes, geometry and seeds. Training response queries supply both
their values and derivatives; only remaining ordinary values are acquired. A
declared subset of shared training seeds is independently rerun as an execution
audit. Test response seeds must remain disjoint from test value seeds. Batch
payloads are saved atomically and reused on resume; changed contracts fail.

Costs record measured batch times and logical force/HVP counts. They are not
divided into fictitious per-shot acquisition times, and response-call time is not
used as a value-only counterfactual. With 32 values, 8 responses and one ordinary
audit per training parent, this protocol executes 33 trajectories instead of 40;
the separate value-only benchmark remains available. The historical training
collector and its cost schema remain on their recorded protocol. The validation
exercise replays four archived value streams and two responses with one audit
(five executions), then verifies that a restart makes zero model calls. It creates
no new scientific training run or independent simulation collection.

The isolated October 2 float32 comparison on RTX PRO 6000 Blackwell Server
Edition passed all four checked parent configurations. At replica batch4, with
cuEquivariance in both cases, ordinary-query cost fell from 7.400 to 0.807 seconds
(9.16x), and value-plus-two-response cost from 22.489 to 1.872 seconds (12.01x).
These are amortized batch costs, not single-query latency. Response peak allocated
memory fell from 4.121 to 2.094 GiB. The largest checked relative response error
against archived float64 was 3.033e-5 (0.00303%); maximum absolute response error
was 9.454e-8. This validates the declared 20/100-fs synthetic protocol, not longer
liquid trajectories. The first prioritized float32 attempt is retained separately;
the accepted measurement was repeated without concurrent GPU work. See
[the numerical receipt](../output/response_atlas/oracle-performance-20261002/analyses/benchmark-v1/technical/variants/cueq32-b4.json).

## Al-neighborhood PaCMAP transfer comparison

`python -m src.research.response_pacmap --config configs/analysis/response_pacmap_20261002.json`
compares the response-trained MACE128, the previously selected
MM-TDA-BLOCK-DIRECT-FULL, and the matched GeoFormer S1/seed17/epoch24 encoder.
Run in `pointnet-torch214` on one allocated GPU with two CPU threads and
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. This is local frozen inference, with no W&B
training runs. It uses all24,960 uniform held-out Al64 neighborhoods and9,360
label-free training landmarks for normalization and PaCMAP fitting. All model
inference imports come from the recorded frozen producer trees.

The response encoder's local80-atom adapter is an explicit transfer from its
periodic256-atom training support; it retains learned weights and normalization.
The adapter is checked against the original full-cell export before inference.
The selected response checkpoint minimizes the existing validation NLL across
the three seeds. The other two checkpoint identities are fixed in the recipe.

Output: `output/response_atlas/al64-transfer-pacmap-20261002/analyses/comparison-v2/`.
Open `index.html` for three interactive panels with structure, crystal-fraction
and source-audit colors; `plots/` contains PNG/PDF exports. The six-entry leased
feature cache protects active inference. See the
[input and projection definitions](metrics/response_pacmap.md); these maps do not
establish predictive performance or information sufficiency.

The next stage now implements [atomistic response training](#atomistic-response-training):
nine matched full-cell fits after fresh label collection. Historical feasibility
and toy bundles below remain on their original definitions.

Use conda `pointnet-torch214` and the existing project paths. Implementation is
in `src/research/response_atlas/`; no standalone script or test-suite dependency
is needed. The supplied reference module remains unchanged.

```
python -m src.research.response_atlas.queue numerical --config configs/response_atlas/feasibility_20261001.json
python -m src.research.response_atlas.queue gate --config configs/response_atlas/feasibility_20261001.json
python -m src.research.response_atlas.queue submit --config configs/response_atlas/feasibility_20261001.json
```

Submission checks the potential, shooting protocol, numerical gates and metric
contracts, then freezes code and definitions. CPU reliability/mechanism work
depends on successful completion of the existing Al480 seal. The GPU pilot waits
for that CPU stage and reruns its backend gate on the allocated device. Existing
shooting fits and diagnostics remain on their original frozen queues.

Scientific toy predictor fits use online `teshbek/PointCloudMaterials` runs with
stable IDs, local receipts, checkpoints and selection by feature likelihood.
Numerical checks, frozen descriptor readouts and response measurements stay local.
Authentication/network errors fail loudly and retain execution receipts.

Results are under `${storage:training_storage}/response_atlas/feasibility-20261001`.
Simulation query work is staged in `${storage:simulation_runs}/response-atlas-20261001`
and copied per branch/pair and on parent completion/failure to
`${storage:archive}/simulations/response-atlas-20261001`. Full-precision query
states are numerical restart artifacts, not trajectory exports. No integration
state is reconstructed from float16. The registry records the fixed potential
and shared FCC prototype ancestry.

`technical/launch.json` records scheduler IDs and dependencies. Stage receipts
record running/completed/failed states. Resume with the original frozen code and
config; completed parent bundles and per-branch response/verification caches are
checked before reuse. Scheduler termination is trapped to publish partial data.
Changed scientific producers require a new output and cache revision.

The 16 FCC configurations are a synthetic numerical pilot. They do not replace
independent liquid sources or establish crystallization prediction performance.
The five-arm active-training campaign and large-cell MEAM extension remain gated.

Measured eager float64 preflight costs were about 8.7 seconds for a five-step
value branch and 20.7 seconds with two tangent directions on RTX6000PRO. The
initial budget therefore uses 20/100 steps for all 16 parents, 500 steps for the
first two, four discovery branches, and four fresh verification branches per
direction on those two parents. This is an explicit cost-driven deviation from
the attachment's suggested 16 verification branches. It does not support a
statistical acceptance decision. L40S timings may differ; partial branches/pairs
are published for continuation if the 24-hour allocation is insufficient.

## Submitted October 1, 2026

- CPU job `1016070` waits for successful Al480 seal `1015984`.
- L40S job `1016071` waits for successful CPU completion and has a 24-hour limit.
- Frozen protocol identity: `94754448ca8ee9694a60372eba495f191834c807d35866a6c77d2bcc97ff01f2`.

The canonical combined pilot recipe resides in
`configs/simulation/response_atlas_feasibility_20261001.json`; the research config
path above is a relative symlink to the same file. Its parsed content exactly
matches the submitted frozen `technical/code/config.json`.

The original .20-A displacement proposal failed the 1.8-A pair-separation screen
for parent15 (1.69565 A). That rejected state and recipe are preserved in the
run's `technical/` directory. The final proposal uses .03/.08/.12/.16-A Gaussian
coordinate displacement standard deviations. All 16 configurations pass the
overlap, energy and force screens. Prefix RFF equality, batched student parameter
gradients, reference integration and physical finite-difference gates passed.
W&B authentication was verified before submission.

## Follow-up execution

The [follow-up](../experiments/response_atlas_20261001/FOLLOWUP.md) strengthens toy
controls and measures fixed-parent shot precision. Submit with:

```
python -m src.research.response_atlas.followup submit --config configs/simulation/response_atlas_followup_20261001.json
```

Five CPU seed workers (at most three concurrent) and an independent L40S job use
new immutable output `response_atlas/followup-20261001`. The GPU reruns physical
gates before fresh collection; failure stops it. CPU collection waits for all
five workers. Simulation archives: `simulations/response-atlas-followup-20261001`.
Recipe options parent_indices and branch_seed_base select declared geometries
and a disjoint stream namespace. Historical bundles remain immutable. Partial
branches and failure receipts are published. Completed toy arms skip on restart;
an interrupted arm restarts training under its stable online run ID, not an exact
optimizer resume.

Follow-up jobs: CPU array 1016542, collector 1016543, GPU 1016544. All 15 toy fits
and CPU collection completed; GPU gate passed. The metric exporter first exposed
a missing recipe path in the frozen bundle. The exact declared recipe was supplied
and verified by its original contract hash; no existing frozen file was changed.
Four seed workers were requeued to export their already-completed arms. The repair
receipt is `technical/packaging-repair.json`; future freezes include simulation
recipes. The collector dependency was changed to afterany only after all five stage
receipts and all 15 model receipts were complete, because Slurm retained the failed
array status. Both collectors use frozen scientific producers.

Saved-result visualization:

```
python -m src.research.response_atlas.publish --config configs/simulation/response_atlas_feasibility_20261001.json --followup-config configs/simulation/response_atlas_followup_20261001.json --destination output/response_atlas/pilot-explained-20261001/analyses/pilot-summary-v1
```

## Atomistic response training

Use `pointnet-torch214`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` and two CPU threads:

```
python -m src.research.response_training.queue preflight --config configs/response_atlas/atomistic_training_20261001.json
python -m src.research.response_training.queue submit --config configs/response_atlas/atomistic_training_20261001.json
```

The single GPU worker collects fresh32-value/8-response labels, trains values8,
responses8 and values32 at three seeds, then exports paired evaluation and plots.
Selection configurations need only32 value branches. The preflight validates the
full-cell student JVP, its encoder parameter gradients and physical100fs AD/FD.
Scientific training uses online W&B; collection and numerical gates remain local.

Source and config freeze under the new output `response_atlas/atomistic-training-20261001`.
Initial and per-branch float64 query states stage on SCRATCH and publish to
`simulations/response-atlas-training-20261001` on STORE. They are numerical restart
states, not trajectory exports. The sealed training cache uses IDS. Check collection
progress and stage receipts under `technical/`; each fit has its own `analyses/`
checkpoint, W&B receipt and progress. Each allocation is24hours. The worker can
requeue itself at allocation exhaustion at most twice, preserving the same job,
frozen code, branches, optimizer states and W&B IDs. Other scientific failures
stop explicitly. Requeue requests are recorded under `technical/requeue-*.json`.
Further manual continuation uses its frozen `technical/worker.sbatch` only after
checking that the recorded job is no longer active.

### Parallel collection on allocated GPUs

The October 1 run now uses the operational adapter
`src/research/response_execution.py`. Job1017340 was stopped with its branch-resume
signal and replaced by L40S coordinator1017386. Both GPUs in the existing node60
allocation1016443 run detached collection helpers. Its original scientific bundle,
configuration, query seeds, metric definitions and protocol identity are unchanged.
Use `technical/parallel-v1/launch.json` for current execution, rather than rerunning
the superseded single-writer script. The historical launch and stage receipts are
preserved; the old worker's signal failure records the intentional handover.

The adapter imports the original frozen collector and scopes each invocation to
one prepared parent. Exclusive shared-filesystem locks cover collection and final
scratch publication. Only progress routing and task assignment change; global
metric export and cache sealing wait for all parents and all locks. The coordinator
then runs the original nine fits and paired evaluation. Helpers never requeue the
interactive allocation; they stop at its deadline and leave saved branches for
the continuing coordinator. Collection stays local; scientific fits retain online
W&B and their original resumable IDs. Per-branch records identify the actual GPU,
so combined acquisition timings reflect this heterogeneous hardware.

To start another helper inside an allocated, idle GPU, use the recorded adapter
and frozen bundle, a unique lane name and explicit `CUDA_VISIBLE_DEVICES`:

```
python /RUN/technical/parallel-v1/response_execution.py --bundle /RUN/technical/code --lane UNIQUE_LANE
```

Inspect `technical/parallel-v1/*/status.json` and `collection-progress.json` for
each lane, and `owners/parent-*.json` for assignments. The top-level collection
receipt is a coarse coordinator update; individual lane receipts update per branch.
The adapter and its execution-only revisions have hashes and retained sources in
the launch receipt. Do not mix the legacy unguarded collector with these helpers.

Two additional L40S collection helpers were requested on October 1. Slurm rejected
the initial submission with `QOSMaxSubmitJobPerUserLimit`; no extra job was accepted.
`src/research/response_gpu_request.py --execution EXECUTION_DIR --deadline UNIX_TIME`
retries only that rejection every60seconds, records each accepted job immediately,
and skips already submitted helpers. The detached requester is bounded by the
node60 allocation deadline, stopping five minutes before it ends. It also stops
when collection is sealed. Other submission errors fail explicitly. Inspect
`technical/parallel-v1/gpu-request-status.json` for waiting/submitted/expired status
and `gpu-request-launch.json` for its PID, deadline and frozen source hash. These
helpers collect only; the existing coordinator still owns the nine sequential fits.

The [scientific protocol](../experiments/response_atlas_20261001/ATOMISTIC_TRAINING.md)
and [metric definitions](metrics/response_training.md) explain the full observation,
synthetic ancestry, fresh test streams, short horizons and cost-comparison limits.
