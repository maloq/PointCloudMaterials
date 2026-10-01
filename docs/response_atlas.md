# Response-atlas feasibility execution

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
