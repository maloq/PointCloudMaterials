# Dissolved-embryo and random-liquid restart controls

Use conda `pointnet-torch214` and
`configs/simulation/nucleus_controls_20261005.json`:

```bash
python -m src.research.nucleus_controls.queue prepare --config configs/simulation/nucleus_controls_20261005.json
python -m src.research.nucleus_controls.queue submit --config configs/simulation/nucleus_controls_20261005.json
python -m src.research.nucleus_controls.queue summarize --config configs/simulation/nucleus_controls_20261005.json
```

The original 150-source fate catalogue yields 12 eligible independent dissolved
embryos after the declared liquid-cell and 12 ps follow-up restrictions. Each
is paired with an establishing precursor at the same temperature and within two
initial cluster atoms, and a random eligible liquid frame. Future labels are not
used to choose liquid frames. All source roles and ancestry remain audit metadata;
this campaign trains no model and creates no W&B run.

Each full 70,304-atom cell gets one source-temperature native NPT continuation
and four response-style 450 K BAOAB continuations. Every branch lasts 12 ps,
with 3/6/12 ps endpoints at exact native 0.75 ps sampling. Integration, PTM,
precise checkpointing and verified float16 trajectory publication reuse the
existing recovery workflow. The established FastAl implementation is used without
new force-validation gates. [Metric definitions](../metrics/nucleus_controls.md).

`submit` freezes code and records ownership in `technical/submission.json`.
It requests two GPUs in RTX6000PRO or L40S, eight CPUs and 48 GB RAM. Workers
detect the allocated CUDA architecture and use separate compilation caches.
Other campaigns, including the existing node60 recovery workers, are unchanged.

The **shared hard deadline is 16 hours from submission, including queue wait**.
The scheduler request has an eight-hour allocation and that absolute deadline.
Workers checkpoint with 15 minutes reserved; the supervisor bounds only its own
recorded child process groups and publishes final coverage. No automatic budget
extension or requeue occurs. The exact Paris deadline is in `technical/budget.json`.
An unfinished campaign is reported as partial, never as a full negative result.

On October 5, the first allocation ended with 19 published continuations after
two verified conversions failed during NFS source retirement. Their already
computed results were recovered without rerunning MD, bringing coverage to 21.
MPI/UCX error logging continued during failed worker finalization; diagnostic
head/tail excerpts and original log sizes are preserved under
`technical/incident-20261005`, replacing the repeated log flood.

The user authorized continuation on node60 allocation 1022218. The detached
receipt is `technical/resume-20261005/launch.json`: GPU 1 continues controls;
GPU 0 finishes the earlier recovery queue, then joins controls. The original
16:59:52 Paris deadline on October 5 remains in force, with a process hard stop
two minutes earlier. New code snapshots retain all scientific producers and
metric contracts, changing only NFS retirement and operational failure handling.
Single-rank MPI uses `UCX_TLS=tcp,self`; no MEAM validation or physics changes
were introduced. Compiler caches resolve to IDS through recorded local links.

Additional GPUs can join the same fixed queue with exclusive per-task leases:

```bash
python -m src.research.nucleus_controls.queue assist --config configs/simulation/nucleus_controls_20261005.json --frozen-code /absolute/path/to/existing/frozen/code --gpus 4 --partitions L40S,A100,A40 --exclude node52
```

This submits one GPU per array task, with unique worker/hardware receipts,
four CPUs and 24 GB per worker. Each allocation is at most two hours and must
fit the original absolute campaign deadline. Workers use the frozen plan,
seeds, scientific code and native observation protocol; they skip leased or
completed tasks. This increases completion coverage across all three cohorts,
without counting repeated futures as independent initial states.

The October 5 request for more positive and negative continuations submitted
array 1022253; its four node52 workers failed CUDA initialization before claiming
scientific tasks. Local CUDA diagnostics also failed on the assigned device,
including explicit UUID selection. No potential validation was involved.
Replacement array 1022265 excludes node52 and uses the available L40S/A100/A40
partitions; ownership, source/code identity, and unchanged budget are preserved
in `technical/helpers-20261005T140641/submission.json`. Existing node60 workers
continue concurrently. The complete target remains 12 matched blocks / 36
states / 180 continuations, with 23 independent ancestors.

At 14:09 Paris, the shared scratch `simulations` path temporarily resolved to a
root-owned 0700 directory, interrupting array 1022265 and both node60 workers.
By 14:12 the original owned tree and checkpoint files were accessible again.
No relocation or permission change was performed by this workflow. All 117
initial snapshots across the two campaigns were checked against their recorded
hashes. Failure-stage records and logs are preserved; only unfinished tasks
resume from their existing checkpoints. See
`technical/scratch-relocation-20261005/access-restored.json`.

The active replacement helper array is **1022302**, submitted at 14:13 Paris,
with receipts under `technical/helpers-20261005T141347`. The restarted node60
workers are recorded in `technical/resume-after-scratch-20261005/launch.json`.
Both use the original frozen scientific plan and the unchanged 16:59 deadline.

Scientific output is under
`${storage:training_storage}/nucleus_recovery/controls-20261005/analyses/controls-v1`.
Initial states, precise latest checkpoints and verified completed observations
archive under `${storage:archive}/simulations/nucleus-controls-20261005`.
Active MD is under `${storage:simulation_runs}/nucleus-controls-20261005`.
New-source collections are registered in `configs/datasets.json`.

These dissolved objects are sub-established embryos; the catalogue's rare
interface-associated established nucleus that later dissolved is outside this
isolated-liquid comparison. Shooting, dense descendants, million-atom cells and
relaxed data are also outside this campaign, with exclusions preserved in its plan.
