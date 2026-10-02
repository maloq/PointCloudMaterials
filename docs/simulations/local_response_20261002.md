# Local80 MLIP response collection

Recipe: [local_response_20261002.json](../../configs/simulation/local_response_20261002.json).
Implementation: `src.research.local_response`; numerical/metric definitions are
in [local_response.md](../metrics/local_response.md).

This collection samples real70304-atom Al MEAM parents using the unchanged fixed
Al64 source roles and original64 center identities. It generates a new fixed-MACE
teacher experiment, not a continuation under the original MEAM potential.
One outcome-independent neighborhood per90/15/30 train/selection/test source is
used. Calibration sources are excluded. Initial source frames and identities are
hashed; atom rows link every local displacement and random stream to its parent.

Moving open environments18/24/30A are checked against larger environments through
36A on four training sources before labels or scientific fits can be collected.
The selected radius is a training-only approximation choice. It is not verified
against exact complete-cell dynamics. Failure of convergence/numerical gates
halts the dependent comparison without silently changing physics or precision.

The teacher reuses cuEquivariance float32 force/HVP kernels and conditional graph
construction from the accelerated response workflow. GPU distance search is
chunked to bound quadratic intermediate memory; a0.6A Verlet skin is filtered to
the exact potential cutoff each force call. Replica count is min(4,4096/atoms),
rounded down and floored at1, an explicit deviation for larger environments.
Original-order full-cell float64 RNG draws are gathered and cast to float32.

Commands, in conda `pointnet-torch214`:

```bash
python -m src.research.local_response.queue prepare --config configs/simulation/local_response_20261002.json
python -m src.research.local_response.queue submit --config configs/simulation/local_response_20261002.json
```

Submission freezes the source and metric contracts. One detached two-GPU Slurm
job performs gated collection, twelve scientific fits and saved-prediction
comparison/plots. Parent locks distribute work; query batches publish to STORE
after each completion. Model/optimizer states resume by epoch. Allocation expiry
permits at most two scheduler requeues; scientific errors never trigger fallback.
Gates may also run on one allocated GPU with the frozen `gate` command; a lock
prevents simultaneous gate producers. No W&B run is created for gates.

Paths resolve through machine.local.yaml:

- Output: `${storage:training_storage}/response_atlas/local80-training-20261002`
- Simulation work: `${storage:simulation_runs}/local80-response-20261002`
- Durable query archive: `${storage:archive}/simulations/local80-response-20261002`
- Training cache: `${storage:cache}/response-atlas/local80-training-20261002`

`technical/launch.json` identifies the detached job. Per-lane logs and stage
receipts show current execution. `technical/gates/complete.json` records the
selected environment and measured gate costs. `analyses/comparison-v1/plots/`
receives the final scientific PNG/PDF; all CSVs have frozen definitions/hashes.
No permanent generated encoder-feature cache is produced. Query initial states
retain native precision; no trajectory-export conversion is needed.

An existing GPU allocation can contribute while the coordinator is pending:
`python src/research/local_response_execution.py --bundle FROZEN_CODE --queued-job JOB_ID`.
This operational adapter uses the unchanged frozen collector, seeds and archive.
It holds the existing gate lock while collecting and checks scheduler state before
each parent. When the queued coordinator starts, the helper finishes and archives
its current parent, then releases the lock. The coordinator can then collect the
remaining parents and seal safely. Adapter hashes and handoff receipts live under
`technical/execution-v1/`; scientific bindings and historical source remain intact.
