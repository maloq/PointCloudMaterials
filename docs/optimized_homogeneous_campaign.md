# Optimized homogeneous-crystallization campaigns

The optimized campaign keeps the scientific MTK-NPT protocol while removing repeated
setup and scheduling work:

- A model-specific, immutable 500 K/0 GPa liquid-only artifact is prepared once. No
  solid-liquid interface is simulated for a homogeneous campaign.
- One long-lived process owns each GPU and loads its selected MACE model once. Workers
  claim seeds transactionally, so replicas are scheduled dynamically.
- MD is committed in chunks. Each checkpoint includes positions, momenta, cell, all MTK
  thermostat/barostat coordinates and momenta, accumulated trajectory frames, and online
  event observations. Every artifact is SHA-256 verified before resume.
- PTM connected-cluster checks run online without RDF every 0.25 ps, but threshold
  persistence is evaluated only on the original 1 ps saved-frame cadence (three frames
  spanning 2 ps). Dense monitoring therefore does not redefine the event. A confirmed
  event may retain a configured growth interval and then release the GPU.
- Full PTM/cluster/RDF analysis runs in independent CPU workers concurrently with MD, or
  can be deferred. Online and offline observables must agree exactly at shared frames.

The 16,384-atom production configs are:

- `configs/simulation/atomistic/al/liquid_source_16384_mpa.yaml`
- `configs/simulation/atomistic/al/liquid_source_16384_mh1.yaml`
- `configs/simulation/atomistic/al/campaign_16384_mpa.yaml`
- `configs/simulation/atomistic/al/campaign_16384_mh1.yaml`

Both source configs use a 16 x 16 x 16 conventional FCC parent (16,384 atoms), fixed-shape
compiled inference, and a final 500 K NPT equilibration. The MPA and MH-1 outputs are
distinct and immutable.

Compiled graph rebuilding reuses the already allocated fixed-shape batch. The real
`edge_index`, periodic shifts, positions, cell, reciprocal cell, volume, and PBC tensors are
refilled in place; the fake padding partition is rewritten explicitly. A fresh
`AtomicData`/padding/`Batch` construction remains the initialization path and the reference in
tensor-for-tensor tests. On the 945,128-edge MPA liquid endpoint, the CPU portion of a rebuild
dropped from a median 0.380 s to 0.290 s (1.31x). The padding helper alone dropped from
0.0780 s to 0.0421 s (1.86x) by cloning a two-atom, zero-edge schema template rather than the
full real graph. The performance report persists the maximum real-edge count and fixed-budget
utilization.

An audited edge-envelope scan supports a future fresh-run setting of
`neighbor_skin_A: 0.30` and `pad_num_edges: 1100000` for this 0 GPa workload, provided
volume per atom remains at least 16.0 A^3. The observed maxima were 945,128 edges for every
500 K liquid-source frame, 957,219 for archived thermal FCC frames, 937,052 for interface
frames, and 1,059,791 after compressing all phase frames to the volume floor. Below about
15.82 A^3/atom, the FCC fifth-neighbor shell enters the 6.3 A cached cutoff and can raise the
edge count to 1,277,952; compressed workloads therefore need a separate approximately 1.35M
compiled bucket. Do not change the edge bucket inside an existing campaign: padding is part of
the performance evidence and checkpoint identity. The current 1.2M campaign remains unchanged
until a fresh performance/selection record is created.

The launcher requires a completed strict potential-selection report and selects the
matching campaign explicitly:

```bash
PY=/path/to/python-with-mace-0.3.16

$PY -m src.data_utils.synthetic.atomistic_potential_benchmark \
  --config configs/simulation/atomistic/al/potential_benchmark.yaml
CUDA_VISIBLE_DEVICES=0 $PY -m src.data_utils.synthetic.atomistic_homogeneous_liquid_source \
  --config configs/simulation/atomistic/al/liquid_source_16384_mpa.yaml &
mpa_source_pid=$!
CUDA_VISIBLE_DEVICES=1 $PY -m src.data_utils.synthetic.atomistic_homogeneous_liquid_source \
  --config configs/simulation/atomistic/al/liquid_source_16384_mh1.yaml &
mh1_source_pid=$!
wait "$mpa_source_pid" "$mh1_source_pid"
CUDA_VISIBLE_DEVICES=0 $PY -m src.data_utils.synthetic.atomistic_potential_performance \
  --config configs/simulation/atomistic/al/potential_performance.yaml
$PY -m src.data_utils.synthetic.atomistic_potential_selection \
  --config configs/simulation/atomistic/al/potential_selection.yaml

PYTHON=$PY DEVICES=0,1 scripts/run_optimized_al_homogeneous_campaign.sh
```

The selection report projects the exact full-duration MD workload for scheduling: all
configured equilibration and measurement steps for every replica, distributed over two
persistent GPU workers. It uses the slowest measured NPT timing block, adds each worker's
measured calculator initialization, first compiled evaluation, and warmup cost, then applies
the configured safety factor. This projection is advisory and never stops a run. The report
is bound to the exact performance report and to the
SHA-256 of each model-specific homogeneous workload config and immutable source artifact.

The launcher rejects a missing or changed selected-model source because performance evidence is
bound to its exact hashes; prepare both immutable sources before performance benchmarking and
selection. A launched campaign runs to its configured endpoint. Periodic hash-verified MTK
checkpoints remain available for recovery after an external interruption; an unexpected hard
kill can lose the current uncommitted chunk but never the latest committed checkpoint.
The launcher has no interpreter fallback: `PYTHON` must contain exactly MACE 0.3.16,
ASE 3.28.0, and the cuEquivariance 0.10.0 packages pinned in `requirements.txt`, matching
the compiled backend and exact MTK checkpoint implementation.

The performance stage evaluates every exact compiled production config against its
unpadded/uncompiled checkpoint twin before timing NPT. Energy, force, and stress parity must
pass the configured thresholds. Scientific quality is primary: MH-1 is preferred if it is a
qualified upgrade over an unqualified MPA baseline, or if it does not regress any scientific
metric relative to a qualified MPA baseline and strictly improves at least one. A scientifically
preferred MH-1 is selected even when its projected runtime is longer or its measured
throughput is lower than MPA's. Relative speed and projected total runtime are reported but
are not model-selection thresholds or execution limits.

As supplied, the scientific benchmark has neither the independent DFT set nor the replicated
melting-scan summaries needed to qualify either model, so it cannot select MH-1. When both are
unqualified, the policy asserts no scientific preference and retains MPA only as an explicit
exploratory fallback. Populate those evidence fields before treating
the selection as a physical-model validation; this workflow never interprets a foundation-model
leaderboard as validation of undercooled-Al crystallization kinetics. Performance and selection
outputs are immutable and must be archived or removed explicitly before a deliberate rerun.

There is no model fallback. A requested missing, changed, or identity-inconsistent
selection report fails before MD. The campaign also binds SHA-256 hashes for the source
manifest, metadata, atom table, trajectory, each replica's raw trajectory artifacts, and
the deferred-analysis products.

To resume after an external interruption, use the identical campaign config and output root:

```bash
PYTHON=$PY DEVICES=0,1 scripts/run_optimized_al_homogeneous_campaign.sh
```

`campaign_status.json` is `complete` only after all MD and configured analysis commits finish.
The device count is excluded from checkpoint identity, so a resumed run may use a different
number of GPUs without invalidating trajectory state.

For a campaign configured with `analysis_mode: deferred`, run CPU-only analysis later:

```bash
python -m src.data_utils.synthetic.atomistic_homogeneous_campaign analyze \
  --config configs/simulation/atomistic/al/campaign_16384_mpa.yaml \
  --workers 2
```

Replica outcomes are never collapsed into an ambiguous “finished” state:

- `event_stopped`: a persistent event was confirmed and the configured growth tail ended
  before the planned trajectory duration.
- `event_observed_full_duration`: an event occurred, but the run reached its planned end.
- `right_censored`: no event was observed through the planned end.
- `left_censored`: the event onset was already at the waiting-time origin; excluded from
  survival analysis.
- `invalid_initial_liquid`: the post-equilibration state exceeded the configured maximum
  liquid crystalline fraction; excluded from survival analysis.

These optimizations do not turn a finite campaign into a converged nucleation-rate
calculation. Stationarity, finite-size convergence, sufficient independent events, and
model-specific thermodynamic/kinetic qualification remain separate requirements.
The supplied campaign deliberately minimizes source preparation by cloning one immutable
liquid coordinate/cell configuration and assigning independent Maxwell-Boltzmann momenta,
followed by 5 ps of MTK-NPT equilibration. Its Kaplan-Meier output is therefore a descriptive
curve conditional on that shared configuration. It does not provide uncertainty over
independently sampled liquid configurations; that requires a decorrelated source bank and an
autocorrelation/effective-sample-size gate.
