# Neighborhood JEPA v2 execution and provenance

Use conda `pointnet-torch214`; dependencies are unchanged. The new module is
`src.training_methods.neighborhood_jepa.v2`. The original implementation/checkpoints
remain v1; shared StructuralMACE atom-feature refactoring preserves its state keys
and outputs. No existing data arrays, jobs, frozen source or historical metrics
are rewritten. Metric contract enforcement remains disabled as requested; new
exports still carry definitions and source hashes.

Recipe: `configs/neighborhood_jepa/v2_native_al_20260920.json`. New cache:
`${storage:cache}/neighborhood_jepa/v2-native-al-20260920-final`; registry includes parent
ancestry/potential and coordinate precision. It references checksum-verified old
graphs and adds fixed smooth moments, subset normalizers and an identity manifest.
No MD is generated. Raw quantization cannot be undone by float32 graph storage.

Launch:
```
python -m src.training_methods.neighborhood_jepa.v2.queue submit --config configs/neighborhood_jepa/v2_native_al_20260920.json
```

The launcher freezes code/config once and submits two L40S GPUs on node52 for up
to eight hours. Two workers join node59's existing allocation only after its two
legacy long fits complete. Task locks prevent duplication. Each arm has a fresh
common initialization; checkpoints resume only with identical config/source/data
identity. No short-run cosine schedule is silently extended. The queue then runs
frozen crystallization extraction/readouts, retaining source-level feature caches.
Legacy encoder extraction executes in a separate process under its own frozen code.

Inspect `output/neighborhood_jepa/v2-native-al-20260920/technical/launch.json`,
`lane-*.json`, `runs/*/status.json`, `crystallization/*/*/metrics.json` and
`CRYSTALLIZATION.md`. Per-fit manifests include dependencies, loss/layout contract,
source identity, statistical sampling and draw budget. A deadline checkpoints
training/probes; completed source extractions are reused after interruption.

Checks and limitations are recorded under `technical/checks/`. Correctness tests
cover CPU contracts and real CUDA MACE, including BF16. A 32-update small-batch
compiled smoke is a plumbing check, not a benchmark or scientific winner.
The five production fits require approximately 2–5 GPU-hours in aggregate; frozen
feature extraction/readouts are expected to dominate the remaining budget and
are resumable. This is an estimate, not a measured completion guarantee.

Deferred plan extensions: multi-seed confirmation, history/shuffled-history arms,
tensor-product/cross-degree architecture expansion, comprehensive retrieval and
coordinate-noise/TDA continuity assays. They are not prerequisites for the minimal
causal A–E comparison and are not represented as completed evidence.

The user subsequently added node53 allocation **1000616**. Its idle H100 joins
immediately as lane 5; lane 4 waits for the existing `spacetime-pred003-long` fit.
The operational launcher is `src/training_methods/neighborhood_jepa/allocated_join.py`,
captured separately as `technical/node53-join.py`; training source/config are unchanged.
`technical/launch-node53.json` records the command and hash. All six lanes share
the same task locks and scientific comparison.

## Paired-GPU large runs

`configs/neighborhood_jepa/large_20260920/campaign.json` targets the existing
node53 and node59 allocations, two GPUs per run. The coordinator waits for the
previous queue's local workers to finish, then performs a separate timing
preflight, freezes the update budget, trains and runs frozen crystallization
probes. It never cancels the preceding comparison. Logs and launcher status live
in each run's `technical/` directory; all phases use an immutable source snapshot.

Launch once from `pointnet-torch214`:

```bash
python -m src.training_methods.neighborhood_jepa.v2.large submit --config configs/neighborhood_jepa/large_20260920/campaign.json
```

Preparation uses `v2.expand` with the adjacent `data.json`. It selects existing
native Al anchors and creates graph and corrected-moment caches under IDS. The
registry records their potential and parent collection. No simulation is run.

The trainer uses two-device gradient replay with a single complete-batch loss.
A smoke test covers training, validation, checkpoints and metric export;
`tests/test_neighborhood_jepa_parallel.py` checks compiled/eager gradient parity
against one-device training and the wider encoder's rotational behavior.

Large-batch loading requires a 65,536 file-descriptor soft limit, set before spawning workers. The 2,048-anchor end-to-end smoke test exercises this transport limit.
