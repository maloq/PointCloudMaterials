# Tests

Run the suite from the repository root in `pointnet`:

```bash
conda activate pointnet
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
```

This runs on CPU and skips the fused CUDA backend test. Run that test separately
inside an allocated GPU session:

```bash
python -m pytest tests/test_liquid_sro_benchmark.py -k fused_mace_gpu -q
```

For a local change, start with the corresponding test file. The retained suite
covers numerical invariance and gradients, temporal identity and leakage,
trajectory conversion and corruption, checkpoint recovery, simulation protocol
generation, and workflow regressions. Small EMT simulations and temporary
artifacts exercise the simulation machinery without submitting cluster jobs.
Some atomistic fixtures load repository potential configurations; their configured
model files must be available even when the test uses an injected EMT calculator.

`test_embedding_forecast.py` checks causal time windows and disjoint mean bins,
matched anchors across history lengths, batched mmap loading with spawned workers,
training-only scaling, whole-lineage separation, gradients from every historical
frame, joint Gaussian covariance/NLL, and learning/checkpoint/evaluation round trips
for both mean-bin and full-path forecasts. Autoregressive tests also verify
prediction feedback, gradients through earlier rollout steps, correctly shifted
teacher-forcing targets, and the prohibition on teacher forcing during evaluation.
Scale-up coverage checks causal history augmentation, bit-exact augmented training
across an explicit checkpoint continuation, streaming metric agreement, and frozen
Slurm specifications with independent training chains and a shared final join.
It also checks one-job fits that depend on an existing shared preparation job.

Add a test when it catches a meaningful failure. Prefer observable results over
private call sequences or Python source-text matching. Experiment seeds, widths,
paths, temperature grids, and rendering presets belong in their configuration
and run record; avoid repeating them in assertions solely to freeze a run.
Scientific protocol constraints, independence, and destructive-operation safety
still warrant regression coverage.

`test_mace_denoising.py` checks zero-initialized residual fusion, gradients from
every observation, shared atom-identity permutation symmetry, the matched
anchor-only control, equal homology-block weighting, per-history timing in
mixed-cadence batches, and exact paired source-level potential statistics. `test_mace_temporal.py`
also verifies that exposing atom features preserves the existing MACE pooling.

`test_experiment_registry.py` covers cleanup preflight and symlink boundaries,
retained prerequisites, verified recoverable log archives, execution failure logs,
and immutable provenance across attempts. See [the registry guide](../docs/output_registry.md).

`test_elemental_conversion.py` covers Ti/Ta position-producer conversion,
periodic coordinate semantics, and rejection of corrupt IDs, cadence, and frame
counts. The real LAMMPS source/branch smoke and full-size preflights are documented
in [the Ti/Ta experiment](../experiments/ti_ta_crystallization_20260907/README.md).

## September 6, 2026 cleanup

Removed 19 test functions (24 collected cases, including parameterization):

- YAML snapshots for historical generator, homogeneous, transition, jumpy-FFS,
  nested-pilot, potential-selection, and runtime-sweep configurations.
- Retired GeoFrame config snapshots, including a hard-coded parameter count,
  and mirror-probability assertions repeated across five configs.
- A redundant encoder registry check and a source-text search purporting to test
  memmap deletion safety.

Kept numerical and integration tests in those same modules. Corrected two
checkpoint migration tests to use their certified historical destination digest
instead of requiring the current checkout to retain that digest. The certificate
test also rejects uncertified source and destination digests. Production migration
certificates and compatibility rules were not changed.

The checkpoint plotting test now imports its maintained implementation directly.
No research implementations or uncommitted test files were deleted.

Validation in `pointnet`: baseline 323 passed, 6 failed, 1 skipped; after cleanup
305 passed, 0 failed, 1 skipped (the CUDA-only test). Disposable inventories and
test logs are under `output/test_cleanup_20260906/`.

Float16 trajectory storage and periodic-boundary decoding are covered in
`test_elemental_conversion.py`, including verified original retirement and legacy-path resolution.

`test_independent_meam_source_campaign.py` also checks recovery from expired Slurm
receipts without disabling duplicate-submission detection.

`test_mace_temporal.py` exercises the temporal transformer with a small real MACE:
history/time dependence, gradients through every frame, rigid-motion and atom
permutation invariance, frame chunking, and registry/checkpoint construction.
See [the encoder input contract](../docs/mace_temporal_encoder.md).

`test_mace_history.py` checks the relaxed target's fixed atom membership through
periodic crossings and exact anchor reconstruction. The temporal encoder tests
also check full-batch covariance gradients under history microbatch replay.

`test_topology_analysis.py` checks the standard pipeline's real-history inference,
anchor visualization and atom identity, including temporal interventions. It
changes only held-out targets and verifies that training-only ridge predictions
remain identical. `test_vicreg_relaxed_histories.py` also checks source pairing
and seed averaging for both trained-head and ridge comparisons.
Its collector test covers legacy and flat report directories and repeats
aggregation to ensure the generated comparison is never read as a model report.

`test_analysis_storage.py` checks portable flat reports, checkpoint identity
protection, verified cache relocation with preserved loader paths, and targeted
inference-cache removal. It also verifies that post-training analysis failures
retain the recovery checkpoint while successful analysis may remove it.

`test_elemental_conversion.py` also verifies the explicit Al/FCC and Ti/BCC
source lattice contracts and atom counts.

## Research layout and metric documentation

`test_research_layout.py` checks inference-cache deletion boundaries, retained
provenance, legacy analysis locations and metric-definition/source hashes.
`test_analysis_storage.py` covers portable readable galleries; the forecast tests
check exact optimizer/sampler resume in both recorded output layouts.

```bash
conda run -n pointnet python -m pytest -q tests/test_research_layout.py tests/test_analysis_storage.py tests/test_embedding_forecast.py tests/test_experiment_registry.py
```
