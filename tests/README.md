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

Add a test when it catches a meaningful failure. Prefer observable results over
private call sequences or Python source-text matching. Experiment seeds, widths,
paths, temperature grids, and rendering presets belong in their configuration
and run record; avoid repeating them in assertions solely to freeze a run.
Scientific protocol constraints, independence, and destructive-operation safety
still warrant regression coverage.

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
