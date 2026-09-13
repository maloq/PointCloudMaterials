# Portable project and storage placement — 2026-09-13

Question: can the project launch outside this cluster without embedding mount
points in scientific settings, while using IDS for cache and SCRATCH for new MD?

Implementation and reproduction commands: [portable setup](../../docs/portability.md).
Configuration: `configs/machines/{local,cluster.example}.yaml`, ignored local machine
profile, `configs/datasets.json`, and the three `configs/simulation/*_crystallization.json`
recipes. Existing frozen manifests and exact-resume configurations remain unchanged.

Research records live here; operational receipts, logs and disposable fixtures are in
`output/portability/storage-20260913/technical/`. New maintained code is
`src/project_runtime/` and `scripts/project.py`; maintained environment declarations
are under `environments/`, with tests in `tests/test_project_portability.py`.

## Completed storage changes

- Repository `datasets/` copied with SHA256 source/destination/source verification to
  `/work/PERSO/vmorozov/datasets`; the checkout now contains a compatibility symlink.
- Training cache copied back from WORK to `/home/ids/vmorozov/training-cache`, verified
  before deleting the old copy. The former WORK path forwards to IDS.
- The repository dataset cache now lives at
  `/home/ids/vmorozov/training-cache/repository-datasets`, verified through the existing
  `experiment_registry.py relocate-caches` workflow. `datasets/cache` forwards there.
- Existing simulation input trees remain on WORK. Fresh elemental source/branch runs
  stage on SCRATCH and publish checksum-verified completed runs to STORE. Historical
  controllers and exact-resume state were not moved or relaunched.

Receipts: `datasets-verification.json`, `cache-verification.json`, and
`dataset-cache-relocation.json` in the operational technical directory.

## Validation

- 57 targeted tests passed: portability, elemental conversion, spatiotemporal TDA and
  embedding forecast. A cache was physically moved between training chunks; resumed
  weights and sampling state matched uninterrupted training exactly.
- A genuinely separate Python 3.12.13 venv with no system-site-packages installed the
  pinned CPU requirements. `pip check` reported no conflicts; its PyTorch is
  `2.11.0+cpu`. The CPU transitive lock is included as a maintained environment file.
- `technical/bundle-plan.json` selects a tiny synthetic forecast fixture and the Ti
  potential. Build with `project.py bundle --plan ... --destination ... --apply`, move
  the exported directory, verify its inventory, then run the existing forecast module
  using `technical/cpu-smoke.json` in that bundle. The relocated export completed both
  CPU epochs and held-out evaluation using its own data paths, without cluster mounts.
- CPU Hydra composition, shooting YAML resolution and all three elemental potential
  paths were checked. The LAMMPS doctor checks MEAM and EAM/alloy availability and reports
  missing shared libraries with an actionable message.
- A 128-atom pure-Ti integration smoke (20 steps, 0.02 ps, five frames) used the same
  Kavousi potential and exercised SCRATCH integration, verified float16 conversion,
  restart retention and STORE publication. It is an infrastructure test, not a
  crystallization dataset. Maximum coordinate quantization error was 0.00390244 Å and
  RMS 0.00156207 Å; full checksums are in its binary conversion receipt. It is archived
  under `/store/PERSO/vmorozov/simulations/portability-ti-smoke-20260913` and identified
  as a test fixture in the catalog.

No production simulation was launched. Linux CPU export was exercised on this host;
a second physical machine and a fresh Slurm production allocation were not tested.
Historical cluster-specific launchers retain their original constraints. A full
checkout snapshot includes dirty and untracked work, but preserves external dataset
links rather than duplicating every external dataset. Its sibling audit on STORE
identifies the exact capture and external dependencies.

Additional verification: 13 shooting/registry/workflow regression tests passed; the
final elemental/portability suite passed 21 tests. Metric documentation contracts
and the experiment registry build completed successfully.
