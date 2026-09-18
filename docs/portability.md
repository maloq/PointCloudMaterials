# Portable machines, datasets and simulations

The checkout contains code and scientific recipes. One machine file selects where
its data lives. Copying a self-contained bundle to another directory or Linux
machine does not require editing scientific configs or data manifests.

## Setup

Use Python 3.12. For new GPU work on this cluster, activate
`conda activate pointnet-torch214`. Existing runs and exact resumes retain their
original `pointnet` environment; see [the GPU upgrade](pytorch214_upgrade.md).
For a fresh Linux CPU installation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r environments/requirements-cpu.lock.txt
python scripts/project.py doctor
```

`requirements-cpu.txt` records the direct requirements; the CPU lock also pins
transitive dependencies. `requirements-gpu.txt` selects the cluster's PyTorch
2.14.0 / CUDA 13.0 implementation and optional equivariance kernels. Install it
instead for a compatible NVIDIA machine. A matching GPU driver and a LAMMPS build
with the requested potentials/MPI are separate system dependencies.

Configuration precedence is `PCM_MACHINE_CONFIG`, then ignored
`machine.local.yaml`, then portable defaults. Explicitly naming a missing machine
file is an error. Copy `configs/machines/local.yaml` or
`configs/machines/cluster.example.yaml` to `machine.local.yaml` and adjust it.
Relative roots are anchored to the checkout. Do not commit your local profile.
`python scripts/project.py paths` shows the effective settings.

| Storage role | This cluster | Portable default |
| --- | --- | --- |
| datasets | `/work/PERSO/vmorozov/datasets` | `datasets/` |
| simulations (existing inputs) | `/work/PERSO/vmorozov/simulations` | `data/simulations/` |
| cache | `/home/ids/vmorozov/training-cache` | `data/cache/` |
| simulation_runs (new results) | `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations` | `tmp/simulations/` |
| archive | `/store/PERSO/vmorozov` | `data/archive/` |
| output | checkout `output/` | `output/` |
| analysis | `/work/PERSO/vmorozov/analysis` | `output/analysis/` |

The repository `datasets` link points to WORK. Its `cache` child points to
`/home/ids/vmorozov/training-cache/repository-datasets`. Old WORK cache paths also
forward to IDS. Compatibility links retain existing research references.

## Recipes and identity

Use `${storage:cache}/NAME` for an output location and `${dataset:ID}` for a
registered input. `configs/datasets.json` records IDs, root roles, relative paths,
dependencies and aliases. `project.py datasets` lists resolved locations.
Hydra/OmegaConf configs and supported JSON commands use the same resolver.
New plain JSON entry points should call `src.project_runtime.paths.load_json`;
file loaders should call `resolve_path` where they consume recorded paths.

Frozen configs, exported metric definitions and dataset manifests remain byte
unchanged. Legacy paths are translated when read through the resolver. Forecast
resume compares a canonical path spelling while preserving all scientific settings,
then verifies the original cache manifest and implementation hashes. Moving data
alone does not authorize changing a model, seed, horizon, augmentation or split.
Implementation changes still require frozen code or an explicit reviewed transition.
The `storage-relocation` transition additionally names the exact cache manifest hash;
it is not created automatically for existing checkpoints.

For forecast training use the existing module, with `--device cpu` for a CPU run.
For MACE VICReg use `--config-name vicreg_mace_full_cpu`: this keeps the scientific
settings and selects the CPU-compatible full precision encoder path. CUDA and CPU
arithmetic are not promised to be bitwise identical. Historical Slurm shell
launchers retain their recorded cluster assumptions; the portable elemental and
forecast entry points are the supported starting points on another machine.

## New simulation runs

```bash
python scripts/project.py doctor --lammps
python scripts/run_lammps_campaign.py elemental run \
  --config configs/simulation/ti_crystallization.json --run-name ti-new --ranks 8
```

The Al, Ti and Ta templates preserve their recorded physical protocols and potential
checksums. Fresh elemental runs require a unique name, stage resolved inputs and
machine settings under SCRATCH, and use the current machine's LAMMPS/MPI launcher.
Use `al_crystallization.json` or `ta_crystallization.json` for the other materials.
Ti and Al use 100,000 atoms; the archived Ta protocol retains its original atom count.
The maximum duration and free-space requirement remain explicit in each recipe.
`--ranks` must fit the allocated CPUs. A serial machine profile defaults to one rank;
an MPI profile defaults to the available CPU allocation. Run the same command inside
an existing Slurm allocation, or place it in an ordinary batch script with explicit
resources. The machine profile does not silently submit a Slurm job.

`elemental branch --config CONFIG --parents PARENTS --index INDEX --run-name NAME`
also stages new branch output on SCRATCH; the parents file retains the existing
`{"branches": [...]}` format. `elemental sequence --ta-config CONFIG --ti-config
CONFIG --run-name NAME` preserves Ta-before-Ti execution. Recovery of an existing
failed Ta campaign uses its original config and `--resume-ta`; it retains its location.
No submitted launcher path is renamed by this change.

Completed elemental runs convert position exports with the existing verified
float16 converter. Box bounds stay float32 and identity/timeline arrays remain
exact integers; integration and restart files retain their original precision.
After all writers close, the complete run is copied to STORE, checked against
SHA256 inventories of both destination and source, registered by ID, and the SCRATCH
run becomes a compatibility link. A failed copy retains the originals and raises.
Publication audits sit beside the archived run. `--keep-on-scratch` explicitly skips
publication; later use `project.py publish-simulation SOURCE --id ID --move`.

SCRATCH is purged after 30 days of inactivity and is not backed up. Interrupted or
failed runs stay there for diagnosis. After stopping all writers, preserve failure
and restart evidence with `project.py archive-failed-simulation SOURCE --id ID
--inactive`; this makes a verified STORE copy and retains the working directory.
It does not label failed dynamics complete. Historical shooting families with explicit
`--campaign-root` must be given a path under the configured simulation_runs root;
maintained shooting configuration defaults now use that root. Their original resume
and finalization semantics are preserved.

## Self-contained export

Write a JSON selection, for example:

```json
{"datasets": ["potential-ti-kavousi2019"], "files": []}
```

```bash
python scripts/project.py bundle --plan selection.json --destination /path/to/export
python scripts/project.py bundle --plan selection.json --destination /path/to/export --apply
python /path/to/export/scripts/project.py verify-bundle /path/to/export
```

Preview reports selected dataset bytes; application copies maintained source,
configs, environment files and metric documentation plus selected data/artifacts.
`files` lists checkout-relative checkpoint/config paths or directories. Dataset
dependencies are included recursively. External symlinks require their target dataset
to be selected; missing dependencies fail explicitly. Internal aliases become relative.
Data manifests retain their original bytes, with recorded paths resolved through the
bundle's own catalog and local machine file. Copy the whole export to the destination
machine, install the environment and run there. Verify the bundle before running:
new output files intentionally change its full inventory.

## Full checkout snapshot

`python scripts/project.py snapshot /store/PERSO/vmorozov/projects/UNIQUE-NAME`
copies the entire current checkout, including `.git`, dirty/untracked source and local
outputs. It preserves external symlinks instead of duplicating all datasets. The
sibling `UNIQUE-NAME.snapshot.json` contains file hashes, external targets and storage
settings. Keep the checkout idle during verification; source changes abort the copy.
This is a faithful cluster snapshot. Use a selected bundle for a self-contained
export; external datasets need their own durable copies for disaster recovery.
