# Backbone v2 execution

Use conda `pointnet`. The [scientific protocol](../experiments/local_predictability_20260917/BACKBONE_V2.md)
and [configuration](../configs/local_predictability/backbone_v2/rtx6000_screen.json)
define the matched screen. Existing v1 modules and receipt checks are unchanged.

Install the extra pinned dependencies into the existing torch 2.11.0+cu128,
cuEquivariance 0.10.0 environment:

```bash
python -m pip install --no-deps -r environments/requirements-backbone-v2.txt
python -m pytest -q tests/test_axial_gatr.py tests/test_backbone_v2.py
```

GATr's upstream package metadata still pins historical NumPy/opt_einsum versions.
The adapter is tested with the existing NumPy 1.26.4 and opt_einsum 3.4.0; install
the listed extras without replacing the running environment's dependencies.
GATr imports xformers, but tensor attention masks select its unmodified PyTorch
SDPA path. The recorded CUDA trace determines which attention kernel ran.
No third-party algebra, attention or normalization kernels are vendored or edited.

```bash
python -m src.research.local_predictability.backbone_v2 \
  --config configs/local_predictability/backbone_v2/rtx6000_screen.json --stage screen
```

This explicitly includes separate fitting gates and real-workload profiles before
the two snapshot fits. `--stage fit` never runs hardware benchmarks. The additional
stages are `gate`, `profile`, `fit`, and `export`; these require `--encoder mace` or
`--encoder axial_gatr`. Fit/export also accept `--objective physical_means|onset`
and `--variant snapshot|history12|repeat12`. A continuation can pass `--parent`
pointing to its own v2 snapshot checkpoint. Architecture/objective/data identities
must match. A v1 checkpoint or receipt cannot authorize a v2 fit.

Use `--resume` only for an existing run. Checkpoints preserve model, optimizer,
sampler and all RNG states, including an unconsumed prefetched batch. The training
deadline can be extended for a new allocation; other configuration/code changes
require fresh runs and gates. The run has an exclusive worker lock. Finished
split exports are checksummed and reused only with the same best checkpoint.

The RTX6000 screen is intended to follow the existing native-readout tracked
execution, including its history extraction and both final readouts. The launch
specification and Slurm job details belong in its `technical/` folder. Its training
cutoff is 18:02:01 UTC on September 17, leaving an hour before allocation expiry.
The deadline saves an exact-resume checkpoint; a timed-out screen is incomplete.
Scientific interpretation remains deferred until requested.

Launched detached at **2026-09-17 15:50:08 UTC** on node58, allocation 996727,
after all six prior native readouts completed. The
[tracked execution](../output/local_predictability/rtx-backbone-v2-20260917/technical/execution-rtx6000/execution/run_record.json)
contains the immutable source/config snapshots; the
[launch specification](../output/local_predictability/rtx-backbone-v2-20260917/technical/execution-spec.json)
retains its explicit dependency. Validation before launch: 20 focused tests passed
on the RTX6000 (including actual CUDA attention profiling, both task objectives,
checkpoint resume and export), and real Al snapshot/history forward-backward
smokes passed for both architectures on the idle H100. The broader native
regression suite also passed before the final onset-export checks.

Artifacts are under `output/local_predictability/rtx-backbone-v2-20260917/technical/`:
`mace/` and `axial_gatr/` each contain `gate/`, `profile/`, and
`physical_means/snapshot/`. Validation tables include frozen metric definitions
and implementation hashes. Per-update records contain sampled row indices,
examples seen and accumulated elapsed time for matched-example/time curves.

## H100 comparison and onset repeats

The H100 NVL on node53 (allocation 995957) was idle after its previous native
history run. A detached screen started at **2026-09-17 15:57:35 UTC**, using
`configs/local_predictability/backbone_v2/h100_screen.json`. It compares fresh
cuEquivariance MACE and GATr on the H100 with effective/microbatch eight, a 24 GiB
device input cache, FP32 and one seed. Its explicit profiles use 20 timing steps.

The follow-up configuration is
`configs/local_predictability/backbone_v2/h100_repeats.json`. It waits for the
tracked H100 screen, exports the matched speed and physical errors, and runs a
GATr onset parent followed by snapshot, history12 and repeat12 continuations.
Parent and child configurations differ only in output location. The identical
passing gate receipt and its checksummed checkpoint are copied without rewriting
their contents; the trainer verifies the same complete scientific identity.
All source/condition/label/selection identities and the final sampler state are
checked against the completed MACE reference before fitting the new parent.

The screen, onset parent, onset controls and comparison tables use separate runs:

- `output/local_predictability/h100-backbone-v2-20260917/`
- `output/local_predictability/h100-gatr-onset-parent-20260917/`
- `output/local_predictability/h100-gatr-onset-controls-20260917/`
- `output/local_predictability/h100-backbone-comparison-20260917/`

Training stops and checkpoints before **2026-09-18 02:59:30 UTC**, leaving an hour
before allocation expiry. Use the driver's `--resume` for an interrupted repeat;
configuration and implementation must match its recorded identity. Existing
RTX6000 and v1 training modules are untouched. This extension reuses available
data and does not generate simulations or add training seeds.
