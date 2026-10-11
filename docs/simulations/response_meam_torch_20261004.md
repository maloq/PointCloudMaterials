# Differentiable Lee2003 Al response benchmark

The [PyTorch implementation](../../src/research/response_meam/potential.py)
specializes the exact pinned Lee2003 Al potential. It includes 2NN corrections,
the original pair-table cubic interpolation, angular densities and three-body
screening. Forces and Hessian-vector products use AD; no energy/force fit or
potential distillation is performed. See the [upstream attribution and scope](../../src/research/response_meam/NOTICE.md).

This is a numerical and hardware experiment on the existing local80 response
task, not a new encoder-training campaign. The original MACE collections and
fits remain frozen. Use conda `pointnet-torch214`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m src.research.response_meam.benchmark all \
  --config configs/simulation/response_meam_torch_20261004.json
python -m src.research.response_meam.report \
  --config configs/simulation/response_meam_torch_20261004.json
```

`validate` and `measure` may also be run separately; measurement requires the
completed numerical gate. Source/config hashes bind a run before execution;
changes require a new output revision. `technical/code/` retains the producer.

The [recipe](../../configs/simulation/response_meam_torch_20261004.json) reuses
the original four training-source neighborhoods and two noise streams from the
completed force-engine benchmark, with unchanged atom IDs/displacement bases.
It also reuses the frozen response-geometry training-only observable normalizer.
No dataset resplit, new quench, crystal-label selection or held-out tuning occurs.
The richer768 targets are the current campaign's three alternative256-feature
banks evaluated along the same20/100fs trajectories. The student would still
receive only initial unrelaxed local80 geometry; no student is fitted here.

Validation compares energies, forces and force derivatives with LAMMPS, then
full768-coordinate trajectories and responses with CPU float64 paired finite
differences. Float32 PyTorch must also match float64 under declared tolerances.
Complete-path validation uses one seed per ancestor; timing uses both original
seeds. All validation sources are train ancestors from the fixed Al64 release.

Timing uses one GPU sequentially for MACE float32 cuEquivariance, PyTorch MEAM
float64 and PyTorch MEAM float32. Each shape gets one full-query warmup. Three
timing repeats rotate engine order and synchronize the GPU. Batch1 corresponds
to the existing24A local collection's4096-atom budget; batch2 is explicitly a
separate throughput measurement. Setup/IO is excluded; online full-cell noise,
neighbor construction/updates, integration, forces, directional derivatives and
all observable derivatives are included. Timing repetitions are not additional
independent trajectories. W&B remains unused for this hardware benchmark.

Simulation feature/response tensors are written on SCRATCH and hash-verified
into STORE. Positions are not exported. Input timelines and integration precision
remain unchanged. The registry ID is `response-meam-torch-20261004`.
Results: [benchmark report](../../output/response_meam/torch-port-20261004/README.md).
Definitions: [metric contract](../metrics/response_meam_torch.md).

The port currently supports the exact pure-Al parameter files and open
environments, not arbitrary MEAM parameterizations, alloys or periodic cells.
Piecewise screening and spline derivatives follow the original local branch.
The speed comparison does not imply matching MACE physics or demonstrate better
crystal clustering or nuclei prediction.
