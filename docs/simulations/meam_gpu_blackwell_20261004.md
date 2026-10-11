# Differentiable CUDA MEAM, October4

The user requires fast response derivatives as well as crystallization MD.
The implementation is `src/research/meam_gpu/`; the recipe is
`configs/simulation/meam_gpu_blackwell_20261004.json`. This is a pure-Al Lee2003
specialization, not a general multicomponent MEAM replacement.

## Literature and implementation

[Official MEAM documentation](https://docs.lammps.org/latest/pair_meam.html)
now lists `meam/kk`. The installed July2025 LAMMPS has no Kokkos package, so a
separate native Blackwell120 executable was built from upstream commit
`8de817dd79bfe4525d5d39246a212d833e6dee07` (September30,2026).
The [Kokkos MEAM source](https://github.com/lammps/lammps/blob/stable/src/KOKKOS/pair_meam_kokkos.cpp)
credits NVIDIA and Sandia contributors. That engine provides a useful native
performance reference, but does not supply our differentiable response path.

[Johansson et al., LAMMPS-KOKKOS](https://arxiv.org/abs/2508.13523) motivates
keeping particle operations on the accelerator, arranging parallel work around
memory access, and limiting synchronization. Its reported application examples
are not a benchmark of this Al implementation. The
[official precision guidance](https://doc.lammps.org/Build_extras.html) also
motivates explicit conservation checks before accepting mixed precision.

The new differentiable path uses CUDA spatial bins, a reused Verlet candidate
list and one warp per candidate pair for screening. Completely screened pairs
exit early. Only compact surviving pairs and partial screeners reach PyTorch;
there is no dense pair-by-neighbor intermediate. Overflow is an explicit error.
Continuous screening, angular density and embedding derivatives remain in AD.
Compiled energy gradients give forces; a second reverse pass gives Hessian-vector
products. A development forward-AD compiler failure was resolved by exploiting
the symmetric energy Hessian; no finite-difference approximation was substituted.

The cutoff is evaluated as `[x*(4-6*x+4*x²-x³)]²`, algebraically the original
`[1-(1-x)^4]²`, with less cancellation. Conservative topology guards and explicit
zero screening avoid logarithms of zero near floating-point boundaries.
Force arithmetic has float64 and float32 modes; integration and scalar energy
reduction remain float64. The latter is not Kokkos's mixed-precision scheme.

## Protocol and execution

1. Check periodic forces, energies, virial precision and a matched1ps NVE path.
2. Check/time full20/100fs responses for four original train parents, both noise
   seeds,768 features, two directions and batch sizes1/2, against the validated
   original MEAM. No new encoder training or W&B runs.
3. Continue the shared16,384-atom liquid for200ps at450K with the original
   BAOAB/PCG64 protocol. Compare with the already-running CPU and float64 GPU
   descendants, preserving their frozen code and historical output.
4. Prepare a separate crystal/liquid slab (20ps at2500K with one half frozen;
   45.68% PTM crystal before the instantaneous450K quench) and release every atom for
   matched50ps mixed/float64 runs. This checks seeded growth, not nucleation.
5. Publish speed, energy conservation, response errors, structural evolution and
   phase-restricted descriptive thermodynamics. NPT/EOS and melting-point
   equivalence are not established by these fixed-volume assays.

The native Kokkos executable and license are archived under the configured
software archive. An unapplied native-source optimization idea is retained only
in the run's technical development record; it is not a live production variant.
The initial response-v1 development measurements are also excluded from the
final all-parent/two-seed response-v2 comparison.

Run/resume the frozen worker recorded in `technical/launch.json`; do not launch
another copy into the same output. Maintained commands:

```bash
python -m src.research.meam_gpu.build upstream-double --config configs/simulation/meam_gpu_blackwell_20261004.json
python -m src.research.meam_gpu.workflow launch --config configs/simulation/meam_gpu_blackwell_20261004.json --allocation ALLOCATION
python -m src.research.meam_gpu.report --config configs/simulation/meam_gpu_blackwell_20261004.json
```

Outputs: `output/meam_gpu/blackwell-20261004/`, with grouped analyses, plots,
tables, execution receipts, frozen source and failures. New observations go to
SCRATCH then verified STORE archival; native integration states are retained.
See [metric definitions](../metrics/meam_gpu.md). This is one explicit MEAM
engineering study within the shared Al pool, not completion of all Al research.

The verified interface is checked before its450K production quench: mobile
MSD101.75A² and0% mobile PTM crystal, against acceptance requirements ofMSD>4A²
and mobile crystal<10%. The reported descendants are
`verified-interface-{mixed,f64}-50ps`. No force-field parameters were fitted to
crystallization outcomes.

A1325K pilot was rejected after a diffusion audit: mobile MSD was only0.219A²,
so its rapid return to99.8% PTM order was cooling of a thermally distorted
crystal. Those original paths remain preserved with `phase-audit.json`
corrections, including the earlier `interface-preparation-failed-v1` attempt.
They are excluded from the verified growth comparison.

The supplementary same-GPU benchmark compares the new mixed engine against
the prior float32 MEAM and float32 MACE, preserving their older precision:
`python -m src.research.meam_gpu.response_speed --config configs/simulation/meam_gpu_blackwell_20261004.json`.
Native-coordinate divergence and RDF:
`python -m src.research.meam_gpu.diagnostics --config configs/simulation/meam_gpu_blackwell_20261004.json`.
The original CPU200ps reference may finish after this GPU campaign and retains
its own completion record.

## Response API

```python
potential = FastAl(files, atoms, dtype=torch.float32)  # open response environment
force, force_jacobian_times_directions = potential.force_tangent(q64, directions64)
# q64: (batch, 3*atoms); directions64: (batch, 3*atoms, nonzero_direction_count)
```

Use `FastOracle` for complete response targets: it propagates coordinates and
displacement tangents through the original BAOAB/noise/observable protocol.
Use `FastAl(..., box=box).force(q64)` for periodic MD. Topology buffers are
reused, so finish the force/derivative evaluation before another support call;
retaining several raw `forward()` energy graphs across MD steps is outside this
interface. The maintained oracle completes and detaches each force/HVP before
advancing. This provides analytic trajectory responses without retaining a
full reverse-mode tape across the MD trajectory. Both public production paths
were numerically checked; arbitrary higher-order derivatives were not audited.

Validated toolchain: PyTorch2.14.0+cu130, CUDA compiler13.4, sm120,
RTX PRO6000 Blackwell. CUDA source, compiled extension, build flags and hashes
are archived beside the native Kokkos executable. Compilation is separate from
steady-state timings. New training campaigns can use the API; already-running
campaigns keep their recorded engines and precision.

`python -m src.research.meam_gpu.publish --config configs/simulation/meam_gpu_blackwell_20261004.json --wait`
refreshes figures after both GPU workers complete. The phase-specific solid
check uses `python -m src.research.meam_gpu.phase_stats --config configs/simulation/meam_gpu_blackwell_20261004.json --wait`.
