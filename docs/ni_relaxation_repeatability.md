# Ni quench repeatability investigation

The October9 response campaign found substantial differences between independent
FIRE quenches of identical full-cell inputs. This diagnostic reuses its five
audited pre-quantization endpoints and atom identities. It is separate from the
running training recipe and does not replace any response targets or minima.

The production bin builder uses atomic insertion; bin order controls the order
of float32 summation in the directed-neighbor EAM kernels. The private
`src.research.ni_relaxation_repeatability` backend canonically sorts neighbors by
atom ID after each rebuild, then uses the original density and force kernels.
Float64 forces are a small explicit ablation, not the default proposed fix.

Recipe: `configs/ni_response/relaxation_repeatability_20261009.json`.
Frozen code, process ownership and diagnostic progress are in
`output/ni_relaxation/repeatability-20261009-v1/technical/`.
Full cells and failures are published under the recipe's simulation archive.
Each repeated execution deliberately bypasses the relaxation index; already
completed diagnostic receipts are reused. No automated test suite or online
training run is created by this numerical assay.

Stages:

1. Eight force evaluations with independent neighbor rebuilds for each endpoint,
   raw32/canonical32/canonical64; atom permutation and native CPU comparisons.
2. Two independent quenches for raw32, canonical32 and tighter canonical32 on
   all five endpoints; canonical64 on the first endpoint only.
3. Canonical quenches after local atom-vector RMS1e-7/1e-4-angstrom input changes.
4. Re-quench the same signed MD endpoint bank to measure how response targets
   change when the minimizer's arithmetic order is repeatable.

The two existing training workers continue. Diagnostic memory use is small and
may share their GPUs; timings therefore record actual elapsed time, not isolated
device throughput. Scientific definitions and implementation hashes accompany
the metric tables: [metric definitions](metrics/ni_quench_repeatability.md).

Completed October9:89 quenches and all force audits, with frozen producer
identity`af9130939503204117fb3f6fab70abbf328529f5d0fb69c54ad403a666704400`.
Sorted float32 forces produced exact native32000-atom repeat coordinates on all5
inputs; numerical input sensitivity remains even in the single float64 reference.
[Scientific findings and figure](../experiments/ni_quench_repeatability_20261009/README.md).
