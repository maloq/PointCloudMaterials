# Can identical Ni inputs be relaxed reproducibly?

The response-training audit found that independently quenching identical
32000-atom MD endpoints produced different local structures and embeddings.
This experiment isolates neighbor summation order from minimizer tolerance and
input sensitivity, using the same five audit sources without outcome selection.

The candidate sorts directed neighbor lists by original atom ID after each
rebuild, then calls unchanged GPU EAM kernels. Native FIRE uses float64 states.
Raw and sorted float32-force recipes share the0.01-eV/angstrom tolerance; a
sorted0.001 recipe checks convergence sensitivity. Sorted float64-force quenches
are limited to the first endpoint. Two repeats per recipe and source are followed
by tiny local coordinate perturbations and matched signed-response endpoints.

Recipe: `configs/ni_response/relaxation_repeatability_20261009.json`.
Definitions: [Ni quench metrics](../../docs/metrics/ni_quench_repeatability.md).
Results: `output/ni_relaxation/repeatability-20261009-v1/`.

Reproduce the recorded v1 with its frozen source and config from
`technical/code`, using conda `pointnet-torch214` and an allocated GPU:

```bash
python -m src.research.ni_relaxation_repeatability.experiment forces --config config.json
python -m src.research.ni_relaxation_repeatability.experiment quenches --config config.json --lane 0
python -m src.research.ni_relaxation_repeatability.experiment quenches --config config.json --lane 1
```

The initial source snapshot intentionally uses only its in-process backend runner.
The maintained source additionally exposes the standalone LAMMPS-style adapter
CLI and rejects cached quench results from another plan. These interface/provenance
changes do not alter the frozen v1 execution or results; future runs need a new
output and archive identity. All current scientific training recipes stay frozen.

## Findings

All89 controlled quenches completed:32 repeated-recipe quenches,10 float32
input-jitter quenches,45 additional signed-MD endpoint quenches and2 float64
input-jitter extensions. The force audit completed8 independent neighbor rebuilds
for each of5 endpoints under raw32, sorted32 and sorted64, plus permutation and
native CPU reference comparisons. This is one hardware/node and five inputs;
it does not establish universal cross-hardware bitwise reproducibility.

| Independent repeats, five sources | Original float32 | Sorted float32 |
|---|---:|---:|
| Local coordinate component RMS difference |0.0367–0.8331 A|0 on all five|
| Embedding component RMS difference |0.176–1.464|2.94e-7–4.39e-7|
| Exact full-cell float64 coordinate agreement |Not achieved|5/5 cells, all32000 atoms|

All five sorted32 baseline minima met the original0.01-eV/angstrom force
criterion when reevaluated with native CPU EAM; maximum was0.00996913.
At identical inputs, sorting removed force-repeat and atom-permutation differences
exactly, while preserving float32/CPU force RMS agreement (maximum1.29e-5eV/A).
The original force-repeat difference reached1.91e-6eV/A. These numerical changes
are small as force errors but can be amplified during nonlinear FIRE descent.

The tighter sorted32 tolerance0.001 also gave exact full-cell repeats on all5
sources and native CPU maximum force at most0.000996171eV/A. Relative to the
0.01 minima, local coordinate changes ranged0.00247–0.12314A, so tolerance still
changes the chosen observation. Sorted float64 forces gave exact repeats on the
first endpoint, but are not required to fix identical-input repeatability.

Input robustness is a different issue. Sorted32 quenches after local atom-vector
RMS1e-7A perturbations differed by0.02335–0.09449A in local coordinate-component
RMS;1e-4A input perturbations gave0.06907–0.67520A. The first endpoint still gave
0.04363A and0.38740A differences, respectively, with sorted float64 forces.
This demonstrates sensitivity of this FIRE/quench map; it does not identify the
underlying mathematical basin boundary or prove that every minimizer behaves so.

Re-quenching identical saved signed-MD endpoints changed response directions:
old-versus-sorted position-response cosine ranged−0.396 to0.968 and velocity
response cosine−0.162 to0.846. Both were negative on ni-T1150-melt09. The new
target is repeatable under this execution recipe, but is not an exact physical
ground-truth derivative through a globally smooth relaxation map. Existing
training targets retain their original producer and were not replaced mid-run.

For the next target-bank revision, use sorted-neighbor float32 quenches and
separately quantify response stability versus perturbation size. Finite-amplitude
basin changes need not behave like an infinitesimal linear response. Switching
all force arithmetic to float64 alone did not remove the observed sensitivity.

The published [figure](../../output/ni_relaxation/repeatability-20261009-v1/analyses/overview/plots/repeatability.png)
and [exact rows](../../output/ni_relaxation/repeatability-20261009-v1/analyses/overview/tables/quenches.csv)
retain source-level values. Force tables and their frozen definitions are in
`analyses/forces`; native repeat checks are in`analyses/overview/technical`.
The explicit float64 jitter extension and its source hash are under
`technical/precision-jitter*.json` and`technical/precision_jitter.py`.
