# Ni relaxation repeatability diagnostics

The five endpoints are exactly the five sources already used in the October9
repeatability audit. This is a small numerical assay, not a random population
sample, a training run or a revised production recipe. It reuses pre-quantization
float64 signed-MD checkpoints, original nearest80 atom identities and the same
frozen relaxed eSEN checkpoint. It uses no crystallization outcomes for selection.

Independent neighbor rebuilds at identical coordinates measure component-wise
force RMS/max differences, exact equality, neighbor order/membership identities,
and agreement against native CPU EAM. Eight rebuilds compare seven outputs to
the first. Permuting atom array order retains atom IDs and compares restored
force arrays. The canonical GPU variant sorts each directed neighbor list by
atom ID before the original EAM kernels; membership and physical tables are
unchanged. Production callback and kernels remain unmodified.

Two independent FIRE minimizations per endpoint/recipe measure centered nearest80
coordinate-component RMS in angstrom, eSEN embedding-component RMS, total-energy
difference divided by32000 atoms (meV/atom), and native CPU maximum force at the
first converged endpoint. Raw32 and canonical32 use0.01-eV/angstrom tolerance;
canonical32_tight uses0.001. Canonical64 is an explicit force-precision ablation
on the first endpoint only. All FIRE states and energy reductions are float64.
Whole-cell native quench states and float32 local offsets are preserved before
verified float16 global archival conversion. Repeats deliberately bypass the
shared index; completed diagnostic receipts are reused on continuation.
Continuation binds the diagnostic to the plan in its source manifest and reuses
saved native coordinates and float32 patches after dump conversion. Failure
archives live under `failures/`, leaving the completed archive destination free
for a successful retry. Reporting requires the native repeats for each declared
profile; the first-endpoint-only canonical64 ablation remains explicit.

Two deterministic local80 coordinate jitters have atom-vector RMS1e-7 and1e-4
angstrom, with zero mean displacement. Their canonical quench is compared to the
unperturbed canonical minimum; this distinguishes repeatability from sensitivity
to changes in input. Signed response comparisons quench the same two-shot/five-
variant MD endpoint bank using canonical32. Responses are differences between
mean plus/minus embeddings divided by2*epsilon, epsilon0.5, exactly as in the
parent. Report their RMS, difference RMS and cosine against original targets.
No ratio is interpreted as an unbiased signal-to-noise estimate.

Timings use seconds within the actual minimizer call and exclude input writing,
publication, conversion and CPU reference evaluations. Diagnostics may share a
GPU with existing training, so these are not dedicated-hardware speed benchmarks.
Convergence failures are explicit retained failures, not silently excluded data.
