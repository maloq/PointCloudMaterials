# CUDA differentiable Lee2003 Al MEAM

This engineering comparison preserves the SHA256-pinned pure-Al Lee2003 2NN
MEAM energy, pair interpolation and all angular/geometrical screening terms.
CUDA selects discrete neighbor/screening support; PyTorch differentiates every
continuous term. Stable polynomial evaluation changes floating-point rounding,
not the polynomial. Force Jacobian products use the symmetry of the energy
Hessian and compiled reverse-over-reverse AD. They are not finite differences.

Precision names describe force arithmetic. Both modes integrate coordinates,
momenta and displacement tangents in float64 and reduce scalar energy in float64.
`cuda-float32` uses float32 density/force arithmetic, not float64 force accumulation.
A conservative CMIN/CMAX boundary guard may retain extra candidates; the energy
still applies the exact physical cutoff. Overflow raises instead of truncating.

Periodic force timing: the shared native 16,384-atom liquid, one warmup plus30
synchronized individual calls. Median seconds include support updates, not
initialization/JIT compilation or moving-cell list rebuilds. Energy error is
absolute difference per atom; force RMS averages all Cartesian components.
Virial trace is minus the derivative of energy with respect to uniform cell and
coordinate scaling. Configurational pressure is trace/(3V)*1.602176634e6 bar.
Mixed pressure error is against the same energy differentiated in float64.
This alone is not a pressure comparison against native LAMMPS or an NPT test.

Complete response timing uses all four original training parents0/1/2/45,
24A open environments, the central80 displacement support, two saved original
full-cell-order noise seeds,100 BAOAB1fs steps,20/100fs observations,768 fixed
features and two tangent directions. There is no environment cropping to80
simulator atoms. Batches1/2 cover the same two seeds. Each timing sample is a
complete two-shot bundle; seconds_per_shot divides by2, never by batch size.
The three-repeat median excludes a complete warmup. Old MEAM and both CUDA
precisions run on the same GPU, with the same float64 states/features/noise.
Relative errors are ||candidate-reference||2/||reference||2; RMS averages all
entries. No AP objective, trained encoder, predictor or online W&B run is involved.
The earlier response-v1 development output timed only seed0 for batch1 and is
excluded from the final comparison; its source and outputs remain preserved.

The1ps NVE gate compares mixed forces/float64 states with the archived CPU
trajectory using identical initial positions and momenta. Energy excursion is
max|E(t)-E(0)|/N at0.1ps observations; coordinate RMS uses native unwrapped final
coordinates. Passing this short gate does not imply identical chaotic paths.
Its exact executed source is retained with that gate, including retired helper
classes, separately from the final implementation.

Production uses the same450K,1fs,100fs-friction BAOAB NVT and exact PCG64 streams
as the prior CPU/float64 GPU comparison. One200ps mixed-force descendant shares
that ancestor; it is not another independent melt. The explicitly seeded
interface is a separate new preparation:16^3 FCC unit cells at4.10A, one slab
frozen during20ps2500K partial melting, followed by an immediate450K quench with fresh velocities. The1325K preparations were rejected: even the apparent37.4% disorder before cooling had mobile MSD0.219A², indicating a hot crystal rather than liquid. They remain archived as hot-crystal cooling controls. The final mobile region must have MSD>4A² and PTM crystalline fraction<10%; measured values are101.75A² and0%. All atoms are released
in production. Its mixed/float64 descendants use matched50ps BAOAB trajectories.
Seeded growth must never be called spontaneous nucleation.

Thermodynamic quantities, PTM fractions, largest distance-connected solid cluster,
MSD and RDF reuse the documented meam_md definitions and unchanged producer.
Structural observations use native coordinates at1ps, not quantized positions.
PE and temperature comparisons restrict to exact common observation times after
10ps at which both structural samples have <1% FCC/HCP/BCC. Mean differences
are descriptive tolerances, not an equivalence test or independent-replica CI.
Seeded trajectories are plotted separately; differing interface positions are
not evidence of different equilibrium phase thermodynamics. Fixed volume gives
identical density by construction and is not a density-validation result.
Pressure/EOS, melting point and independent-ancestor nucleation statistics remain
unresolved until directly measured. Never claim full thermodynamic equivalence
from force agreement, one NVE trajectory, or liquid mean agreement alone.

Native Kokkos is a separate force-only reference, pinned upstream commit and
binary, float64, Blackwell120. Timing uses100 warmup then3x1000 NVE steps, with
native neighbor reuse and no per-step Python force exchange. It does not support
our response AD. Native CPU timing is the preserved prior benchmark on8CPU
ranks, with its own hardware record; do not describe it as a same-device ratio.
Production steps/s includes noise, stepping, native checkpoints and PTM/output;
compilation and final trajectory conversion are reported separately.

New position observations are verified float16 after the maintained conversion,
boxes float32 and identity/time arrays exact. Full-precision restarts and RNG
states are preserved. Source hashes, configuration, GPU, and execution receipts
accompany exported metric tables. Partial reports explicitly retain run status.

October 5 converter maintenance changes only source retirement on NFS: verified
targets and quantization calculations are unchanged. Open `.nfs` source handles
are recorded as pending retirement, rather than failing after successful export.
Historical exported contracts retain their original implementation hashes.
