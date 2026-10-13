# Ni relaxation speed and structural fidelity

Twelve exact full-cell 32000-atom inputs: original Q and one 100fs endpoint from
each of the previous five repeatability sources and the recorded iteration-limit
source. The hard endpoint retains its actual positive-velocity variant. Sources,
checkpoint hashes, atom identities and boxes are pinned. This is a numerical
pilot, not an unbiased population estimate. No birth labels select minimizers.
Each minimization starts independently from its own original input and retains
the fixed periodic cell and Mendelev2012 Ni EAM/FS potential.

Native profiles compare original unsorted FIRE, sorted FIRE, ABC-FIRE, Verlet
FIRE with tmax6, timestep2fs with tmax6, and force tolerances0.03/0.003. Others
use0.01eV/angstrom. ASE unpreconditioned and Exp-preconditioned L-BFGS use the
same Armijo search and maxstep0.1; PreconFIRE uses Exp, Armijo and maxmove0.1.
ASE pilots use cases0,5,11 with a60-second solver budget (checked every10 outer
iterations), and all case processes have a240-second absolute cap. PyAMG5.3.0
is installed in the run directory, not the shared environment. Failed or timed
out cases remain reported; later independent cases still execute.

The separately requested tolerance005 follow-up uses the same twelve inputs
and sorted native FIRE at0.01,0.03 and0.05eV/angstrom. All three settings are
timed anew on one GPU, so speed ratios do not mix hardware from separate jobs.
It uses these same metric definitions and no ASE profiles. These repeats are
numerical timing controls, not additional independent source observations.

All optimizers use float64 coordinates and the same original float32 GPU force
kernels with float64 scalar energy reduction. Sorted neighbors stabilize force
summation order except in the explicitly unsorted control. ASE convergence is
overridden to the same maximum absolute force component as LAMMPS norm inf;
the default maximum atom-vector norm would not be comparable.

`solver_seconds` includes optimizer/context/graph construction and execution,
excluding CUDA-extension warmup, input serialization, endpoint analysis and
archive publication. `callback_seconds` measures callback.compute wall time
including host/device transfers and synchronization. `force_calls` counts all
energy/force evaluations, including line-search and preconditioner setup calls.
`publication_seconds` separately measures checksummed archive publication.
Native logs report exact minimizer iterations. ASE logs and progress are retained.

Every completed optimizer endpoint is independently reevaluated with native CPU
EAM. `cpu_reference_tolerance_pass` permits0.2% rounding slack around the requested
force tolerance; actual float64 CPU fmax is reported. Solver convergence and CPU
reference pass are both required for the summary's converged count. Speed ratios
use only pairs for which sorted FIRE and the candidate converge. Missing, failed
or capped results never count as speedups; attempted counts remain visible.

Endpoint comparisons use original fixed nearest80 identities: periodic centered
coordinate component RMS and frozen relaxed-eSEN128 embedding component RMS.
Full-cell OVITO PTM labels use RMSD cutoff0.1; report exact-label disagreement and
FCC/HCP/BCC fraction. Energy changes are(candidate minus sorted FIRE)/N in
meV/atom. Summary values are medians, not confidence bounds. Different minima
can meet the same force criterion, so speed alone does not authorize replacement
of frozen training targets. Looser-tolerance gains are explicitly different
accuracy profiles. Full-precision minimizer checkpoints and failures are retained
as audit artifacts, not new sampled MD trajectories or shared production minima.
