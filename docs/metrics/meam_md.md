# Matched periodic Al MEAM MD and runtime

One newly melted 16,384-atom periodic pure-Al ancestor is shared by two engine
descendants, not two independent replicas. SHA256-pinned Lee2003 MEAM files are
identical. A native LAMMPS NPT melt at1325K for50ps and NPT cooling/equilibration
at450K for5ps precede the comparison; PTM solid fraction must be below1%.
Production is fixed-volume NVT BAOAB at450K,1fs timestep,100fs friction,200ps.
Initial velocities are freshly sampled once conceptually using PCG64 seed26100423,
with center-of-mass momentum removed. Both engines reproduce these same arrays;
PCG64 seed26100429 supplies identical OU noise at each step. No encoder/predictor
inputs, labels, training, W&B runs or data selection by crystallization results.

Both arms integrate in float64 using the same driver. CPU forces come from the
original LAMMPS MEAM implementation on8MPI ranks; GPU forces differentiate the
ported periodic PyTorch energy. Position RMS is sqrt(mean_i(sum_xyz(difference²)))
in Angstrom, using minimum-image differences between native-precision states.
Unwrapped COM-removed RMS first subtracts mean displacement across atoms.
Velocity RMS uses momenta divided by mass with ASE-unit conversion to A/ps.
Initial coordinates/momenta must be bitwise identical and checkpoint RNG states
must match. Neither metric uses float16 observations. Early checkpoints are
steps0,1,2,5,10,20,50,100,200,500; subsequent checkpoints are every1ps.

Force validation reports absolute total-energy difference per atom and RMS over
3N Cartesian force components. Gates are1e-8eV/atom and1e-8eV/A on the actual
prepared liquid. A separate1ps NVE pair must have maximum absolute total-energy
excursion below5e-4eV/atom at0.1ps observations, and final coordinate RMS below
1e-4A. This is a short-path fidelity and numerical-stability gate, not a proof of
long-time ensemble equivalence. Energy change is final minus initial; excursion
is the maximum absolute difference from the initial observed total energy.

Thermodynamics are sampled every0.1ps. Potential and kinetic energies are total
values divided by N; kinetic energy is sum(p²)/(2m). Temperature removes COM
momentum and uses3N-3 degrees of freedom. MSD uses unwrapped positions relative
to the initial liquid and removes the mean displacement. The thermostat may
change total momentum; total kinetic energy intentionally retains COM energy.
No pressure or diffusion coefficient is inferred from an arbitrary fitting window.

Native-coordinate PTM is evaluated every1ps, RMSD cutoff0.1. FCC/HCP/BCC labels
1/2/3 define the solid fraction; each component is also reported. Largest solid
cluster is the largest periodic connected component with a3.5A distance edge.
This is not an identified birth event or orientation-coherent nucleus lineage.
RDF uses160 uniform bins on[0,10]A, unique periodic pairs, and normalization
2*pair_count*volume/[N*(N-1)*shell_volume]. Initial and latest common checkpoint
RDFs are reported; no interpolation of trajectory histories.

GPU force timing compares eager and torch.compile on the same liquid, with3
repeats of50 force evaluations, synchronized at each timing boundary. Warmup
is separate; exact geometric support updates are included, initialization is
excluded. A static-cell profile does not include moving-cell neighbor rebuilds.
CPU force timing uses the same3x50 external scatter/run0/gather calls and the
maximum wall time across8MPI ranks. Native LAMMPS throughput is a distinct NVE
control,100 warmup steps then3 consecutive1000-step blocks, with normal neighbor
reuse and no per-step Python exchange. It uses450K native sampled velocities,
not the matched BAOAB stochastic path. Do not present external-driver speedup
as native-LAMMPS speedup. Hardware/rank counts accompany each measurement.

Production elapsed time includes stepping, online noise, diagnostics and
checkpoint/archive IO; setup/compilation and final trajectory conversion are
separate. On a resumed segment steps_per_second uses only steps since restart
and elapsed_seconds is that segment's wall time, not a full-run estimate.
Repetitions measure hardware variation, not independent scientific samples.
Completion/partial coverage is explicit. One paired trajectory cannot establish
equal crystallization rates or nuclei-birth probabilities after chaotic divergence.

Position observations are produced as float32 then converted by the maintained
trajectory converter to verified float16 with measured quantization errors.
Boxes are float32, identities/timesteps exact; native integration checkpoints,
momenta and RNG remain float64/exact and are archived independently.

October 5 converter maintenance changes only source retirement on NFS: verified
targets and quantization calculations are unchanged. Open `.nfs` source handles
are recorded as pending retirement, rather than failing after successful export.
Historical exported contracts retain their original implementation hashes.
