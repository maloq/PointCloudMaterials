# CUDA MEAM native-coordinate diagnostics

Read only the full-precision archived checkpoints of the shared16,384-atom
unseeded liquid descendants. For each GPU engine, intersect exact integer step
IDs with the CPU trajectory; never interpolate or read quantized observations.
Coordinate RMS is sqrt(mean((qGPU-qCPU)^2)) over all3N unwrapped coordinates,
without alignment, COM subtraction or minimum-image remapping. Long-time
chaotic separation is expected even for the two float64 engines and is not an
equilibrium thermodynamic discrepancy. Initial zero is displayed at1e-16A on
the logarithmic plot; the exported table retains exact zero.

RDFs use the latest checkpoint shared by all three engines, wrapped periodically
into the identical original box. Count unique pairs within10A in160 uniform
bins. g(r)=2*count*V/[N*(N-1)*shell_volume]. The midpoint is the reported radius.
These same-ancestor trajectories are not independent melts. The comparison is a
single-time structural diagnostic, not an ensemble confidence interval, birth
probability or melting-point validation. All dates/cadences and potential
identities remain those of the original registered collections.
