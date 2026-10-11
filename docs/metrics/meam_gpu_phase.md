# Grown-solid thermodynamic comparison

Use the two matched50ps450K BAOAB descendants of the verified crystal/liquid
interface (mobile MSD101.75A² and0% mobile PTM crystal before cooling). Compare
float32 versus float64 force arithmetic with float64 integration, identical
initial coordinates, velocities, noise and fixed box. Intersect exact1ps structural
and thermodynamic observation times, retain times>=10ps where both cells have
>90% PTM FCC/HCP/BCC atoms. No interpolation, time alignment, outcome weighting
or deletion of the remaining trajectories; full curves are shown in the figure.

Report the arithmetic mean of mixed-minus-float64 potential energy/atom and
COM-corrected temperature differences over that intersection. Number of times
is a correlated observation count, not independent sample size. Descriptive
tolerances are the previously declared0.003eV/atom and5K. No equivalence CI,
melting-point or NPT-density claim follows from passing these tolerances.
Residual interfaces/defects remain; this is a grown-solid control rather than
an independently equilibrated perfect bulk solid. Endpoint crystal fractions
are the last native-coordinate PTM observations of each recorded trajectory.
The temperature/energy/PTM producers retain their frozen meam_gpu definitions.
