# Known-nucleus saved-state recovery, v1

This is a simulation diagnostic, without encoder/predictor fitting, checkpoint
selection or AP optimization. The fixed original preappearance test cohort has
15 isolated establishments from 11 independent source ancestors. A separate
precision panel contains two retained float32 shooting births from distinct
ancestors; it is a small causal rounding diagnostic, not a population estimate.

## Targets and matching

PTM RMSD cutoff 0.1 defines FCC/HCP/BCC atoms. Periodic connectivity uses 3.6 Å,
tracked clusters contain at least four atoms, and strong temporal overlap
requires four shared atoms and at least 25% of the smaller component. The
historical ancestry implementation retains at most one missing observation.
Establishment requires at least 64 atoms for a 1.5 ps persistence span: three
samples at the original 0.75 ps cadence, six at shooting's 0.3 ps cadence.
These observation protocols are reported separately and never interpolated.

Each continuation inherits the entire saved cluster graph strictly before its
start. Future clusters use new native coordinates before archive quantization.
Historical established ancestry, merges, interface association and ambiguous
periodic extent retain the established detector's exclusions. A new recovery
must start strictly after the snapshot and be an isolated establishment.
Already present or inherited crystals do not count as a recovered birth.

The original birth core is the exact set of atom identities in the saved
cluster at its first qualifying frame. Its centroid in a continuation is
computed from those same advected atoms, using periodic minimum images.
`same_site` means a new isolated birth whose centroid lies within 8 Å of that
advected core centroid, confirmed by the original confirmation time plus 3 ps.
`same_core` instead requires four shared atoms and 25% of the smaller original
and new birth core; it does not additionally require the centroid condition.
`any_new_nucleus` ignores site/core correspondence. `same_site_on_time` also
requires the new birth time to differ from the original by at most ±3 ps.
`earliest_same_site_error_ps` is signed new minus saved birth time, blank if
unrecovered. These operational births are not critical-nucleus estimates.

`initial_site_crystal_atoms` counts PTM-crystalline atoms among original birth
core identities at restart, not all atoms in an 8 Å sphere. Near-birth offsets
can contain embryos. `preappearance` uses the earliest eligible appearance
among the existing positive rows for that event (tie: atom ID, then row ID),
minus 0.75 ps. That selected original local observation is crystal-free;
the full cell and the entire eventual birth core need not be crystal-free.

## Rates and uncertainty

For each panel/offset/variant/arm, a rate is the unweighted mean of event
recovery fractions. Only events with all four planned stochastic continuations
completed enter that rate; the deterministic NPT arm has one continuation per
event. Branch counts, observed events and fully completed events are explicit.
Partial coverage is preliminary, not a cohort-wide result. Source-ancestry
bootstrap intervals use 2,000 draws, sampling ancestors with replacement and
retaining all their events and branches. Blank intervals mean insufficient
completed events. Four noise branches are not four independent ancestors.

The 450 K sources share one physical BAOAB execution for `sourceT` and
`response450K`. The two labels are never pooled as independent observations.
The paired precision table subtracts the original-float32 branch indicator
from its rounded-Q, rounded-V or rounded-QV partner with the identical initial
source/start and future torch CUDA Gaussian noise seed. A rounded position is
the absolute box-relative coordinate cast float32→float16→float32; velocity
rounding occurs in its native Å/ps units. These isolate added rounding within
the declared simulator, not missing precision in old float16 trajectories.

## Dynamics and divergence controls

BAOAB uses the response experiment's 1 fs timestep, 100 fs friction and fixed
box, actual saved velocities, and either 450 K or the source temperature.
Positions, momenta, noise and force arithmetic are float32; the shared backend
scalar energy reduction is float64. No velocity redraw, relaxation or added
displacement occurs in the as-saved arm. The NPT control uses native LAMMPS
float64 integration, float32 GPU force arithmetic, 3 fs, original source
temperature, 0.3 ps thermostat, 3 ps isotropic barostat at zero pressure and
momentum removal every 100 steps. Its missing bath state is reset. It therefore
matches original settings but is not an exact native restart or a pure
quantization control. Original shooting used another Langevin discretization;
the precision panel preserves that fact and tests paired BAOAB continuations.

RMS position divergence uses per-atom minimum-image differences of fractional
coordinates, multiplied by the original box lengths; it removes homogeneous
box deformation and is reported for all atoms and the original birth core.
It does not remove translation or align structures. `final_core_rms_A` uses
the last observation. Exact trajectory agreement is separate from recovery
of a nucleus or its conditional probability. Finite negative outcomes do not
establish that saved states contain no predictive information.

Raw target/source manifests, graph hashes, atom IDs, native cadence, actual
offsets, source roles, potential hashes and code snapshots accompany results.
The original stored graph is required to reproduce each chosen historical
birth before launching its continuations. Archived trajectories are verified
float16 via the maintained converter; precise latest Q,V/RNG or native NPT
checkpoints and pre-quantization PTM labels are preserved separately.
