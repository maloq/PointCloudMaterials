# Local80 response weight and finite-displacement follow-up

The reference bank is unchanged: identity
852f2b168521b5a2ea567aadb7f5a4f4ef826fe52b2c2dc7a9936cae6198c029.
All90 train,15 validation and30 test sources retain their roles and tensors.
This is an exploratory reuse of the previously evaluated cohort, not a new
untouched test set or the downstream all64 crystallization benchmark.

## Training and held-out evaluation

Five arms, each with the same three initialization seeds, run400 full90-parent
optimizer updates. The value term is half-MSE against standardized mean8-shot
future features. The response term is half-MSE of standardized mean8-shot
directional derivatives divided by the original response RMS0.4324815869/Å.
Weights are0,0.1,0.3,1, or delayed0.3. The delayed arm uses zero response weight
for updates1–100, then0.3*(update-100)/50 through150, then0.3. No patience stop.
The original fixed32-shot validation feature Gaussian NLL selects the primary
checkpoint, including initialization. Final400-update weights are a separately
declared diagnostic. Test errors never choose between these endpoints or arms.

`learning.csv`: one row per accumulated optimizer update. Value NLL equals
value half-MSE +0.5*log(2*pi). Response half-MSE is blank while its weight is zero
(not computed). Training losses are measured before the update; validation NLL
after it. Gradient norm is the preclip Euclidean norm, with threshold10.
Training seconds cover synchronized optimization and validation, excluding
checkpoint serialization and W&B/network overhead. Epoch means optimizer update.

`source-errors.csv`: every fit and both declared endpoints use all30 test sources
and disjoint32-shot value /8-shot response streams from the original bank.
Value MSE compares normalized prediction with mean32 values; response MSE
compares predicted JVP divided by the original response scale with mean8
simulator responses divided by feature standard deviation and response scale.
Value NLL is0.5*(value MSE+log(2*pi)), the established fixed-variance feature
mean score, not a calibrated likelihood for complete atomic futures.
Corrected MSE subtracts mean coordinate sample variance(ddof1)/shot_count.
Corrections may be negative; they are not clipped. Scope20fs is coordinates0:128;
100fs_joint is128:256 and describes the joint20+100fs path; full is all256.

`summary.csv`: average the three fits per source, then average equally over
30 sources.95% percentile source-bootstrap intervals use2000 common draws.
`paired-contrasts.csv`: response-arm minus value-only differences with the same
paired draws; negative means lower error. These intervals condition on the
three trained seeds and do not include training-seed uncertainty or adjust for
multiple comparisons. `costs.csv` retains the selected update, all400 completed
updates and synchronized training duration. Per-seed errors remain available.

## Finite disturbances and actual forward MD

Four predeclared training parents, two original noise seeds, two orthonormal
directions, both signs,24/36Å moving environments and six collective amplitudes
yield384 perturbed100fs trajectories plus16 central trajectories. The unperturbed
central execution includes AD for both directions; its256 Fourier values and
responses must reproduce old gate values within absolute RMS1e-6 and1e-5.
All runs use the same MACE-MPA-0 teacher, float32 cuEquivariance force/HVP path,
BAOAB1fs step,450K thermostat and100fs friction time. Original70304-atom CUDA
float64 random draws are gathered to each halo and then cast, preserving common
noise between signs, amplitudes and environments. Reusing train gate seeds
does not create new independent sources. No model receives temperature or time.

`initial-disturbances.csv`: epsilon is the Euclidean norm of the entire initial
displacement, in Å. The central atom is fixed,79 local neighbors are displaced,
their summed displacement is zero, and exterior initial displacements are zero.
Per-atom vector RMS=epsilon/sqrt(79), not epsilon. Atom identities are fixed;
neighbors are not reselected after perturbation. Minimum distance measures any
pair involving the local80 region, excluding self-pairs. Initial distances below
1Å stop the run; amplitudes are never clipped or silently resampled. This check
does not certify thermodynamic plausibility or integration convergence.

`linearization.csv`: actual change is the perturbed future feature minus the
unperturbed future feature under the identical noise realization. Each path's
observable is a change relative to its own initial configuration, matching the
training target. Linear change is sign*epsilon*central AD derivative. RMS pools
coordinates within the named block. Error RMS pools actual-minus-linear;
relative error=norm(actual-linear)/max(norm(actual),1e-12). Cosine uses a1e-24
floor on the product of norms. RFF_20fs and RFF_100fs_joint have128 dimensions
each; radial_20fs/radial_100fs are eight raw radial-density changes each;
angular_20fs/angular_100fs are four raw orientational-power changes each.
These24 physical coordinates do not use the0.1 path scale or Fourier transform.
They are diagnostics, not additional student targets. Differences can reflect
both physical nonlinearity and the Fourier map's nonlinearity.

`physical-response.csv`: at20 and100fs, local divergence is the RMS vector
distance between corresponding79 neighbors relative to their tracked central
atom in perturbed versus unperturbed paths. It includes the initial displacement.
Amplification divides that RMS by epsilon/sqrt(79). Local motion measures the
same relative coordinates against their own perturbed initial positions.
Initial energy change is total open-environment MACE energy difference in eV,
not per-atom energy or thermostat heat. Minimum observed distance is measured
only at the two recorded horizons, not over every integration step. Seconds
is full100fs forward cost, repeated at both horizon rows; do not sum repetitions.
Positions are transient GPU data; only observables and scalar diagnostics persist.

`boundary.csv`: absolute RMS of24Å-minus36Å perturbed RFF values, separately
for each prefix/sign; passed means<=1e-4. `boundary-secants.csv`: each halo's
central secant=(plus-minus)/(2*epsilon); relative error is the norm difference
divided by the36Å secant norm with1e-12 floor, passed at<=5%. A finite secant
is not an infinitesimal derivative at large amplitudes. Failed boundary checks
are retained and flag the24Å interpretation; the campaign does not silently
promote a different radius. Agreement with36Å does not establish exact periodic
full-cell agreement. The figure uses36Å, with median relative error and min/max
across the four parents, two directions, two signs and two shots (not confidence
intervals). Trajectory-divergence curves average that same small panel.
