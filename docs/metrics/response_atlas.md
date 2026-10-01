# Response atlas feasibility metrics

The fixed protocol is `configs/response_atlas/feasibility_20261001.json`.
This is a numerical/measurement pilot, not the five-arm equal-cost active-learning
comparison. AP and ranking losses do not select any model.

## Existing shooting precision

Reuse the checksum-bound Al480 joint 3/6/12 ps physical targets and historical
source roles. Standardization and the median-distance RFF bandwidth use training
observations only. The 256-feature map is fixed. Source-balanced weighted ridge
regresses the branch-average feature vector; alpha in .001/.01/.1/1/10 minimizes
selection feature squared error, equivalent to fixed-unit-variance Gaussian
feature NLL up to fixed constants. This working feature likelihood is not a full
trajectory density. The prior is the training source-weighted feature average.

For a frozen prediction m and disjoint shot means A,B, corrected squared feature
error is (m-A) dot (m-B), without clipping. Split discrepancy is ||A-B||^2.
Budgets 2/4/8/12 mean 1/2/4/6 shots in each half. Thirty-two partitions are repeated
measurements of the same finite shots, not independent simulation samples.
Tables retain parent-population inclusion weights and equal source mass. The
paired 95% percentile bootstrap resamples whole held-out sources after averaging
partitions. It excludes fitting randomness and does not turn historical sources
into prospective validation.

Retrieval uses fixed present-stratum, cross-source candidates from historical
test sources, without temperature matching. The oracle chooses on one shot half
and scores on the other; predicted neighbors use the frozen ridge mean. Distances
retain finite-shot variance and are not themselves unbiased estimates after
neighbor selection. Saved neighbor identities support inspection of stability.
The sampled query set is fixed across budgets/partitions; its inclusion weights
are exported. No equilibrium frequency is inferred from retrieval queries.

## Analytic mechanisms and toy learning

Gaussian Y=u+exp(v/2)*epsilon uses features [Y,Y^2], exact expectation
[u,u^2+exp(v)], and its analytic 2x2 Jacobian. The controlled surrogate predicts
[u,u^2+1]. For branch errors E[B,M,R], corrected Gram is
((sum E)^T(sum E)-sum E^T E)/(B*(B-1)). Naive Gram is mean E^T E.
The highest eigenvector selects the discovery direction; fresh paired +/- seeds
verify its finite-change response. Negative corrected estimates are retained.
Exact selected error uses the known mean Jacobian; discovery eigenvalues are
not evidence of verification. Fixed budgets have no sequential acceptance rule.

The cancellation diagnostic Y=sin(a+Uniform(0,2*pi)) has constant marginal law
and zero mean derivative; its naive squared branch derivative remains positive.
The eight-coordinate double-well/harmonic system uses a fixed zero-response
surrogate and a disjoint 128-shot reference. That reference is noisy, not analytic
truth. The slow-coordinate fraction is the squared first coefficient of the
selected unit direction in the declared two-coordinate basis.

The six toy scientific fits share 128 fitting, 64 selection and 256 final points,
eight branch labels per fitting/selection point and three paired initializations.
The 32-wide, four-dimensional MLP is a toy capacity exception. Training minimizes
fixed-variance Gaussian feature NLL, with optional response MSE divided by a
training-frozen scalar response scale. Feature scales use fitting data only.
Checkpoint selection always uses selection feature NLL. Exact final value and
Jacobian MSE average coordinates and final points. This is a matched-data
supervision diagnostic, not an equal-cost active-learning result. All scientific
toy predictor fits use online W&B; numerical checks and acquisition diagnostics
remain local.

## Atomistic oracle

256 Al atoms, fixed periodic box, MACE-MPA-0 SHA256 from the recipe, eager float64,
one-femtosecond BAOAB at 450 K, friction time 100 fs. Physical units are A/eV/amu
with time converted through ASE's `units.fs`; these are explicitly mapped into
the reference integrator's consistent-unit equations. Momentum and thermostat
draws are explicit q-independent streams; fixed CUDA layout is recorded.

Parents are controlled displacements from one FCC prototype, development only.
They are not independent melts, equilibrium samples, or a crystallization-rate
cohort. Perturbation directions are Euclidean-orthonormal in Angstrom coordinate
space, with uniform translations removed. Epsilon is total configuration-vector
displacement norm, not per-atom RMS. No relaxation changes an intervention.

Energy and force parity compare against the standard MACE calculator. HVP
finite differences and complete discrete-path JVPs use the same branch streams
within +/- pairs. Relative error is ||FD-AD||/max(||AD||,1e-12). Gate tolerances
and epsilon window are declared in the recipe; a failed gate stops the physical
stage. A whole-lattice-vector atom translation checks periodic image rebuilding.

Smooth radial features use eight Gaussian shells (2..5 A, width .35 A), a
5.5-A compact C2 envelope, and global mean/variance of per-atom shell sums.
The envelope vanishes before any minimum-image discontinuity. RFFs operate on
joint changes from the explicitly differentiated initial anchor, at 20/100/500
steps. All 16 parents have 20/100-step observations; only the first two also have
500-step observations. Four discovery branches and four fresh paired verification
branches per direction are a cost-bounded first pilot, not the proposal's initial
16-branch verification budget or a calibrated acceptance procedure. Each prefix
has 128 fixed RFFs, giving 256 or 384 exported values. Prefix maps are identical
across both protocols; the last block is that protocol's joint-path target.
A fixed .1 physical-feature scale and fixed
random map define this numerical task; neither is learned from evaluation data.

Corrected squared mean response is trace of the off-diagonal Gram. Branch
response variance is the summed unbiased coordinate variance; relative MC noise
is variance/B divided by max(abs(corrected signal),1e-12). This unstable diagnostic
near zero is not a confidence bound. Discovery and fresh FD responses are saved
individually. No clipping or active acceptance is performed.

Cost records include force calls, directional HVPs, discovery and verification
time, and peak CUDA allocation. Per-horizon rows repeat shared whole-path costs:
do not sum those rows to estimate total cost. ASE graph/feature overhead is
included in elapsed stage times; preflight costs are retained separately.
Full-precision query/restart states are preserved; no quantized trajectory is
used for a derivative calculation. Completed and failed partial queries are
published to STORE. No atomistic encoder is fitted at this stage.

Initial and perturbed states must have minimum pair separation >=1.8 A,
-5 < potential energy per atom < 0 eV, and maximum force norm <=20 eV/A.
These declared numerical admissibility bounds do not establish equilibrium
sampling or potential accuracy. Their additional screening calls are separate
from the trajectory force/HVP counts.

Independent FD comparison subtracts both sample-mean variance estimates from
||mean(AD)-mean(FD)||^2, estimating squared difference between the two population
responses at a fixed direction. Negative estimates are retained. The separate
FD signal uses the off-diagonal Gram; neither statistic is an acceptance test.
