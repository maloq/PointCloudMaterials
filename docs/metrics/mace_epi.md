# Paired MACE alignment/Epi training metrics

Protocol `mace_paired_epi_v1`, direct128-dimensional raw invariant encoder export.
For B current/next pairs, I=mean((z0-z1)^2). For each view, center over B,
covariance=ZᵀZ/(B−1), V=mean(relu(1−sqrt(diag(cov)+1e-4))) and
C=sum(offdiagonal(cov)^2)/128. V,C are averaged over the two views.
VICReg=(25I+25V+C)/51. Epi=25I/51−0.1E/E_initial. The Epi-variance
treatment adds25V/51. Reported `terms` already include these coefficients.

E uses the unchanged `regularization.objective.epiplexity`: center each learned
view and divide by its global RMS with epsilon1e-4; standardize frozen random
reservoir channels (std floor1e-6), divide by sqrt64, fit ridge rho3,
then E=0.5 log2 det(I+30WWᵀ). Ridge/logdet are FP64. Average both views;
E_initial averages their scores over four fitting-only B512 batches at the
initial encoder. This is a geometric reservoir adaptation, not a claim to
reproduce the original image-based Epi experiment.

`epoch`=completed updates/64; `anchor_exposures`=updates×512. A full pass is
one permutation of all32768 fitting anchors with no repeated/missing rows.
`loss` is the sum of active terms; a negative Epi loss is expected and cannot
be compared numerically with a VICReg loss as a quality ranking.
`gradient_norm` is the pre-clipping parameter gradient L2 norm (clip5).
`export_rms` is the uncentered state RMS; `projector_std` and
`projector_participation_ratio` retain the reused VICReg diagnostic names but
refer to the direct encoder output, since no projector exists here.
`vicreg` diagnostics are the hypothetical unnormalized25I+25V+C sum for all
arms; they are not necessarily the optimized objective. No physical metadata
or future structural labels enter this objective.

Structural/future evaluation retains the separate `encoder_screen` and
`encoder_parameter_search` metric definitions and hashes. Their source
bootstrap intervals do not represent variation between encoder training seeds.
