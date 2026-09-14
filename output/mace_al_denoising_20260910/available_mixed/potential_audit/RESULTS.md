# Matched potential audit

Target-potential sensitivity on MEAM-generated structures at 400, 450 and 510 K. Both labels use FIRE, identical boxes/centers/neighbor IDs and convergence threshold. This does not isolate the effect of generating dynamics under different potentials.

Nine independent sources; 256 identical neighborhoods per source; same FIRE procedure for both potentials.

Predictor: atom_temporal_blocks, seed 20260910 selected on the earlier validation split.

MEAM-target balanced MSE: 0.093581; EAM-target MSE: 0.849230.
EAM minus MEAM relative error difference: 807.48% (source bootstrap 95% interval [782.39%, 834.09%]).
Exact paired two-sided p = 0.00390625, using 512 assignments. Primary alpha: 0.05; prespecified practical effect: 5% relative error.

The test unit is an independent melt source. Neighborhoods are averaged within each source, not treated as independent replications. Secondary H0/H1/H2 tests use Holm correction.

This tests sensitivity of the relaxed target and prediction error to the potential on fixed observed inputs. It does not test how the potential changes the distribution of generated trajectories. Nonsignificance does not establish equivalence.

[Full metrics](metrics.json). [Paired permutation method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
