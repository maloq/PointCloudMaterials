# Complete response speed: fastest prior precision and MACE

All engines execute on the same dedicated RTX PRO6000 Blackwell GPU. Reuse the
four original train parents0/1/2/45,24A environments, two original full-cell-order
noise seeds,100 BAOAB1fs steps,20/100fs horizons,768 fixed future features and two
analytic response directions. Both batch1 and batch2 process both seeds in each
timed bundle. Seconds/shot is synchronized wall time divided by two. Warm up a
complete bundle per shape/engine; rotate the three-engine timing order across
three repeats. Report each parent/engine/batch median, then the mean over the four
parents. JIT/model initialization is excluded. No training or W&B run.

The prior MEAM and MACE engines use their historical float32 forces, integration,
tangents and features. The new engine uses float32 MEAM force arithmetic but
float64 integration, tangents and observables. Each gets the same original
float32-generated Fourier maps at its declared precision, not newly drawn maps.
MACE and MEAM are different potentials: their runtime comparison is not numerical
or thermodynamic equivalence. Accuracy of the new MEAM against the original
float64 MEAM is recorded separately in the full four-parent response-v2 gate.
Actual tensor roles, potential file hashes and source identities remain the
original response benchmark's registered inputs; no new independent ancestry.
