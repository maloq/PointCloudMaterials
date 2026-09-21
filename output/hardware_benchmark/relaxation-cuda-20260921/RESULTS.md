# Matched MEAM CPU/GPU relaxation benchmark

Partial until every backend status is complete. One GPU per frame; CPU uses 32 MPI ranks. Both repeats start from identical original MD coordinates. Median end-to-end LAMMPS invocation seconds excludes build, input writing, target extraction, force probe and archiving. Failed attempts remain in receipts; speedups describe successful runs only.

| Case | Backend | Completed | Seconds | Speedup vs same-release CPU |
|---|---|---:|---:|---:|
| liquid-520K | cpu-current | 2/2 | 249.8 | 1.00 |
| liquid-520K | cpu-legacy | 2/2 | 483.2 | 0.52 |
| liquid-520K | h100 | 2/2 | 76.3 | 3.27 |
| transition-520K | cpu-current | 2/2 | 153.4 | 1.00 |
| transition-520K | cpu-legacy | 2/2 | 807.8 | 0.19 |
| transition-520K | h100 | 2/2 | 78.5 | 1.96 |
| post-onset-520K | cpu-current | 2/2 | 80.5 | 1.00 |
| post-onset-520K | cpu-legacy | 2/2 | 95.5 | 0.84 |
| post-onset-520K | h100 | 2/2 | 18.6 | 4.33 |
| liquid-400K | cpu-current | 2/2 | 62.4 | 1.00 |
| liquid-400K | cpu-legacy | 2/2 | 137.0 | 0.46 |
| liquid-400K | h100 | 2/2 | 16.7 | 3.74 |

See tables/throughput-fidelity.csv and tables/METRICS.md for force equivalence and final target differences. Case labels describe sampled local onset context, not a full-cell phase certification.
