# Original versus dense Al reruns: 20 matched parents at 520 K

The dense reruns are new microscopic paths from the same melt ancestry. They
cannot supply missing frames of the original histories, and original future
labels cannot be copied to the reruns. Keep parent/daughter trajectories in the
same ancestry split; they are not independent melt preparations.

[Open the grouped gallery](../../output/al_replay/completed20-20260929/analyses/paired-physical-v1/index.html) ·
[Per-source results](../../output/al_replay/completed20-20260929/tables/paired_sources.csv) · [Frozen definitions](../../output/al_replay/completed20-20260929/tables/METRICS.md)

## Measured differences

- All 20 pairs have identical prepared-liquid and melt-restart bytes and the same
  recorded velocity seeds and atom identities. Old 3 fs versus new 2 fs integration;
  saved measurement time zero follows 15 ps equilibration in each run.
- Mean same-ID, periodic, box-corrected RMS position difference is
  **3.93 Å at measurement zero** and
  **13.54 Å at 600 ps**. This is configuration
  divergence, not an unwrapped diffusion displacement.
- Shared 12-nearest-neighbor fraction averages
  **25.9% initially** and
  **1.3% finally** (2,048 fixed sampled centers/pair).
- Mean corresponding-atom velocity correlation at measurement zero:
  **0.0004**.
- Mean trajectory-averaged temperature difference, rerun minus original:
  **0.0068 K**;
  paired-source 95% bootstrap interval **[-0.052165493105319545, 0.06432200789369093] K**.
- Final crystalline fractions average **77.7% original**
  versus **79.4% rerun**. Mean paired change:
  **1.71 percentage points**, with
  95% source-bootstrap interval **[-16.12, 19.79] points**.
- **18/20 original** and **18/20 rerun** histories
  reach a confirmed sampled crystal fraction >=50%. This requires the following
  structural sample also to exceed the threshold; structural resolution is normally 15 ps.
- At exactly 600 ps, the >=50%-crystalline classification differs for
  **4/20 pairs**: source IDs **[864, 865, 872, 877]**.
- Among the **16 pairs** with confirmed t50 in both histories, the
  median absolute t50 shift is **142.5 ps**.
  This conditional descriptive statistic excludes unconfirmed/censored pairs.
  The rerun reaches t50 earlier in **12/16** of these pairs.
- Trajectory-averaged potential energy changes by
  **-13.69 meV/atom** (new minus old;
  paired-source 95% interval **[-25.88, -0.55] meV/atom**).
  Earlier crystallization is a possible contributor. This is a kinetic-distribution
  difference in this completed subset; close final averages do not rule it out.
- Mean absolute crystal-curve difference: **0.266**
  for matched parents versus **0.256** averaged over
  shuffled parent assignments. Lower-tail pairing permutation p:
  **0.6249**.

## Figures

![Overview](../../output/al_replay/completed20-20260929/plots/overview.png)

![All paired crystallization curves](../../output/al_replay/completed20-20260929/plots/paired_crystallization.png)

![Opposite final outcomes](../../output/al_replay/completed20-20260929/plots/opposite_outcome_examples.png)

The two illustrated extreme examples were selected after examining final
potential-energy differences, not as representative examples for estimating
population differences. Parent 865 changes from essentially noncrystalline to
92.4% crystalline, while 877 changes from 93.5% to essentially noncrystalline.
PTM Other includes disordered environments and crystal defects; it is not by
itself a proof of liquid state.

New-run conversion receipts record at most 0.03125 Å coordinate rounding error,
far smaller than the observed cross-run atomic separation.

## What this comparison can establish

Atomic paths and some same-parent crystallization outcomes differ strongly.
Similar mean thermodynamic/final-order observables, if seen, do not establish
protocol equivalence. One old and one new realization per parent cannot isolate
timestep bias from numerical trajectory divergence. A controlled same-timestep
rerun baseline and replicated 2 fs/3 fs branches would be needed to isolate that.
The completed subset contains only 520 K and is completion-limited; it cannot
establish a conclusion over 400–520 K. Source bootstrap intervals are descriptive
for these 20 ancestors, not proof of equivalence.

Both paths are matched on 401 exact times 0:1.5:600 ps with no interpolation.
Full-cell PTM is recomputed from both float16 exports with RMSD cutoff 0.1.
Crystal means FCC/HCP/BCC; largest crystal cluster uses 3.5 Å connectivity.
The structural grid is every 15 ps plus 1.5, 3, 6 and 12 ps. The same-parent/shuffled
control compares curve MAE with equal weights on those declared samples.

This is a descriptive simulation audit, including some held-out ancestry, with
no encoder/predictor fitting, selection or source reassignment. All scientific
input identities, completion exclusions, implementation hashes and per-pair rows
are under `technical/`; original simulation campaigns and artifacts are unchanged.
