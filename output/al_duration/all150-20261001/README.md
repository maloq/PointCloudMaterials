# How long do the main Al trajectories need to be?

**Updated user target, October 1: 50% crystallinity is sufficient.** The
[half-crystal follow-up](../half-crystal-20261001/README.md) evaluates much earlier
event-based cutoffs. The bulk-plateau recommendation below applies to the earlier
question about completing the transformation, and is retained with its original
analysis. It does not require waiting for a terminal plateau under the new target.

Each of the **150 dense Al reruns** measures **600 ps**, following **15 ps of
equilibration**: **615 ps of newly integrated dynamics per source**. Each has
70,304 atoms and saves 6,001 position/velocity frames at 0.1 ps. The original
300 ps melt is reused, so it adds ancestry/history, not rerun computation.
There are 30 independent melt ancestors at each of 400, 450, 500, 510 and 520 K.
The two additional 0.01-ps trajectories are separate observations of existing
ancestors and are excluded here.

**Recommendation: retain the 600 ps maximum and the current queue. A common
300–450 ps cutoff would lose substantial crystallization dynamics.** Some runs
settle earlier, but a seemingly convincing plateau can precede another growth
or annealing episode. No jobs, trajectories or duration settings were changed
by this audit.

[Temperature curves](plots/duration_by_temperature.png) ·
[Cutoff risk](plots/cutoff_risk.png) ·
[Source-by-source results](tables/sources.csv) ·
[Frozen metric definitions](tables/METRICS.md) ·
[Exact inputs and protocol](technical/protocol.json).

## Evidence across all 150 ancestors

Reuse all 801 historical full-cell PTM observations per source, every 0.75 ps,
and their original thermodynamic measurements. Verify each small artifact's
SHA256 against its original producer outcome. Primary comparisons use the same
exact 15 ps structural grid for historical runs and dense descendants; no
interpolation. Twenty complete dense histories reuse the already verified
September 29 physical replay analysis; source 876, completed after its native
restart continuation, receives fresh PTM on the same grid. This provides **21
complete dense trajectories, all at 520 K**. Availability is frozen in the recipe.

“Bulk settled” below means ≥70% PTM crystal, ≥98% of those atoms in the largest
connected crystal cluster, and all remaining measured states close to the
540–600 ps reference: within 3 percentage points crystallinity, 5 meV/atom energy,
and 0.5% volume. At least 60 ps follow-up is required. It is a retrospective
diagnostic, **not proof that every atom is solid or an executable stopping rule**.
Potential energy and volume use trailing 15 ps means.

| Temperature | Confirmed ≥50% crystal by 600 ps | Bulk settled by 300 ps | Bulk settled by 450 ps | >5 percentage points further growth after 450 ps |
| --- | ---: | ---: | ---: | ---: |
| 400 K | 21/30 | 0/30 | 9/30 | 14/30 |
| 450 K | 30/30 | 13/30 | 24/30 | 3/30 |
| 500 K | 29/30 | 10/30 | 25/30 | 2/30 |
| 510 K | 30/30 | 7/30 | 22/30 | 5/30 |
| 520 K | 27/30 | 4/30 | 15/30 | 7/30 |

Further growth in this table compares the cutoff with the **mean during
540–600 ps**, which avoids treating one endpoint fluctuation as the terminal
state. Summed across sources, a 300 ps cutoff misses >5 percentage points of
such growth in **87/150** runs; at 450 ps it misses **31/150**; at 525 ps,
**13/150**. Nine of the last group are at 400 K. Comparing instead with the
single 600 ps endpoint gives **89/150, 35/150 and 19/150**, respectively; these
counts can be reproduced directly from `original_fine_crystallinity.csv` by
subtracting the exact cutoff sample from the exact endpoint. The difference
itself reflects appreciable growth during the terminal window in some sources.

Only **34/150** meet the retrospective bulk consistency criterion by 300 ps,
**95/150** by 450 ps, and **120/150** by 525 ps. Overall, 137/150 reach confirmed
majority crystallinity. At 400 K, confirmed majority crossings extend to 555 ps
on the 15 ps grid, and **14/30** change by more than two percentage points during
540–600 ps. This temperature has a particularly long, variable transformation
tail; 600 ps is not enough to establish a terminal plateau for every source.
Do not discard an untransformed liquid history merely because its bulk energy
is temporarily flat: its survival time is useful crystallization information.

## Check against the new dense dynamics

For the 21 complete 520 K dense runs, 19 reach confirmed majority crystallinity;
17 meet the retrospective consistency criterion by 450 ps. Two still gain
>5 percentage points after 450 ps relative to their terminal mean. None do after
525 ps relative to that mean, although source 876 continues substantial change
inside the terminal window. The source set is completion-limited, and the other
temperatures have no complete dense descendants in this capture. Historical
durations therefore cannot be used as individual stopping schedules for the new
paths. The earlier [paired audit](../../al_replay/completed20-20260929/README.md)
already established substantial same-parent trajectory and transition-time
divergence when changing integration from 3 to 2 fs.

## A long apparent plateau can still be misleading

Audit a causal candidate using only values available at each hypothetical stop:
60 ps of plateau followed by 60 ps of retained stable observations. Across the
entire 120 ps, require ≥70% crystal, connected crystal share ≥98%, total crystal
range ≤2 percentage points, energy range ≤3 meV/atom and volume range ≤0.3%.
Then examine the later, deliberately retained dynamics with the wider
3-percentage-point / 5-meV / 0.5%-volume tolerances. Separate 80% and 90%
crystal floors are sensitivity checks; no floor is selected or deployed.

| Crystal floor | Historical early stops | Stops missing later changes | Hypothetical saved measurement time |
| --- | ---: | ---: | ---: |
| 70% | 88/150 | 9/88 | 14.95% |
| 80% | 64/150 | 5/64 | 10.83% |
| 90% | 27/150 | 1/27 | 3.65% |

At 70%, the dense subset would stop 17/21 early and save 22.5% of measurement
time, but **2/17 stops miss later changes**. The clearest example is **dense
source 876**: the rule passes at **405 ps**, yet later crystallinity rises by
**9.78 percentage points** and potential energy changes by **12.19 meV/atom**
relative to its pre-stop reference. Historical source 862 also passes at 315 ps
before another 6.37-percentage-point increase. Late improvement of crystal order
and defects matters for the interface representations being studied.

Savings here are counterfactual integration durations, not measured wall-clock
speedups: equilibration, conversion, I/O and scheduler overhead remain. Failures
are observed only through 600 ps. A stop near 600 ps has very little future
follow-up; passing it does not establish long-term stability.

## Consequence for the campaign and encoder research

Keep 600 ps as the common maximum for this frozen collection, especially for
400 K and sources that remain disordered or are still growing. At 450–510 K
many sources have settled bulk observables before 450 ps, so **selective savings
are possible in principle**. The tested bulk rule is not reliable enough to
implement automatically. A future shortening protocol should first distinguish
mobile liquid pockets from crystal interfaces/defects and validate late local
rearrangements on retained full-length controls; any fitting/selection of that
rule should use training ancestors, with separately held-out validation.

Do not demand 100% PTM crystal as completion: the terminal median crystal
fractions are about 75–87% across these historical temperature groups. Thermal
disorder, interfaces and defects can contribute to PTM Other, while HCP can
capture stacking faults. **PTM Other is not a measured liquid fraction.**
Conversely, a bulk plateau is not evidence that all local environments or
interfaces have stopped changing. The audit does not assign a precise full-solid
completion time or detect the absence of all liquid pockets.

Reproduce with conda `pointnet-torch214` using the
[research protocol](../../../experiments/al_duration_20261001/README.md).
The exported CSVs retain calculation hashes and definitions. No encoder,
predictor, probe or W&B run is created by this local physical diagnostic.
