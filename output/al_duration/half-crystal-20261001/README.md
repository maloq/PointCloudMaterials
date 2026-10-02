# Earlier Al cutoffs when 50% crystallinity is sufficient

Updated user objective, October 1: reaching **50% crystallinity** is sufficient.
The **90% criterion refers to the fraction of other runs reaching that target**,
not to 90% crystalline atoms. Runs still below the target when most peers have
crossed are useful delayed-transformation examples. This supersedes the earlier
duration recommendation that assumed a terminal bulk plateau was wanted.

The audited trajectories have **600 ps measurement + 15 ps equilibration**,
with their original 300 ps melts reused. The numerical results below remain a
counterfactual analysis of that fixed data. Following user authorization,
[a separate queue version](../../../docs/simulations/al_main_half_stop_20261001/README.md)
applies confirmed-halfway stopping to 128 unstarted trajectories. Completed and
already-running sources retain their original protocol and files.

[Cutoff plot](plots/half_crystal_cutoff.png) ·
[Every proposed source cutoff](tables/sources.csv) ·
[Peer cutoffs and censored counts](tables/peer_cutoffs.csv) ·
[Frozen definitions](tables/METRICS.md).

## An individual halfway stop saves much more than waiting for a plateau

Use the already measured exact 15 ps PTM grid. At the first ≥50% sample, wait
for the next check also to be ≥50%, then retain **6 ps** for the existing 3/6 ps
prediction horizons. Thus the proposed stop is **first confirmed crossing sample
+ 15 ps confirmation + 6 ps prediction tail**, capped at 600 ps. No crossing
means the 600 ps maximum remains. There is no terminal-energy, volume or
near-perfect-crystal requirement under this objective.

All statistics below count the full 30-source temperature cohort. “Typical”
conditions on runs with a crossing; non-crossers are reported separately.
Times are measurement times; add 15 ps for equilibration. Monitoring resolution
is 15 ps, so these are conservative sampled crossing estimates.

| Temperature | Runs reaching confirmed 50% | Typical individual stop, including tail | Latest individual event stop | Time when ≥90% of peers are confirmed + tail |
| --- | ---: | ---: | ---: | ---: |
| 400 K | 21/30 | 441 ps | 576 ps | Not reached by 600 ps |
| 450 K | 30/30 | 216 ps | 306 ps | 291 ps |
| 500 K | 29/30 | 216 ps | 546 ps | 411 ps |
| 510 K | 30/30 | 261 ps | 501 ps | 411 ps |
| 520 K | 27/30 | 306 ps | 561 ps | 561 ps |

At a fixed 300 ps cutoff, including the full confirmation/tail, only 1/30
400 K sources are ready, versus 27/30 at 450 K, 21/30 at 500 and 510 K, and
13/30 at 520 K. A single early cutoff is therefore a poor match across temperatures.

With an individual stopping rule and a 600 ps maximum for non-crossers, the
historical mean measured duration becomes **320.18 ps**, saving **46.64%** of
measurement duration, or **45.50%** of integration including equilibration.
This retains all 150 ancestors and includes the 13 no-crossing runs at 600 ps.
No confirmed crossing returns below 50% on the remaining observed 15 ps grid;
the corresponding original fine-grid return checks are exported separately.
Later growth and annealing above 50% is not a failure under the user's criterion.

For the **21 currently complete dense 520 K trajectories**, 19 cross and two do
not. Their median individual event stop is **246 ps**; the last is **426 ps**.
The mean including the two non-crossers at 600 ps is **275.43 ps**: **54.10%**
measurement-duration savings. The 90%-peer cap is 426 ps, after which two paths
remain without a crossing; combining the cap with individual stops would save
**56.86%** within this subset. This is completion-limited evidence, not a forecast
for the other temperatures or an unbiased estimate of all rerun kinetics.

## The remaining runs are censored observations, not permanent non-crystallizers

For 30 same-temperature peers, 90% requires 27 crossing events. The confirmation
times themselves are **285, 405, 405 and 555 ps** at 450/500/510/520 K; the
table adds six ps of retained prediction tail. At 400 K only 21 reach halfway
by 600 ps, so a population 90th-percentile time cannot be estimated within the
observed horizon. Taking the 90th percentile of those 21 successful runs would
incorrectly omit nine delayed/no-crossing histories.

As a distinct mixed-temperature diagnostic, **90% of the full 150-run cohort**
is achieved with confirmation/tail by **576 ps**. At that time 137 are ready
because of ties; **13 have no confirmed crossing** through 600 ps. This overall
time differs from the time for comparable same-temperature peers.

Under the temperature-specific peer caps, three 450 K and three 510 K histories
would be censored despite reaching the criterion later in the original complete
data. At 500 K, two cross later and one has no event by 600 ps. At 520 K, three
have no event by 600 ps. These are all useful observations of delay, but their
labels should mean **“not confirmed by the declared cutoff”**, never “will never
crystallize.” The exported table distinguishes later-known events from events
still unobserved at 600 ps. Prediction targets without a complete future horizon
must be masked rather than converted into negative labels.

Combining individual stops with these temperature-specific peer caps would save
**47.51%** of historical measurement duration. Most of the saving comes from
stopping successful runs at halfway; the peer caps provide a modest additional
saving and preserve a declared observation time for delayed runs.

## Reference-derived next simulation protocol

Use **an individual confirmed-50% stop plus a 6 ps tail**, with an explicitly
declared maximum observation time for delayed sources. The historical peer
calendars suggest planning limits around **300 ps at 450 K, 420 ps at 500/510 K,
570 ps at 520 K**, and **600 ps at 400 K**, whose 90% time remains unobserved.
These were rounded planning proposals, not guarantees that
new runs inherit their original parent's transformation time. The applied queue
uses the exact reviewed caps of 291/411/411/561 ps, retaining 600 ps at 400 K.
A new 520 K
subset already has different timings from its parents.

Retain the delayed source's coordinates, velocities and native restart so it
can support rare-event analysis or later extension. Record whether termination
was a halfway event or a declared cap, the actual final time, the exact observation
cadence and ancestor role. Variable endpoints require a versioned simulation and
conversion contract: the original queue expects exactly 6,001 frames/600 ps and
must not be changed by simply editing a LAMMPS `run` line. The linked implementation
has a separate native endpoint certificate and verified variable-length converter.
This analysis does not recalculate metrics when the queue changes. Scientific data,
including the older full-length controls,
remain intact.

Full-cell PTM is the same physical reference used in the
[original audit](../all150-20261001/README.md), not a thermodynamic volume-phase
ground truth. Inputs are geometry-derived physical measurements; there are no
models, fitted selectors, time/temperature predictor inputs or W&B evaluation runs.
