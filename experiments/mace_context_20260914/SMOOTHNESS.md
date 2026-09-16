# Embedding smoothness after adding message context

Smooth inner pooling reduces temporal embedding variation relative to the
representation's spread. The tracked-center embedding removes the tested
membership discontinuity but still changes more rapidly along real trajectories
than the original mean embedding. Input continuity and temporal variation answer
different questions.

The comparison uses the same 144 tracked series from six held-out simulations,
17 frames spaced 0.75 ps, and the three completed matched eight-epoch VICReg
continuations. All 42 original temporal scores and 28 controlled-crossing scores
across the seven frozen/trained variants were independently reproduced from the
retained float32 features, using float64 differences and reductions.

Define normalized squared temporal change as
`D(lag) = mean((z(t+lag)-z(t))**2) / mean_channel_var(z_train)`.
The denominator is fitted separately for each representation on training anchors.
It removes uniform feature scaling; this is not per-channel whitening.

| Trained representation | D(0.75 ps) | D(3 ps) | D(12 ps) |
|---|---:|---:|---:|
| Original 80-atom mean | 0.048409 | 0.060391 | 0.108189 |
| Complete context + smooth inner mean | 0.027008 | 0.035642 | 0.075617 |
| Complete context + tracked center | 0.135856 | 0.162495 | 0.234068 |

Smooth inner pooling has **44.21% lower squared change at 0.75 ps** (paired
source bootstrap 95% interval: **42.28–45.88% lower**) and **30.11% lower at
12 ps** (20.20–38.73% lower). In normalized RMS distance, these reductions
are **25.31%** and **16.40%**; squared-distance and distance percentages must
not be interchanged. Each of the six sources improves at 0.75 ps.

The center has **2.81 times** the control's normalized squared change at 0.75 ps
(95% interval 2.64–2.99) and **2.16 times** at 12 ps (1.66–2.85). Its larger
temporal response appears in every source at 0.75 ps. These differences include
physical evolution and cannot be labeled numerical noise.

The scaling check subtracts each training frame's mean embedding from its 64
anchors, then uses the resulting within-frame variance as denominator. This
removes between-frame differences from the scale without altering any temporal
trajectory. At 0.75 ps, control / inner / center become
**0.235920 / 0.148365 / 0.463833**. Inner is still **37.11% lower**
(95% interval 34.94–38.99% lower), and center remains **1.97 times larger**
(1.85–2.10). The direction of both findings therefore survives this check.
Absolute raw feature units differ: the respective raw step MSEs are
7.5225e-6 / 1.4106e-5 / 2.0800e-4. Claims of lower temporal variation here
refer explicitly to variation relative to representation spread.

The controlled 80th/81st atom crossing gives a different result. As the radial
perturbation decreases tenfold, the new representations' squared response
decreases approximately a hundredfold. The original approaches a nonzero plateau.
At epsilon 0.0001 Angstrom, crossing squared change divided by each model's own
natural 0.75 ps change is **0.03671 / 1.41e-9 / 5.40e-10** for control / inner /
center. Both modifications remove the finite jump in these 72 tested crossings;
this does not mean their whole trajectories are millions of times smoother.
The hard-80 TDA target itself retains a crossing discontinuity.

Most of the inner-pooling effect comes from changing the architecture. Its frozen
D(0.75 ps) was already 0.026181, versus 0.027008 after continuation: eight epochs
give no additional short-lag smoothness gain. Center continuation reduces its own
normalized squared change by **53.89%** at 0.75 ps and **49.99%** at 12 ps, while
remaining more variable than the matched control. Within-frame normalization
gives center reductions of 39.12% and 33.98%, respectively. The frozen
complete-context hard-80 mean also has low temporal variation (D(0.75)=0.019786)
while retaining a finite boundary jump, further separating the two diagnostics.

All uncertainty intervals resample six whole sources with replacement 4,000 times
with paired draws; tracks and overlapping windows remain grouped. Training
normalization is fixed. These exploratory pointwise intervals do not quantify
training-seed uncertainty or provide a new untouched confirmation cohort.

For downstream use, smooth inner pooling is supported when the objective is
less relative temporal variation and relaxed patch topology. The center retains
stronger signals for several immediate local physical changes. Lower variation
alone does not establish better forecasts; see the readout tradeoffs in the
[completed pilot report](RESULTS.md). No forecaster was retrained for this analysis.

![Smoothness comparison](../../output/mace_context/forecast-seed20260910-pilot-20260914-smoothness/plots/smoothness.png)

Reproduce with conda `pointnet`:

```bash
python -m src.research.mace_context.run --config configs/analysis/mace_context.json --stage smoothness --device cpu
```

The stage preserves completed outputs and writes a separate run suffixed
`-smoothness`. [Tables and plot](../../output/mace_context/forecast-seed20260910-pilot-20260914-smoothness/README.md),
[paired intervals](../../output/mace_context/forecast-seed20260910-pilot-20260914-smoothness/tables/paired-comparisons.csv),
and [metric definitions](../../docs/metrics/mace_context_smoothness.md) retain the
calculation details, source contributions and implementation/input hashes.
