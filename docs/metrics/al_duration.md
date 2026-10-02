# Al duration audit

Descriptive physical audit of the frozen 150 independent Al ancestors (30 each
at 400, 450, 500, 510 and 520 K) and the explicitly recorded complete dense
descendants. No fits, split changes, stopping decisions or simulation changes.
The two supplemental 0.01-ps trajectories are excluded. Old/new daughters share
ancestors and must not be counted as independent replicates. Dense coverage is
completion-limited at 520 K; other temperatures have only historical evidence.

## Observations and provenance

Historical `crystallization_progress.npz` and `thermodynamics.npz` are verified
against their original outcome hashes, and binary manifest hashes against the
frozen main campaign. Existing producer: `independent_meam_source._ptm_progress`.
PTM used original native text coordinates, RMSD cutoff 0.1, periodic full cells;
crystal is FCC/HCP/BCC. Largest selected crystal cluster uses 3.5 Å connectivity.
New trajectories use the identical physical definition on verified float16
binary inputs. Reuse completed replay observations only when their recorded
input manifest hash matches; otherwise recompute missing full-cell PTM samples.
The paired replay's original saved contract and pair hashes are retained.

All primary comparisons use exact structural times 0,15,...,600 ps (41 samples).
No interpolation. Fine original crystallinity at 0.75 ps is exported separately;
it does not determine the primary landmarks. Crystal fraction and largest cluster
fraction divide by 70,304. FCC/HCP/BCC/ICO/Other fractions retain the producer's
definitions. **Other includes defects, strained interfaces and liquid; it is not
itself a liquid fraction or evidence of persistent liquid pockets.** The final
plateau is therefore not expected to reach 100% PTM crystallinity.

Energy is potential energy in eV/atom; volume is Å³/atom. At each structural time
these are trailing 15-ps means over actual logged samples in (t−15,t], with t=0
using the single zero observation. Historical samples are 0.75 ps; dense samples
are 0.1 ps. Values are averaged within each trajectory, never interpolated or
pooled across sources. These are physical audit observables, not model inputs.

## Source and temperature tables

`t50/t70/t80/t90_ps`: first structural sample reaching the threshold and remaining
above it at the following sample. Resolution is 15 ps. Missing means no confirmed
crossing, including endpoint-only crossings; it is not time zero. Median and
latest crossing times condition on confirmed events and are not unconditional
survival quantiles. First majority crystal crossing is not transformation completion.

Terminal reference: equal mean over the five structural times 540,555,570,585,600 ps.
Terminal crystal/energy ranges are max−min on those times. `near90_count` and
`majority_count` apply ≥90% and ≥50% to terminal mean, respectively. HCP/Other
terminal means describe stacking/defect/disorder content, not a phase diagnosis.
`still_moving_last60_count` counts crystal ranges >2 percentage points.

`settled_time_ps` is a **retrospective bulk plateau diagnostic**, not ground truth
for a fully solid cell and not an online stopping time. It is the first exact
sample with ≥70% crystal, ≥98% of crystal atoms in the largest connected cluster,
and **every remaining observed sample** within ±3 percentage points crystal,
±5 meV/atom energy and ±0.5% terminal volume of the terminal reference. Require
at least 60 ps remaining follow-up, so only times ≤540 ps qualify. This measures
bulk-state consistency through the observed endpoint; changes after 600 ps are
unknown. Missing combines untransformed, still changing and insufficiently
confirmed cases, and must not be relabeled as liquid. The thresholds are declared
physical sensitivity criteria, not fitted proof of complete crystallization.

Temperature aggregation gives each source equal weight. `settled_by_*_count`
counts confirmed plateaus by the exact cutoff. `terminal_fraction_gain` is terminal
reference minus cutoff crystallinity; `maximum_later_fraction_gain` uses the
largest later structural sample. `gain5pp_after_*_count` counts terminal gains
strictly >5 percentage points. Potential-energy change is signed terminal minus
cutoff in meV/atom. These diagnostics quantify information a fixed shorter run
would miss. Counts retain all 30 ancestors at every temperature.

## Hypothetical causal stopping sensitivity

Separate explicit crystal floors 70%,80%,90%; none are selected as optimal.
At each observed time before 600 ps, consider the trailing **120 ps** (60 ps
plateau plus 60 ps stable retention tail, inclusive nine structural samples).
All samples must exceed the floor, have ≥98% of crystal atoms in the largest
cluster, and total observed ranges ≤2 percentage points crystal, ≤3 meV/atom
energy, ≤0.3% mean window volume. First passing time is a hypothetical causal
stop: only then-available values determine it. No passing time means 600 ps
retained and zero saving. Actual simulations continue unchanged.

Audit **later** values against the mean over the last five samples through that
hypothetical stop, using ±3 percentage points, ±5 meV/atom and ±0.5% volume.
`later_change` is any later violation in any of the three observables;
`later_fraction_increase/deviation` and energy deviation report largest later
differences. This separates a seemingly stable interval from validated absence
of subsequent observed growth/annealing. Earlier small nuclei and local interfaces
may remain dynamic even when all bulk criteria pass. At late stops little future
evidence remains; post-600 changes are never assessed.

`saved_ps` sums 600−stop only for passing sources; `saved_fraction` divides by
600 times the full group count. These are counterfactual measurement-duration
savings, not measured wall-time gains. Equilibration, conversions, restart/IO cost
and heterogeneous hardware are excluded. No uncertainty interval or independence
claim is inferred by pooling parents and daughters.
