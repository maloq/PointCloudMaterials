# Al cutoff at 50% crystallinity

User criterion, 2026-10-01: 50% crystallinity suffices; complete solidification
and terminal annealing are no longer prerequisites for this duration decision.
This follow-up uses frozen observations from the all150 duration audit, checking
their CSV hashes against the original table bindings and recording that audit's
contract and protocol identity. Its earlier full-plateau analysis remains intact.
No simulation jobs, saved trajectories or original metrics are changed.

PTM crystal is FCC/HCP/BCC, RMSD cutoff 0.1, all 70,304 atoms. This is a local
classification fraction, not an independent thermodynamic solid volume fraction.
Old native-coordinate observations and dense float16 observations preserve their
original provenance; see the frozen `al_duration` description. There are 150
historical ancestors (30 each at 400/450/500/510/520 K), and 21 complete dense
descendants at 520 K. Daughter paths are not independent additional ancestors;
the dense subset is completion-limited and cannot establish other-temperature
cutoffs or individual new-path stopping times from old-parent timings.

## Proposed causal rule, evaluated counterfactually

Check the whole-cell PTM fraction at exact 15-ps measurement times. `t50_observed_ps`
is the first sample ≥50% followed by another ≥50% at the next check. The actual
decision is made at `confirmation_time_ps = t50_observed_ps + 15`, not retroactively
at the crossing sample. Then retain **6 ps of further dynamics** for existing
3/6-ps crystallization-prediction horizons. Proposed stop = min(600,t50+15+6) ps.
No confirmed crossing means the full 600 ps remains. Equilibration adds 15 ps.
These are proposed variable durations, not durations of the existing files.

The monitoring resolution is 15 ps; the crossing can occur between checks.
Both consecutive checks are required, without a plateau, energy criterion or
requirement for near-perfect crystal order. No numerical threshold is fitted.
Original fine t50 is a separate diagnostic using the same two-sample rule at
the actual 0.75 ps cadence; it does not determine the proposed shared stopping
rule. Fine confirmation therefore has different physical duration from the
primary rule and is never called a matched stopping experiment.

`later_observed_return_below_half` checks every subsequent available 15-ps sample
after the initial confirmed crossing. The original fine return diagnostic checks
all remaining 0.75 ps measurements. These can reveal retreat below the target;
dense excursions between 15-ps observations are not assessed. Subsequent growth
or annealing above 50% is **not a failure for this user-defined criterion**.

## Summaries and censoring

Each source is weighted equally. `confirmed_half_count` and `no_half_count`
retain the full group denominator, including paths not reaching the criterion.
Half-time and stop-time conditional medians/maxima use confirmed crossings only,
so they are not unconditional survival quantiles or guaranteed fixed cutoffs.
`mean_stop_all_sources_ps` includes all non-crossers at 600 ps. Fixed-cutoff
counts require the full confirmation and 6-ps tail to finish by that cutoff,
not merely crossing 50% at the cutoff. All historical event stops precede 600 ps
in this capture; missing crossings remain censored there.

Saved measurement time = 600−proposed_stop. Total saved fraction divides the
sum by 600 times the full source count. Integration saving including equilibration
divides by 615 times the full count. These are counterfactual physical-duration
savings, not measured execution speedups. Conversions/IO, monitoring cost,
scheduler barriers and heterogeneous hardware can change wall-clock savings.
Existing complete files are not trimmed. New stopping would require a versioned
simulation/conversion protocol supporting variable endpoints, explicit termination
metadata and target censoring; rows without sufficient future horizon must remain
masked rather than assigned negative labels. Training/evaluation sources keep
their fixed ancestor roles and their exact physical observation spacing.

## Time when 90% of peer runs have reached the target

User clarification, 2026-10-01: 90% refers to **other runs reaching the 50%
criterion**, not to 90% crystal atoms in a run. For a cohort of N sources,
require ceil(0.9*N) confirmed half-crossings with their retained prediction
tails. Use the corresponding ordered event-stop time. The denominator includes
**all sources**, not just eventual crossers. If fewer than the quorum reach
the target through 600 ps, the peer90 time is missing and the effective cap
remains 600 ps; do not compute a misleading conditional event percentile.

Report same-temperature cohorts separately, plus the full mixed-temperature
cohort as an explicitly distinct diagnostic. Temperature is an audit grouping,
not a model input. Rounded counts can make the reached fraction exceed exactly
90%. Dense peer estimates apply only to the captured 21-run 520 K subset.

For a hypothetical policy combining individual event stops with the peer cap,
each source stops at min(individual_stop,peer_cap). `right_censored_half_event`
means the required half event was not ready by that cap; it does **not** mean
the source never crystallizes. The full historical observations separately
identify cases crossing later by 600 ps, and cases with no crossing through
600 ps. Both are useful delayed/survival observations. The table retains all
sources and source identities. Peer-cap savings are counterfactual and include
non-crossers' retained duration. No extra run is simulated and no censored
event is assigned a negative eventual-crystallization label.

These retrospective peer calendars are descriptive results, not independent
validation of future rerun caps. A prospective stopping protocol must declare
its peer set/maximum in advance, its selection data and censoring policy, and
how it handles a peer quorum that is not reached; it cannot inherit old source
outcomes or treat a censored liquid endpoint as proof of permanent stability.


Table export: 2026-10-01T21:38:07.971022+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/al_half_transform.json` relative to the analysis root.
