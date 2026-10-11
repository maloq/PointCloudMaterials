# Additional negative continuations, v1

This is a negative-only supplement to the frozen `nucleus_controls` campaign.
No encoder/predictor is trained. Physics, inputs, endpoints, PTM and lineage
calculations reuse that campaign's implementations and definitions. In particular,
these are 70,304-atom cells with actual simultaneous saved Q,V; original native
0.75 ps observations, 12 ps forward windows, and 3/6/12 ps endpoints. Original
quantized Q,V provenance is retained. No coordinate perturbation, initial
velocity redraw or relaxation is applied. BAOAB uses float32 states/momenta/noise/
forces; the backend scalar energy reduction is float64. Native NPT retains
float64 integration with float32 forces and reset bath state. Temperature is
only physical thermostat/audit metadata; encoder and predictor inputs are absent.

## Frozen additional sampling

Exclude every ancestor in the previous matched campaign. Screen the original
150-source release for frames with current largest tracked component <=7 atoms,
PTM crystal fraction <=1%, and 12 ps of recorded reference coverage. These
present-structure/availability filters never use a future outcome. Rank eligible
sources by SHA256 of the declared seed and source ID; select the first 64.
Draw one eligible frame and one anchor atom uniformly for each selected source.
Store all eligible frame IDs and all source exclusions. Sources, not frames,
have equal selection probability. Preserve original source roles. The random
liquid cohort is not matched to embryo cases and not representative of every Al
regime. A random-liquid saved or simulated future may crystallize; it is not
assumed negative by construction. Its `initial_cluster_atoms=0` means no chosen
seed, not absence of every small component; `initial_largest_cluster_atoms`
reports the actual full-cell maximum.

Each new liquid state receives one original-settings source-temperature NPT run
and two independent 450 K BAOAB shots. The 12 existing failed-embryo states each
receive four additional independent BAOAB shots (replicate labels 4–7). Reuse
verified starting snapshots and recorded-future metrics, with hashes. No NPT
rerun for those unchanged embryo states. New seeds use a separate protocol
namespace, are unique, and are disjoint from previous seeds. Total: 240 new
continuations from 76 ancestors, of which 64 were absent from the previous
matched cohort and 12 were already used. No new independent MD preparation
ancestry is created. Do not pool these repeated embryos as new independent data.

## Endpoints and uncertainty

`nucleus_controls.analysis.measure` defines every branch metric without changes:
isolated births strictly after restart, confirmed >=64 atoms across three
adjacent observations; local births within 8 Å of a fixed advected anchor;
tracked seed establishment/disappearance; PTM crystal fractions and cluster sizes.
Historical graph prefixes preserve pre-existing ancestry/arrival exclusions.
See the separately hashed `nucleus_controls.md` for complete definitions.

`branches` contains completed new continuations only. `saved-reference` is a
separate observational arm, not a new simulation. `cohorts` is a long table:
mean each metric over a state's planned shots, then average these means across
completed states. A state enters this mean only after all of its arm's planned
shots finish. Undefined seed metrics stay blank for liquid anchors. Repeat
counts and planned/completed state counts are explicit. Resample whole states
(2,000 draws, fixed seed) for percentile 95% intervals. Within each cohort there
is exactly one state per ancestor. Intervals are undefined if fewer than two
values or all values identical; a zero-width bootstrap at zero events does not
establish zero risk. No matched contrasts or pooled old/new significance tests
are produced.

`first-shot-birth-rates` uses one predeclared shot per ancestor: minimum replicate
ID for each state and arm (0 for new liquid, 4 for repeated embryos). It reports
birth counts at 3/6/12 ps, the observed fraction, and two-sided 95% Wilson score
intervals with z=1.959963984540054. With x successes out of n states, center is
(x/n + z²/(2n))/(1+z²/n), half-width is
z sqrt((x/n)(1-x/n)/n + z²/(4n²))/(1+z²/n). These descriptive binomial intervals
refer to the declared selected population and retain a positive upper bound
for zero events. They are not intervals treating every noise shot as independent.
An embryo first shot is independent across the 12 original ancestors, not an
extra independent state relative to the earlier release.

Partial coverage is provisional: shorter jobs can finish first. Queue order is
fixed before dynamics and spread over states before later noise repeats. The
shared three-hour deadline includes scheduler wait, has no automatic extension,
and checkpoints unfinished owned workers; incomplete branches never become
negative outcomes. Additional two-hour GPU helper allocations share the same
absolute deadline. Scientific implementation hashes accompany every metric CSV.
Other native cadences, shooting/dense descendants, million-atom data, relaxed
views, and previously established nuclei that later melted are outside this
supplement. The existing matched positive campaign remains a separate reference.
MD branch publication retains the maintained converter's verified float16 archive
and restart/integration precision records.
