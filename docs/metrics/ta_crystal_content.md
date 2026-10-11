# Ta crystal content and ancestry exclusions

This read-only analysis consumes the completed `ta_births` plan, graph counts,
event labels and transient episodes, verifying each consumed artifact against
the frozen source receipt. It does not recompute PTM, modify the existing birth
labels, rerun MD or fit a model. Twenty-four shooting branches are primary;
six historical trajectories are exported separately as reference curves.

`crystalline_atoms` comes from `external_audit.graph_for_source`: every full-cell
atom with the recorded PTM label FCC/HCP/BCC, including atoms outside tracked
components. The frozen RMSD cutoff is0.1. `crystalline_percent` is100 times that
count divided by the source atom count. Largest cluster is the maximum saved
component size at that frame (the graph tracks components of at least four).
These operational fractions are not a thermodynamic phase measurement, and
unclassified atoms are not proven liquid. Quantization and PTM sensitivity are
not independently reassessed by this calculation.

Every source has49 observations at exact0.5ps spacing from0 to24ps. Endpoint
change is final minus initial fraction, in percentage points. Net atom change
is final minus initial crystalline count, not the number of atoms that ever
changed classification. Parent summaries average the four shooting endpoints;
min/max is their observed range, not a confidence interval. Identical starting
counts are required across the four replicas. Historical endpoints are not
included in these means. Source labels such as2.7ns identify archived parents;
they are not used as predictor inputs or as observation times in this plot.

Establishment category counts are the frozen per-branch labels. Branch siblings
repeat related structures, so the sum is not a count of independent nuclei.
The prior audit calls an establishment parent-connected if any directed path,
including weak-overlap edges, reaches frame0. This analysis additionally
propagates earliest frames through strong edges only. An event is
`strong_path_to_start` if such a path reaches0 at its birth node; otherwise
its parent connection requires a weak edge. Strong edges keep the frozen
minimum shared atoms, overlap fraction and allowed missing-frame rules. This
does not change the spatial classifier, graph or event label, and is not proof
that a weak-link-dependent event is a new physical nucleus.

The dissolution table applies sequential filters to non-establishment-connected
episodes: all; peak≥8; verified dissolved; isolated origin; one root;
strong-connected lineage; sustained≥8 for the frozen1ps requirement.
Weak contact to established graph components already excludes an episode before
this sequence, according to the frozen producer. Missing follow-up and residual
crystal remain excluded from verified dissolution. Counts are source-specific;
the report sums only the24 shooting sources. The historical fourth brief failed
candidate is explicitly separated.

No population rate or bootstrap inference is made: six correlated starting
configurations and four velocity branches each do not constitute independent
melt preparations. Plots use separate vertical scales for readability, explicitly
declared in the title. The complete original numerical definitions are retained
in the completed `ta_births` result bundle.
