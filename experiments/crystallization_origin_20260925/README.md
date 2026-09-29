# Origin of local crystallization: label-availability audit

[Completed results, 26 September](RESULTS.md): all 150 fixed Al sources and
26 external Al/Mg/Ti/Ta records. Existing-crystal arrival explains 98.5% of
positive Al64 windows; regional-birth training windows are absent on that grid.

[Proposed emergence harvest](HARVEST_PROPOSAL.md): causal regional eligibility,
event-enriched training samples, representative evaluation and preserved source
roles. The [first training-source harvest analysis](../../docs/nucleus_harvest.md)
is implemented; it exports diagnostic references before any training release.

Question: how much of our local-onset prediction population represents new
isolated crystal establishment, growth from an existing crystal, or an
externally formed crystal arriving after the forecast origin?

Use the original observed MD for all 150 fixed Al64 sources, every 0.75 ps frame,
and full periodic cells. Do not fit a model, alter the benchmark, or generate
new simulations. The predefined initial training sources are 860, 884, 914, 944
and 974, chosen as the lowest train source IDs spanning available temperature
strata. Temperature is audit/selection metadata only, never a predictor input.
No label thresholds are optimized on held-out outcomes.

Reference operational establishment: 64 PTM-crystalline atoms connected within
3.6 Å, persisting for three frames. Compare 32/128 atoms and five-frame
persistence. Track atom-identity overlap, a one-frame gap, splits and merges.
Keep unresolved, interface-associated, in-progress and initial left-censored
cases explicit. These labels do not estimate the physical critical nucleus.

Report distinct full-cell establishments and coverage by 64/16 tracked centers;
origin-dependent first-onset labels and a separate local-region birth target
at 3/6 ps; counts by unchanged source role; and PTM agreement with historical
center outcomes. Preserve exact sample IDs for subsequent matched probes.

The encoder and predictor input contracts are **not applicable: no model is
trained or evaluated**. Label construction uses full-cell geometry, atom IDs,
periodic boxes and observation order. Future confirmation is label-only;
relaxation, species channels, temperature and simulation age do not enter it.

Reproduction commands and the live report are in the
[audit guide](../../docs/encoder_research/crystallization_origins.md).
[Configuration](../../configs/analysis/crystallization_origin_20260925.json) and
[metric definitions](../../docs/metrics/crystallization_origin.md) are versioned.
The report remains partial until all 150 receipts are complete; no unprocessed
source contributes a zero count.

Training-source validation corrected one attribution error before held-out
analysis: a weak peripheral ancestor must not erase an independently strong
ancestry path. Revision 2 retains possible roots for ambiguity checks while
tracking strong support separately. The earlier pilot is preserved in the run.

An early inspection of source 860 through 191.25 ps found one growing lineage
under all four establishment criteria and no tracked center within 8 Å of its
birth. This motivates measuring birth coverage explicitly; it is not a
cohort-wide estimate. See the run's `EARLY_CHECK.md` and trajectory plot.
