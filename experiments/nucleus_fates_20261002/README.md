# Failed crystal embryos and the fate of established births

Question: how many smaller crystalline episodes disappear before our 64-atom,
three-frame establishment criterion, and do the established births later grow
or dissolve?

Reuse all 150 original Al64 sources with the frozen full-cell PTM/lineage graph.
Search all graph components unconnected to any establishment, verify terminal
atom identities against later PTM labels, and keep isolated, interface-associated,
multi-origin, residual-crystal and censored cases distinct. The primary smaller
candidate requires eight atoms in two adjacent observations; counts also retain
other sizes/durations. These are operational embryos, not validated critical nuclei.

Established-event growth is a sustained doubling after confirmation, with a
minimum size of 128 atoms. Growth and terminal status are separate, allowing
growth followed by dissolution or merging. Retain unchanged sample identities
and append labels to positive examples only; continuously liquid controls have
no nucleus fate. New candidates still require pre-appearance input eligibility
screening before a future classifier can use them.

[Exact definitions](../../docs/metrics/nucleus_fates.md) ·
[Recipe](../../configs/analysis/nucleus_fates_20261002.json) ·
[Execution](../../docs/nucleus_fates.md).

```bash
python -m src.research.crystallization_origin.fates submit \
  --config configs/analysis/nucleus_fates_20261002.json
```

Output: `${storage:training_storage}/nucleus_fates/al64-20261002/analyses/fates-v1/`.
The detached collector writes event/row tables and the complete count report.
No predictor fitting or new simulation is part of this audit.

## Completed findings

All 150 original sources completed on 2 October 2026. The conservative primary
screen found **28 failed-embryo candidates across 24 sources**, with peak sizes
8–40 atoms. Nineteen are on train sources, three on former selection, two on
former calibration and four on original test. These counts are events, not input
histories or independent sources. Allowing single-observation episodes raises
the isolated single-origin, strongly linked, atom-verified pool to 184 episodes
across 99 sources. Seven of the 28 primary candidates reached at least 16 atoms;
one reached at least 32. None reached 64.

Of the **95 retained original birth events**, 87 show the declared sustained
post-establishment growth; eight merge before it can be confirmed. None has
confirmed dissolution. Among the 87, 50 later merge and 37 remain tracked to the
trajectory end: growth is observed, but indefinite survival is not established.

The broader primary-establishment catalogue contains one confirmed dissolution
(source 979, event 3), but it is interface-associated and was never part of the
isolated pre-appearance birth cohort. It is not silently added to that cohort.

The inventory includes 68,518 never-established temporal components. Of 8,160
with peak size at least eight, 581 pass atom-level disappearance verification
before the additional isolation, origin and persistence filters; 44 lack follow-up
and 7,535 retain some crystalline terminal-core atoms in the immediate confirmation
window. The 60,358 smaller components are inventoried without atom-level fate
verification. These exclusions make the strict failed-candidate count conservative.

All 1,475 existing sample IDs have an additional-label sidecar: 280 positive rows
are linked to observed growth and 15 to unresolved merges. The 1,180 liquid
controls retain their original label and have no nucleus fate. Original and
relaxed inputs share the same sidecar. New transient candidates require separate
pre-appearance history screening before predictive experiments.

Full report and machine-readable tables:
`${storage:training_storage}/nucleus_fates/al64-20261002/analyses/fates-v1/`.
