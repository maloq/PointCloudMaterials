# Research results: using the implemented system

The starting page is [Materials research results](../output/registry/index.html).
It joins named analysis bundles and explicit result receipts with the existing
[historical evidence catalogue](../output/encoder_research/catalogue/index.html)
and [filesystem inventory](../output/registry/inventory.html). All three use the
existing repository tooling; there is no database service or new online tracker.

This implements the first adoption of the
[audited design](research_results_system_proposal.md). Existing historical
artifacts and active jobs retain their paths. Producers beyond the standard
analysis pipeline and supervised-onset collector still retain their original
output contracts until explicitly adopted.

A fresh numerical example is now available:
[native MACE EPI-variance, epoch 12](../output/structural_static/mace-epi-epoch12-al-20260926/analyses/standard-v1/index.html).
The verified native checkpoint ran on 684,723 neighborhoods across all six Al
snapshots, producing 77 visual artifacts (76 shown by default, including 14
interactive HTML views; the retained paper SVG is opt-in). Its tracked execution succeeded;
equivariance was disabled and identity-tracked dynamics remain unavailable.
The original checkpoint and inference producer are linked in the result receipt.

Its [local-order bundle](../output/structural_static/mace-epi-epoch12-al-20260926/analyses/order-v1/index.html)
adds three static/interactive figures: sparse representatives, detected local
order, and a detailed comparison using PTM on all 684,723 centers plus continuous
order diagnostics on 2,500 stratified samples. Adding a named bundle preserves
the earlier analysis and authored run metadata in the shared receipt.

## Open the adopted results

- [GATr step 3072: full static Al analysis](../output/structural_static/gatr-step3072-review-20260926/analyses/static-al-v1/index.html).
  All 104 visual artifacts are organized by snapshot, latent view,
  representatives and connected regimes; the original flat gallery had 39 links.
- [GeoFrame V2 VICReg epoch 034: restored analysis](../output/factor_vae_archive/geoframe-vicreg034-review-20260926/analyses/static-al-v1/index.html).
  Navigation over the retained archived analysis, including its historical
  projector export and retired-cache limitations.
- [Distance/future two-seed comparison](../output/structural_state/future-metric-review-20260926/analyses/distance-future-v1/index.html).
  Eight encoder components link their original completion/checkpoint receipts.
  The historical physical-reconstruction treatments are evidence, not future
  queue recommendations.

These publications did not fit models, rerun inference or recalculate metrics.
Authored source READMEs, reports, figures, arrays and exported definitions remain
at their original paths. Images use relative aliases; HTML entry pages under
`plots/` open the originals at their source so their relative dependencies still resolve. New
navigation is self-contained; legacy interactive pages retain their original
external JavaScript dependencies.

The publication audit verified 359 hashed artifacts across these three bundles;
one large checkpoint was indexed without rehashing. All local figure aliases and
HTML asset references resolved. The 25 external HTML asset references remain
external dependencies. Incremental indexing preserved the existing 13,086 raw
artifact entries and 2,055,178 stored evidence records.

The full historical refresh currently fails because the registered
`output/geoframe_evolution` and `output/neighborhood_jepa` collections are
unavailable. Git retains the GeoFrame review text, but that alone does not recover
the scientific collections. The existing historical catalogue remains available;
its capture time is separate from the new receipt index. The main page shows the
failed refresh and links its receipt instead of suggesting a complete refresh.

## Result records and layout

```text
output/<question>/<run>/
  run.json                              identity, study, components, three statuses
  README.md                             authored interpretation, if present
  index.html                            existing report or generated entry page
  analyses/<name-or-revision>/
    analysis.json                       immutable numerical identity, context, stages
    artifacts.json                      semantic roles, locations, hashes, retention
    index.html                          grouped gallery / evidence navigation
    plots/snapshots/<frame>/...          original hierarchy, no flat filename catalogue
    tables/scores.csv                   newly calculated standard-analysis scores
    tables/METRICS.md                    frozen calculation description
    technical/metric-contract.json      frozen verified calculation dependencies
    technical/rendering.json            rendering producer identity
    data/                               new standard producer's native artifact tree
```

The standard pipeline writes new native artifacts into
`analyses/standard-v1/data/`. The publisher exposes scientific figures through
structured `plots/` aliases and assigns roles to tables, models, scientific data,
provenance, execution files and caches. This preserves the native scientific
stages without rewriting every specialized producer. Existing root-level and
`technical/` analyses continue to resolve in their original layout.
New runs do not create empty run-level `plots/` and `tables/` alongside these bundles.
Each gallery lists its interactive HTML views prominently and exposes them in
the same plot hierarchy as the static figures.

`technical/` is never a deletion category. Cache removal remains governed by
existing verified retention workflows and leases. Model weights, predictions,
source snapshots, metric definitions and scientific evidence remain protected.
Machine locations use the existing storage tokens and `machine.local.yaml`.

A run receipt distinguishes:

- **Execution:** observed producer state or explicitly historical/unknown.
- **Evidence:** available stages and requested-but-missing results.
- **Interpretation:** an authored conclusion, preserved across refreshes.

A report over historical evidence is not a new encoder fit. Scientific evaluation
identity uses numerical evidence, checkpoint and protocol/context metadata;
publication locations do not create additional independent evaluations.
Component receipts distinguish an encoder from its frozen probes. Counts in the
filesystem inventory are directory groups, not independent observations.

Missing historical metadata remains visible. Publishing an old result does not
relabel its input conditions, training objective, selector or population. A
figure's source snapshot time remains navigation metadata, not a model input.
The catalogue does not rank models, promote by AP, or fit ensemble weights.

## Commands

Use `pointnet-torch214` from the repository root.

```bash
# Publish the three retained examples from their explicit recipe; no recomputation.
python scripts/experiment_registry.py publish \
  --plan configs/analysis/result_publication_20260926.json

# Refresh one gallery's navigation while preserving its annotations and evidence.
python scripts/experiment_registry.py publish \
  --record output/structural_static/mace-epi-epoch12-al-20260926/run.json

# Refresh explicit records and the shared SQLite relations only.
python scripts/experiment_registry.py results

# Re-register a durable receipt if rebuilding the local registration queue.
python scripts/experiment_registry.py results --record output/QUESTION/RUN/run.json

# Check saved evidence hashes, aliases and local HTML asset dependencies.
python scripts/experiment_registry.py verify-results \
  --record output/structural_static/gatr-step3072-review-20260926/run.json

# Repair grouping/classification using an existing inventory; retain observation date.
python scripts/experiment_registry.py build --from-snapshot

# Existing full filesystem discovery, including configured external roots.
python scripts/experiment_registry.py build

# Verify actual source/doc bytes against every declared metric dependency.
python scripts/experiment_registry.py metrics-docs
```

Publication is repeatable when its numerical evidence has not changed. If the
numbers or scientific context change, choose a new analysis revision/name.
Existing authored README and interpretation text are preserved. The recipe is
also the reproducible registration source for these historical publications.
Cluster-proportion paper SVGs are hidden by default; add `--include-paper-svg`
to expose a retained SVG. New numerical runs generate it only when
`real_md.time_series.paper_enabled: true` is explicitly configured. Existing
source SVGs remain preserved with their original evidence hashes.

`verify-results` streams file hashes and checks original HTML asset paths. It
reports large artifacts that were indexed without rehashing and references to
external assets separately. Missing retained evidence or changed recorded hashes
fail with the artifact path. A deliberately retired cache is reported separately;
it is not treated as a lost scientific result. This audit does not recompute
predictions or assert scheduler liveness.

## Producers and metric contracts

`src.analysis.pipeline` explicitly marks numerical completion when publishing.
Only that numerical path exports a fresh `scores.csv`. Publication-only and
saved-figure publication calls use retained calculation definitions and never
attach the current checkout's definitions to historical values.
Full numerical reruns require a new output directory; completed evidence is not
overwritten. Use publication for a presentation refresh and the saved-representatives
workflow to redraw the frozen native-MACE selections. The former `figure_only`
execution mode has been removed: it also recalculated figure diagnostics. New
diagnostics require a new output, with their own recorded inputs and definitions.
The legacy numerical discovery receipt remains available to the
maintained topology collector; flat plot aliases are not recreated.

`check_metric_docs()` now checks files and SHA-256 values, rejects unknown
families and missing required descriptions, and validates only the requested
family during export. Scientific contract changes require a new analysis
revision. The original `METRICS.md` and root contract remain preserved.
Multiple families in one result root have separate definitions and contracts:

```text
tables/metric-definitions/<family>.md
technical/metric-contracts/<family>.json
technical/table-contracts/<table>.json
tables/METRIC_INDEX.md
```

`write_metric_table()` records the exact CSV hash and its family-specific
contract/description. The evidence catalogue resolves that explicit binding
before using historical nearest-document lookup. New summary tables use
`scores.csv`; training histories remain separate. Git permits compact named
analysis receipts and definitions while keeping images, models, logs and arrays
local. Historical flat `tables/metrics.csv` exclusions remain unchanged.

The generated encoder catalogue is a special case: it is a refreshable index,
not a scientific recalculation. When its own collector contract changes, the
previous exported tables and definitions are hash-verified into
`technical/catalogue-revisions/` before replacement. Underlying scientific
exports and their original definitions are never rewritten by this refresh.

The maintained supervised-onset collector now emits named evaluation bundles
and campaign/component records after its existing numerical export. It uses
actual saved metrics, prediction-context and cohort receipts, distinguishes
missing readouts, and retains unavailable movement diagnostics as unavailable.
It does not change fitting, inputs, selectors or online W&B identities. Future
training continues to require online W&B in `teshbek/PointCloudMaterials`.
Rendering, indexing, verification and operational checks create no online runs.

## One shared index

Producer `run.json` receipts are durable. Tiny location registrations under
`output/registry/registrations/` are generated pointers to those receipts.
Both full encoder-catalogue builds and incremental result refreshes ingest them
into the existing `results.sqlite`:

- `result_runs`: scientific/operational kind, study and separate status dimensions.
- `result_components`: named encoder/probe components and original receipts;
  a shared component identity may be referenced by multiple result records.
- `result_evaluations`: evaluation identities and protocol/context evidence.
- `result_artifacts`: semantic roles, original locations, hashes and retention.

Original `families`, `studies`, `headlines`, raw `records` and artifact tables
remain intact. Missing registered storage is displayed as unavailable; malformed
or conflicting receipts fail explicitly. An index refresh failure is recorded
locally without relabeling a completed scientific fit as failed.

The general inventory now groups modern `output/<question>/<run>` paths
separately. Research collection classifications come from the existing encoder
manifest, with explicit operational/dataset/simulation overrides in
`configs/experiment_registry.json`. Unknown paths remain unclassified.
`build --from-snapshot` intentionally keeps the old filesystem observation time.
It does not refresh scheduler status or claim that newly created directories
have been scanned.

The next adoption should use the existing artifact/receipt interfaces and
scientific workflow, rather than add another launcher or result database.

## Retired numerical producers

An authorized architecture retirement marks its metric family `status: retired`
in `docs/metrics/contracts.json`, with a checksummed JSON record under
`docs/metrics/retired/` containing the exact previous contract. Original metric
descriptions and file-hash maps remain historical definitions. The repository
audit validates the record and description without requiring deleted live source.
`metric_docs.snapshot_metric_docs` rejects new numerical exports for those
families. Use the recorded frozen source for numerical reproduction; use the
publication-only workflow to expose existing tables and their frozen definitions.
Active diagnostic families retain their actual remaining dependencies and require
a new numerical revision when their source contract changes. See
[the September30 retirement](architecture_retirement.md).
