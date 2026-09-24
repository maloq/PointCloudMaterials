# Results database, provenance and refresh

[Handbook](README.md) · [Interactive catalogue](../../output/encoder_research/catalogue/index.html)

The database has two layers. **Curated headlines** carry explicit scientific
context and are convenient for comparison within a declared group. **Imported
evidence** preserves the original heterogeneous tables and reports, including
historical/failed/smoke runs. It is comprehensive over the registered collections,
not a claim that every old file anywhere on every server has been recovered.

| Artifact | Role |
| --- | --- |
| [catalogue.json](catalogue.json) | Versioned families,74 dated study records, registered evidence roots and explicit scan exclusions |
| [highlights.json](highlights.json) | Versioned151 selected numeric entries with exact evidence quotations and interpretation |
| [results.sqlite](../../output/encoder_research/catalogue/technical/results.sqlite) | Generated queryable evidence database |
| [headlines.csv](../../output/encoder_research/catalogue/tables/headlines.csv) | Flat curated scores for spreadsheets; original reported precision |
| [artifacts.csv](../../output/encoder_research/catalogue/tables/artifacts.csv) | Full artifact index, source/path, hash, import status, row count, original definition links |
| [studies.csv](../../output/encoder_research/catalogue/tables/studies.csv) | Dated protocol/finding records, including archived ones |
| [coverage.json](../../output/encoder_research/catalogue/technical/coverage.json) | Actual capture time/counts, manifest identity and explicit coverage limitations |
| [METRICS.md](../../output/encoder_research/catalogue/tables/METRICS.md) | Frozen definitions for the catalogue exports themselves |

No database server is needed. SQLite, Python's standard `sqlite3`, pandas or a
SQLite browser can open the file. The HTML contains its own searchable metadata
and requires no network. Its storage links are generated against this machine's
configured roots; the canonical manifest uses portable storage tokens.

## Schema

`families` describes18 research families. `studies` holds dated scientific records
and their source paths; status `consult_record` means no completion claim is
inferred from the directory. `headlines` stores:

```text
id, comparison_group, family, model, metric, value, unit, direction,
population, split, horizon_ps, seeds, evidence, source, path,
evidence_text, caveat, sha256
```

The comparison group is an editorial grouping, **not** sufficient proof that every
row can be ranked. For example, native and fresh probes may coexist as explicitly
different tests. Horizon/readout can also be encoded in the original metric name.
A portable future canonical evaluation ID with row/normalization hashes is still
recommended; this first index does not invent missing historical metadata.

`artifacts` stores each original file's family, source/path, kind, SHA-256 of
imported bytes, size, logical row count, import status, ordered CSV header,
nearest exported definitions/implementation contract and local navigation link.
`records` exposes `(artifact_id, ordinal, kind, payload_json)` as a view.
Byte-identical record sets are physically stored once in `record_data` /
`record_sets`, while each original artifact keeps its own identity and metadata:

- CSV: ordered original strings; blanks remain blanks and duplicate headers retain
  their column index. Ordinal is a logical CSV data-record number after the header.
- Markdown: literal table header and row lines; ordinal is the one-based source
  line number. No equation/unit inference or rounding is applied.
- Selected result JSON: one complete nested object per file. Nonstandard numeric
  tokens such as NaN become literal strings, not valid scores.
- HTML: linked gallery only. It can embed millions of coordinates, so it is not
  loaded into the database.

`csv_cells` exposes CSV records as `artifact_id, ordinal, column_index,
column_name, value`. `duplicate_artifacts` identifies byte-identical files.
Summaries can repeat a scientific result with different bytes, so removing exact
hash duplicates alone does **not** produce a count of independent fits.

## Example queries

Run from the repository with `pointnet-torch214`:

```python
import sqlite3
con = sqlite3.connect('output/encoder_research/catalogue/technical/results.sqlite')

# A specific matched population and score, with caveats attached.
for row in con.execute('''
    SELECT model, value, population, seeds, caveat
    FROM headlines
    WHERE comparison_group = 'structural-state-repaired'
      AND metric = 'nonlinear_average_precision'
    ORDER BY value DESC
'''):
    print(row)

# Find every original onset table in the latest family.
for row in con.execute('''
    SELECT id, path, row_count, definitions
    FROM artifacts
    WHERE family = 'structural_state' AND kind = 'csv'
      AND path LIKE '%onset%'
'''):
    print(row)

# Recover the exact original columns and row (all CSV values remain strings).
artifact = 'repo:output/structural_state/future-metric-20260923/tables/onset.csv'
for row in con.execute('''
    SELECT ordinal, column_name, value FROM csv_cells
    WHERE artifact_id = ? AND ordinal = 1 ORDER BY column_index
''', (artifact,)):
    print(row)

# Explicitly identify evidence imported from remote summaries only.
print(con.execute('''
    SELECT comparison_group, model, metric, value
    FROM headlines WHERE evidence = 'remote_summary_only'
''').fetchall())
con.close()
```

Do not cast a blank/undefined field to a numeric score: SQLite can turn invalid
text into0 during a naive cast. Use the original field's documented type and
validate before numeric analysis. This catalogue does not recalculate confidence
intervals; use the producer's saved per-source/prediction arrays for that.

## Updating and transferring

1. Add a study/family or output collection to `catalogue.json` with a portable
   source root. Keep source ancestry/dataset changes in `configs/datasets.json`.
2. Add meaningful selected results to `highlights.json`: include exact source text,
   units, split/population, horizon, seed count and a caveat. Do not replace a test
   score with validation or claim remote verification without its exports.
3. Update the relevant human guide when interpretation changes. Preserve historical
   reports, even after correcting a later collector or analysis.
4. Refresh with `python scripts/experiment_registry.py encoders`. The collector
   reads registered evidence; it never trains, benchmarks or modifies those inputs.
   Its imported-byte hashes identify the actual snapshot. Missing registered roots,
   malformed CSV/JSON or a changed curated quotation fail with file context.
5. Inspect `technical/coverage.json`, the family counts and links. Update this
   handbook's dated summary if a new scientific conclusion is established.

The database is written through a temporary SQLite file and replaced only after
integrity/foreign-key checks. An unsuccessful import does not replace the previous
finished database. The snapshot is not a transaction across every source report;
completed immutable exports are the preferred inputs. Do not refresh against files
actively being rewritten if an exact reproducible snapshot is required.

Raw input trajectories, inference caches, code snapshots, checkpoints, training
execution histories and large coordinate tables are not bulk-imported. Explicit
exclusions and the20MB import limit are in the manifest. Oversized result files
remain indexed with a reason; no scientific result is silently substituted.
Original JSON/CSV paths are retained. Metric definitions missing from old exports
are reported as missing, not reconstructed from today's implementation.

To use the existing result database elsewhere, copy the handbook plus the generated
catalogue output (HTML, SQLite, tables and coverage). This supports searches and
SQL without the original datasets. Opening linked figures/reports additionally
requires their source files and suitable storage mappings. Do not copy absolute
symlinks as if they contained the target data. To rebuild the full catalogue,
configure all registered archive/analysis roots first; a source-unavailable failure
is intentional. Raw H200 predictions are not manufactured from its summaries.
