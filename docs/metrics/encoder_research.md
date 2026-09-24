# Encoder research evidence catalogue

This collector indexes existing evidence; it computes no new encoder performance
score, confidence interval, ranking or aggregate across experiments.

`headlines.csv` reproduces selected reported numbers with their comparison group,
original unit, preferred direction, population, split, physical horizon, seed
information, exact source quotation and SHA-256. Values are rounded to the source
report's precision, not claimed to recover precision absent from that report.
`remote_summary_only` means the local report quotes a remote/user summary; it does
not mean raw remote predictions were independently verified. Seed entries can be
text when different tiers or encoder/probe seeds must be distinguished.

`artifacts.csv` inventories reports, CSV tables, result JSON and HTML galleries
in explicitly registered scientific collections. `row_count` is the number of
imported records, not experiments, fits, examples or independent sources.
`studies.csv` indexes research records; `consult_record` deliberately avoids
inferring completion from a folder name or stale execution state.

SQLite `records` preserves CSV fields as ordered strings including empty cells
and literal nonfinite text; `columns_json` holds the original ordered header.
It does not coerce undefined values to zero. CSV ordinal is a logical data-record
number after the header, starting at one, not necessarily a physical line number.
Markdown records retain raw header and row lines, with original one-based line
numbers. JSON records retain their complete nested content in one record; legacy
nonfinite tokens become literal strings and remain undefined numeric evidence.
Use original bytes/hashes when inspecting serialization details.

Exact file copies remain distinct artifacts with the same hash. Their imported
record sets are stored once; the `records` view preserves every artifact alias.
`stored_records` in the coverage report counts physical records after byte-level
deduplication, while `records` counts the logical occurrences across artifacts. Re-exported
summaries and Markdown tables can repeat the same scientific result even with
different file hashes. Neither is extra replication. `duplicate_artifacts`
finds byte duplicates only. There is no global cross-encoder ranking.

The manifest declares skipped code/cache/checkpoint directories, index-only
per-observation coordinate tables, and an explicit import byte limit. Oversized
artifacts are indexed with a reason. HTML galleries are linked without reading
large embedded point clouds. Only imported bytes are hashed; a missing hash on
an index-only entry is explicit. Original reports, arrays and checkpoints remain
unchanged. Missing registered source roots fail the build, rather than producing
an apparently complete catalogue. The coverage record identifies capture time,
manifest/highlight hashes and missing historical metric-definition links.

`definitions` and `implementation_contract` point to the nearest exported metric
documentation where present. Missing historical contracts are left blank, not
filled with today's definitions. The catalogue's own implementation hashes describe
the collector only, never the old metric producers. The snapshot is not a live
scheduler monitor and does not imply reanalysis of any predictions.
