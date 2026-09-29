# Separating establishment from arrival

[Method review](nuclei_and_prestructured_liquid_20260926.md) distinguishes
operational establishment, prestructured liquid and physical critical nuclei,
with literature and the existing training-source threshold sensitivity.

[Dataset proposal](../../experiments/crystallization_origin_20260925/HARVEST_PROPOSAL.md):
harvest existing births with precursor-permitting regional risk rules and
separate enriched training from representative evaluation.

[Completed analysis, 26 September](../../experiments/crystallization_origin_20260925/RESULTS.md)
covers all 150 fixed Al sources and 26 external records. The fixed Al64 task is
98.5% existing-crystal arrival among positive windows, while its regional-birth
training count is zero. The analysis separates full-cell candidates, sampled
coverage, forecast windows and shared branch ancestry.

The [CPU campaign](../crystallization_origin_cpu.md) resumes the fixed audit and
adds the million-atom Al history plus available Mg/Ti/Ta dynamics, with a target
of 26 September 2026 at 10:00 Paris time. Its counts remain separate by material,
potential, sampling density and ancestry group.

The existing local outcome records when a tracked atom first remains crystalline
for three observations. It combines formation of a new crystal with capture by
an existing one. Our encoder results therefore demonstrate local-onset prediction,
not isolated-nucleation prediction.

The first audit uses the original full periodic cells from the fixed Al64 cohort.
It preserves the source splits and every prediction sample, adds cluster ancestry,
and counts label availability at 3 and 6 ps. It runs on CPUs, does not train, and
does not send scientific-audit or debug runs to W&B.

- [Scientific protocol](../../experiments/crystallization_origin_20260925/README.md)
- [Exact definitions and limitations](../metrics/crystallization_origin.md)
- [Configuration](../../configs/analysis/crystallization_origin_20260925.json)
- [Live report](/work/PERSO/vmorozov/analysis/crystallization_origin/al64-20260925/RESULTS.md)
- [Original outcome inventory](/work/PERSO/vmorozov/analysis/crystallization_origin/al64-20260925/INVENTORY.md)
- [Counts by split, label and threshold](/work/PERSO/vmorozov/analysis/crystallization_origin/al64-20260925/tables/label-counts.csv)

Run in `pointnet-torch214`:

```bash
OVITO_THREAD_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.research.crystallization_origin.extract \
  --config configs/analysis/crystallization_origin_20260925.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.research.crystallization_origin.audit \
  --config configs/analysis/crystallization_origin_20260925.json --watch --workers 2
```

These stages can run concurrently: ancestry waits for each full-source PTM
receipt. Extraction resumes verified 32-frame chunks; ancestry resumes completed
sources. The first five training sources are analyzed before held-out sources.
`--report-only` regenerates tables from completed receipts without recomputing
clusters. Neither command submits GPU jobs.

All derived arrays and receipts are under the configured external analysis root,
not in the repository. `technical/extraction-state.json` and
`technical/audit-state.json` record completion/failure; per-source progress files
show active extraction and graph construction. A failed source remains explicit
and is never treated as a zero-event source. Resuming requires the same producer
hashes and configuration, or a new output directory for changed scientific rules.

The training-source pilot exposed overly persistent uncertainty after a weak
peripheral ancestry link. Revision 2 retains possible roots while separately
tracking independently strong paths. The original training-only pilot, code,
tables and correction receipt are preserved under `technical/pilot-v1/`;
full-cell PTM extraction was unchanged. No held-out outcome selected this fix.

Use distinct cluster counts to assess how many independent establishment
examples exist. Use center coverage to decide whether additional centers from
the existing trajectories are needed. Use positive-window counts to size the
new forecasting task, retaining source-level grouping. A large window count
alone does not establish a large sample of distinct nuclei.

The regional-birth window count still uses the historical, center-liquid risk
set. It can omit a useful precursor whose central atom is already locally
ordered while its cluster is not yet established. A later regional nucleation
benchmark needs its own causal risk definition and center/origin coverage;
the present count is availability on the existing grid, not the maximum
number of regional examples obtainable from these trajectories.
