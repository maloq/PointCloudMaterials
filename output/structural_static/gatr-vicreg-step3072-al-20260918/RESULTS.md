# GATr–VICReg: standard static Al results

The full analysis completed successfully on the RTX PRO 6000 on September 18,
2026 (Europe/Paris), taking 606 seconds. Open [the gallery](index.html).

The source run completed 4,096 updates; its best selected encoder was update
3,072 (selection score 0.4045023210346699). All 684,723 interior centers in the
six standard Al snapshots were encoded using the original trained z128, fixed
Al scale and complete 16.869063 Angstrom support. The sampling grid uses three
excluded edge layers, as in the existing MACE context analysis. Clustering and
plot settings otherwise match `configs/analysis/static.yaml`.

## Findings

- The first principal component explains 98.33%
  of raw feature variance on the standard 8,000-sample diagnostic subset.
  Variation is strongly concentrated; this is not evidence for seven independent
  physical states or a comparison against another encoder.
- After the standard channel scaling, the clustering retained
  3 PCA components (99.92% variance)
  and fitted seven spherical clusters to all centers. The 3,000-center evaluation
  sample has cosine silhouette 0.5631 and Euclidean silhouette
  0.3194; these measure separation in the prepared feature
  space, not physical classification accuracy. Definitions are in
  [METRICS.md](tables/METRICS.md).
- C1 (stored label 0) grows from 0.033% at 166 ps
  to 84.71% at 240 ps. Its selected representatives
  have face-centered-cubic packing at 170–240 ps and hexagonal-close-packed
  packing at 166 ps under the existing PTM local-template check.
  A representative label does not establish the structure of every cluster member.
- The connected-regime diagnostics find substantial overlap between several
  groups. Use the continuous projections and real structures together with hard
  cluster labels.

## Scope and checks

These exact six static files were included among the broad training sources.
This is descriptive analysis, not held-out accuracy. Frames share source ancestry.
The displayed transition table matches nearby sample centers, not persistent atom
IDs; it is not a kinetic transition matrix.

Exported tensors exactly match the selected training checkpoint. A real prepared
static example reproduces all seven native input tensors and encoder output
exactly. Six-center checks on every analyzed frame give a maximum batch/order
replay difference of 3.5762786865234375e-7. All centers exceed the required support
margin. Five targeted tests passed, and exported metric contracts passed validation.

- [Checkpoint provenance](technical/encoder/provenance.json)
- [Input/output verification](technical/static-verification.json)
- [Full-frame verification and source hashes](technical/structural-inference-protocol.json)
- [Metrics](tables/metrics.csv) and [definitions](tables/METRICS.md)
- [Tracked successful execution](technical/tracked-analysis/execution/run_record.json)
- [Scientific protocol](../../../experiments/structural_pretraining_20260917/STATIC_AL.md)
- [Structural-state terminology](../../../docs/research_glossary.md#shared-structural-pretraining)

The cached inference and exported checkpoint are retained for reproducible reruns.
