# Encoder research handbook

A working reference for understanding **what each encoder was trained to retain,
what its exported state contains, and what the measurements actually establish**.
Scope: GeoFrame through the September23 distance/future factorial and the
completed 35-pass GeoFrameV2 reproduction,
including archived studies, local H100/RTX/A100 work and the H200 results reported
to this repository. This is an evidence snapshot, not a live run monitor.

| I want to… | Start here |
| --- | --- |
| Understand the models and training losses | [Encoder family guide](encoders.md) |
| Understand a metric or compare two results correctly | [Evaluation guide](evaluation.md) |
| See the main findings without reading every run | [Results overview](results.md) |
| Search every indexed report, table and gallery | [Interactive catalogue](../../output/encoder_research/catalogue/index.html) |
| Query/export the underlying evidence | [Database guide](database.md), [SQLite](../../output/encoder_research/catalogue/technical/results.sqlite), [curated CSV](../../output/encoder_research/catalogue/tables/headlines.csv) |
| Follow the broad native snapshot comparison | [Queue and evaluation protocol](../encoder_screen.md), [current results](../../output/encoder_research/screen-20260923/index.html) |
| Find plots or repeat an analysis | [Analysis and visualization guide](analysis.md) |
| Improve how we organize and interpret future research | [Workflow recommendations](improvements.md) |
| Improve crystallization AP | [Prioritized experiment proposal](ap_experiments.md) |
| Follow the new predictive/robustness experiments | [Eight-arm literature-led protocol](../../experiments/robust_onset_20260924/README.md), [Slurm operations](../robust_onset.md) |
| Measure state and temporal-movement dimensions | [Matched 0.75 ps table with noise response](../../output/encoder_research/noise-lag075-20260924/RESULTS.md), [guide](embedding_dynamics.md), [definitions](../metrics/embedding_dynamics.md) |
| Measure response to input noise | [Combined noise/trajectory table](../../output/encoder_research/noise-lag075-20260924/RESULTS.md), [guide](input_noise.md), [definitions](../metrics/embedding_noise.md) |
| Find a dated scientific protocol | [Study index](studies.md), [active experiments](../../experiments/README.md) |

In discussion and new result tables we call the model **Geoformer**, following
the user's terminology. Its implementation classes remain `GeoFrameTransformer`
and `GeoFrameTransformerV2`; older files and archived reports use GeoFrame.
Those internal names do not denote a different model in these comparisons.

Our recurring lesson is that four questions need separate evidence: does an
embedding preserve present structure, organize useful neighbors, respond
appropriately to dynamics, and improve future prediction? Smoothness, embedding
rank, attractive clusters and decoder accuracy can move in different directions.
The database deliberately has **no universal encoder leaderboard**.

The [35-pass GeoFrame trajectory](../../output/geoframe_evolution/epoch34-review-20260923/RESULTS.md)
now separates encoder and projector behavior: liquid-order readout improves in
the encoder while declining in the projector across Al, Ta and Zr. Improved
fault/interface-proxy readout does not imply separate liquid clusters or better
conditional crystallization forecasts. The historical epoch34 VICReg and epoch159
VISReg pictures used different objectives; they are not an unchanged training
trajectory. See the [new evidence and limits](results.md#geoframe-through-35-passes).

A compact history is: canonical-frame GeoFrame → continuous density/MACE controls
→ pretrained MACE topology and context studies → native geometry/velocity/history
models → shared structural MACE/GATr and neighborhood prediction → relaxed-input
studies → BCR conditioning audits → simpler fixed-geometry encoder training and
its distance/future additions. This is a research sequence, not a claim that every
later model outperformed its predecessor.

The catalogue retains negative findings, historical revisions, smoke outputs and
reported-only results. Their presence is not an endorsement or a completed
scientific comparison. The human guide selects meaningful comparisons; the raw
database preserves the supporting evidence without silently normalizing unlike
metrics. H200 summaries are explicitly distinguished from locally available raw
results. Archive copies are not new fits, and the same checkpoint can occur in
several analyses.

To refresh after new results, activate `pointnet-torch214`, register their study
and evidence collection in [catalogue.json](catalogue.json), add checked findings
in [highlights.json](highlights.json), and run:

```bash
python scripts/experiment_registry.py encoders
```

This reads existing reports/tables and writes the catalogue. It does not run
training, inference, simulation, scheduler queries or hardware benchmarks.
Machine storage roots come from `machine.local.yaml`; the archive must be mounted.
See [coverage and refresh instructions](database.md). Source scientific protocols
remain in `experiments/`; this folder is their cross-study guide.

The [new parameter search](../encoder_parameter_search.md) trains28 matched fits after the [29-checkpoint review](../../output/encoder_research/parameter-search-20260923/RESULTS.md). Liquid-neighbor measurements favor the late VISReg raw export despite its two-blob visualization; structure and calibrated prediction remain separate selection criteria.

The [paired MACE + Epi study](../../experiments/mace_paired_epi_20260923/README.md) compares direct temporal alignment with VICReg versus geometric Epi regularization (with/without a variance floor), using two scratch seeds and24 complete passes. Results are pending; this is separate from the conditional-JEPA checkpoint screen.
