# Does the physical snapshot embedding retain local topology?

Compare completed physical MACE/cuEquivariance and axial GATr snapshot encoders
on the same current-frame 144-channel persistence-image targets. Both parent
encoders used 2,048 updates and seed 20260919; neither was trained on TDA.

Use all 38,400 fixed windows and the existing source-disjoint splits. Freeze
the encoders and fit matched ridge and nonlinear residual readouts. The target
uses the nearest 80 observed atoms, including the tracked center, at the current
anchor. This is [instantaneous topology](../../docs/research_glossary.md#instantaneous-topology);
it measures present information retained by z, with no future or relaxed target.

The primary comparison is held-out source-balanced error with equal H0/H1/H2
weight. Report block R2, within-frame R2, temperature-specific error and the
observed noncrystalline subset. Include training-mean and condition-only ridge
baselines. Bootstrap whole paired test sources; one seed does not establish
seed robustness. The nonlinear readout tests whether information inaccessible
to a linear readout is still recoverable. See the [exact metric protocol](../../docs/metrics/backbone_tda.md).

Recipe: `configs/local_predictability/backbone_v2/tda_snapshot.json`.

```bash
python -m src.research.backbone_tda --config configs/local_predictability/backbone_v2/tda_snapshot.json
```

Results: `output/local_predictability/tda-backbone-v2-20260917/`.
Implemented and launched September 17 after five focused tests, eight result-layout
tests, and CUDA validation of both frozen encoders and the nonlinear readout.
The experiment is complete. [Results and proposed follow-up](../../output/local_predictability/tda-backbone-v2-20260917/RESULTS.md):
overall nonlinear TDA error is nearly tied (MACE 0.19837, GATr 0.19719).
The noncrystalline linear readout favors GATr in the paired source bootstrap;
the nonlinear difference remains inconclusive. Physical prediction remains
weaker for GATr, predominantly in geometry targets. All findings use one seed.
The operational [launch/resume instructions](../../docs/backbone_tda.md) cover
the existing allocation. No encoder retraining or simulation is part of this experiment.
