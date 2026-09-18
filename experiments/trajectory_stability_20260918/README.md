# Stability of the latest structural encoders along trajectories

Follow-up: [proposed GATr temporal-smoothness continuation](GATR_SMOOTHING_PROPOSAL.md).
This is a scientific proposal; no follow-up training has been launched.

Question: how much do the selected MACE and GATr snapshot states fluctuate along
the same atom trajectories, compared with instantaneous TDA and conventional
structural descriptors?

Use MACE v6 selected step 1465 and GATr v6 selected step 1216 from the completed
Al-only fits. Freeze their hashes and weights before analysis. The newer broad
GATr continuation was still running when this audit started; its selected
export was step 0, inherited from the same Al parent. No training is launched.

Protocol: 10 seeded test sources (2 at each of 400, 450, 500, 510, 520 K), four
seeded tracked atoms per source, all frames from 0 to 600 ps at 0.75 ps cadence.
Every method receives the same source positions and atom centers, with its
declared native neighborhood. Five separate training sources provide the
reference distance scale. Source ancestry is checked against both training
releases. No outcome-selected tracks or temporal smoothing.

Compare training-reference normalized RMS jumps, longer-lag displacement,
second-difference roughness and increment directions. Report variance/effective
rank and coordinate-standardized sensitivity so a small native scale cannot be
mistaken for a stable, informative representation. Bootstrap complete sources
within temperature; retain paired values, numerical repeat checks and hashes.
Definitions are in [the metric protocol](../../docs/metrics/trajectory_stability.md).

```bash
conda run -n pointnet-torch214 python -m src.research.trajectory_stability \
  --config configs/analysis/trajectory_stability.json
```

Stages `prepare`, `encode`, and `report` can be run separately. Completed
reports are immutable; use a new configured output for another evaluation.

Results: [report](../../output/trajectory_stability/al-v6-20260918/README.md),
[gallery](../../output/trajectory_stability/al-v6-20260918/index.html), and
[tables](../../output/trajectory_stability/al-v6-20260918/tables/summary.csv).

Completed: 32,040 matched test observations, 420 training-reference observations.
Normalized 0.75 ps RMS jumps: MACE 0.723, GATr 0.680, instantaneous TDA 0.371,
SOAP 0.587, bond order 0.278. GATr's amplitude is about 5.9% lower than MACE's;
both have second-difference roughness near the independent-frame reference.
TDA's smaller native jump is sensitive to coordinate standardization of almost
absent features. Numerical repeat/batch effects are far below trajectory jumps.
See [interpretation and limits](../../output/trajectory_stability/al-v6-20260918/RESULTS.md)
and [interactive tracks](../../output/trajectory_stability/al-v6-20260918/explore.html).

Limits: one checkpoint/seed per architecture; only Al MEAM trajectories;
previously explored held-out sources; native 0.75 ps cadence; existing float16
full-box coordinates; descriptor supports differ. Observed changes include
physical dynamics, boundary membership changes and quantization.
