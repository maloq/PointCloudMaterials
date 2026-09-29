# Same-parent Al trajectory comparison

Question: how much do the new exact-0.1-ps Al reruns differ from their original
0.75-ps histories, in detailed atomic paths and in bulk crystallization?

Compare all 20 completed pairs as observed on September 29, all at 520 K.
The three interrupted descendants and 127 unstarted sources are excluded by
availability, not by a crystallization criterion. Coverage contains 15 training,
two selection, two calibration and one test ancestry. This is a descriptive
simulation audit: no training, tuning, source reassignment or benchmark promotion.

Verify identical prepared-liquid/melt-restart bytes, velocity seed and atom IDs.
The integration step changes 3 fs to 2 fs; physical equilibration duration,
thermostat/barostat settings, pressure and momentum-removal interval remain
matched. Measurement zero follows 15 ps equilibration in both runs. Numerical
execution can differ as well, so this is not a controlled estimate of timestep
bias alone. Independent same-timestep replicas would be needed for that claim.

At the 401 exact common times (0:1.5:600 ps), compare periodic, box-corrected
same-atom position distance, velocity correlation and logged thermodynamics.
Recompute full-cell PTM identically for both float16 exports on the declared
structural grid; compare crystal fraction, largest crystal cluster, neighbor
retention, atom-label overlap and coarse bulk-transformation landmarks. Compare
matched-parent crystal curves with shuffled parent pairings at the same temperature.
Per-source bootstrap intervals preserve the paired ancestry unit.

Use conda `pointnet-torch214`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen \
python -m src.analysis.al_replay --config configs/analysis/al_replay_completed20_20260929.json
```

Re-export existing numerical results/plots with `--stage report`; do not rerun
completed calculations into the same output. No W&B run is created.

- [Metric definitions](../../docs/metrics/al_replay.md)
- [Outputs](../../output/al_replay/completed20-20260929/README.md)
- [Simulation campaign](../../docs/simulations/al_main_010ps_20260927/README.md)
