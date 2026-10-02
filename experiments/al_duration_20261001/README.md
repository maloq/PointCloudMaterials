# Duration needed for the 150 main Al trajectories

**Latest objective:** the user clarified that 50% crystallinity is sufficient,
and that 90% refers to other runs reaching that target. The
[halfway and peer-censoring follow-up](../../output/al_duration/half-crystal-20261001/README.md)
supersedes the earlier full-plateau duration recommendation for this objective.
Run `python -m src.analysis.al_half_transform --config configs/analysis/al_half_transform_20261001.json`
on the frozen observations; it introduces no new PTM calculation or simulation.
An individual confirmed-halfway stop plus a 6 ps prediction tail would save
46.64% of historical measurement duration. The same-temperature 90%-peer cap
remains unobserved at 400 K; remaining sources are censored, never relabeled as
permanent non-crystallizers. See [definitions](../../docs/metrics/al_half_transform.md).
The subsequently authorized [simulation queue version](../../docs/simulations/al_main_half_stop_20261001/README.md)
applies this target only to unstarted descendants; the numerical audit and
original source release remain frozen.

Question: can dense observations stop before 600 ps without losing bulk
crystallization, later annealing and the interface/defect information needed
for local representation research?

Audit all 150 independent historical melt ancestors, 30 per temperature at
400/450/500/510/520 K, with their frozen ancestry roles. Separately audit the
21 complete 520 K dense descendants available at capture. Two additional
0.01-ps daughters are excluded. This is a descriptive physical audit including
held-out ancestry, with no model fitting, promotion, stopping-rule selection or
modification of source splits or simulation jobs.

Reuse checksum-verified historical PTM/cluster/thermodynamic arrays and the
already frozen 20-pair replay results; compute only missing dense structural
observations. PTM RMSD cutoff 0.1, FCC/HCP/BCC crystal, 3.5 Å selected crystal
connectivity. Use exact 15 ps structural samples and trailing 15 ps energy/volume
averages. Retain the historical 0.75 ps full crystal curves separately. Evaluate
fixed cutoffs and a declared hypothetical causal plateau rule against later
observed change. Thresholds and censoring are in the metric definition, not
tuned to maximize success. PTM Other is not a liquid label.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OVITO_THREAD_COUNT=1 \
python -m src.analysis.al_duration \
  --config configs/analysis/al_duration_all150_20261001.json
```

Use conda `pointnet-torch214`. Completed outputs are immutable; use a new output
revision for a new capture or calculation change. Availability is frozen in the
recipe, and a changed completed-source set fails explicitly. No automated tests
or W&B evaluation runs are added.

- [Findings and recommendation](../../output/al_duration/all150-20261001/README.md)
- [Figures](../../output/al_duration/all150-20261001/plots/duration_by_temperature.png)
- [Definitions](../../docs/metrics/al_duration.md)
- [Recipe](../../configs/analysis/al_duration_all150_20261001.json)
- [Producer](../../src/analysis/al_duration.py)
- [Simulation campaign](../../docs/simulations/al_main_010ps_20260927/README.md)

The shared 600 ps maximum remains useful. A 300/450/525 ps cutoff loses >5
percentage points of terminal-window growth in 87/31/13 of the 150 historical
sources. A 70%-crystal plateau rule offers about 15% hypothetical historical
measurement-time savings but misses later changes in 9 of its 88 early stops;
dense source 876 undergoes nearly another ten-percentage-point order increase
after a seemingly stable interval. Do not deploy that rule. The dense evidence
is completion-limited at 520 K and cannot establish other-temperature kinetics.
