# Frozen embedding information at 0.75–12 ps

Reuse the exact 191,688 natural at-risk rows, 150 Al Lee-MEAM sources and existing
train/selection/calibration/test roles of neighborhood_crystallization_v2. This is
a historically reused cohort, not a new test set. Five frozen exports are pinned
by checkpoint, feature and population hashes; pretraining ancestry records must
show no calibration/test overlap. No feature extraction or encoder updates occur.

Recompute discrete first-onset bins at [.75,3,6,9,12] ps from actual onset frames,
not the coarse historical event bins. Check onset against the checksummed assay's
three-frame sustained PTM labels. Events after 12 ps are right survival through
all five bins. Inputs never contain PTM or future confirmation. Retain the same
conservative at-risk population and ensure followup includes confirmation.

Feature groups trace to physical_packet, bond_order and shell_features:
radial/pair distributions plus five radial moments (69), angular Legendre moments
(16), q4/q6/w4/w6/qbar6/coherence (6), density/coordination (2), local motion (43),
outer-shell count/radius (4), outer-shell motion (8), and changes in geometry85 +
order8 from t−3,t−6,t−12 ps to t (279). Outer shells are 7–17 and 17–25 Å.
These broader observations differ from the snapshot encoder's input support.
No TDA add-back is present; this assay tests existing descriptor caches only.

All hazard models share 562 input slots: 128 frozen features, 427 observation
slots, seven known temperature/time conditions. Absent slots are zero. Fit
source-weighted training-only means and standard deviations (floor .001). Thus
within each head family, initial weights, parameter counts, training samples,
1024 updates, batch512, LR .0005, warmup/cosine, weight decay1e-4 and clipping5
are matched. Linear, width128 SiLU/LayerNorm and stronger 256×256 heads use one
seed20260920. Strong heads test embedding-only and full-observation inputs.
The shuffled full-observation negative control independently permutes rows within
role and temperature with no label access; it is an artificial control, not a
physical causal predictor. Select checkpoints by source-equal development hazard
NLL. Use calibration-only thresholds targeting at most5% source-weighted FPR.

Report source-equal hazard NLL, per-horizon AP/AUROC/log loss/Brier/precision/recall
and realized FPR. Timing uses conditional mean hazard-bin midpoints. MAE includes
only detected event windows and must accompany misses and timing-within3ps recall.
Overlapping windows are not independent events; origin spacing is3ps.

Paired add-back gain is NLL(z)−NLL(z+block), or the analogous binary log-loss/Brier
difference. Intervals resample independent test sources within temperature strata,
1000 draws, retaining paired models. AP gains are point estimates only. Intervals
exclude training-seed uncertainty and lack multiplicity correction: exploratory.

Physical decoders reconstruct the 148 current local/outer descriptors from z plus
conditions (also a conditions-only control), using linear or stronger heads. Loss
is equal-weight mean of seven feature-block MSEs after training-only standardization.
Select by source-equal development loss. Test block R² is
1−sum_j E_source[(prediction−target)^2]/sum_j Var_source(target_j).
Component R² is undefined for zero test variance, never silently clipped.

An add-back gain means information unavailable to the tested readout; it does not
prove information-theoretic absence. Strong decoder recovery distinguishes poor
accessibility from missing observed inputs. Additional motion/history/context
cannot automatically be attributed to compression by a position-only encoder.


Table export: 2026-09-22T04:31:25.226277+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
