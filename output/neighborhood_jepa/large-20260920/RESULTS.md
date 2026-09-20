# Expanded MACE neighborhood-JEPA results — 20 September 2026

Two paired-GPU, one-seed runs used width64 MACE (657,299 encoder parameters), 32,768 native Al training anchors across 90 lineages, physical and instantaneous TDA anchors, fixed angular moments and SIGReg. Each snapshot is encoded independently.

## Training and reconstruction

| Run | Global batch | Updates achieved / planned | Sampled epoch equivalents | Selected checkpoint update | Physical error | TDA error | Selection score | Invariant effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| node53 | 1024 | 2692 / 3040 | 84.125 | 2560 | 0.3822 | 0.1484 | 0.4193 | 1.809 |
| node59 | 2048 | 1520 / 1520 | 95.000 | 1520 | 0.4692 | 0.1655 | 0.5106 | 1.286 |

Node59 completed all 1,520 updates (95 sampled epoch equivalents). Node53 reached the three-hour deadline at 2,692/3,040 updates (84.125 equivalents), preserved its checkpoint, and stopped without numerical divergence. The launcher treated this expected time-budget stop as an error, so its automatic probe was skipped; that evaluation was completed separately on the saved development-selected checkpoint. Training was not extended. Original failure evidence is retained.

Selection is source-equal present Physical85 + 0.25 TDA144 on 480 development observations / 15 sources. Both large runs share normalization; raw reconstruction scores are not directly comparable with the smaller release, which has different train-fitted target scales. More updates and a different batch prevent attributing the node53/node59 difference solely to hardware or batch size.

Next-frame physical error at 0.75 ps is 0.6641 (node53) and 0.6777 (node59), versus 0.9474 persistence and 0.6059 train-fitted geometry ridge. Both improve on persistence but remain worse than ridge. Low invariant effective rank (1.3–1.8 of 128 exported dimensions) is a warning about concentrated variation, not proof that all remaining dimensions contain no information.

## Matched frozen crystallization prediction

Each readout receives one current frozen 128-dimensional invariant embedding plus known temperature/time. The same nonlinear hazard-probe training and historical 30-test-source assay are used. AP means average precision (higher better); event NLL is lower better. These are point estimates from one seed, not significance claims.

| Encoder | Event NLL | 9 ps AP | 24 ps AP | 96 ps AP | 24 ps timing MAE (ps) | Missed event windows / event windows |
|---|---:|---:|---:|---:|---:|---:|
| Large node53 | 1.1177 | 0.1963 | 0.2216 | 0.4734 | 5.439 | 1853 / 2883 |
| Large node59 | 1.1169 | 0.2066 | 0.2275 | 0.4781 | 5.349 | 1898 / 2883 |
| Small corrected JEPA E | 1.1202 | 0.1270 | 0.1728 | 0.4566 | 5.490 | 2032 / 2883 |
| Old VICReg MACE local | 1.1140 | 0.2174 | 0.2307 | 0.4887 | 5.307 | 1852 / 2883 |
| Old VICReg MACE expanded | 1.1143 | 0.2150 | 0.2347 | 0.4821 | 5.290 | 1846 / 2883 |
| Old VICReg GATr | 1.1514 | 0.0953 | 0.1486 | 0.4160 | 5.497 | 2183 / 2883 |
| Hand-crafted geometry baseline | 1.1110 | 0.2365 | 0.2772 | 0.4801 | 5.119 | 1857 / 2883 |

Timing MAE is conditional on detected event windows. Windows overlap and are not independent events; misses must accompany the error. Thresholds target 5% false positives on calibration sources, not a guaranteed 5% on test. Node59 actually has 10.44% test FPR and 31.32% source-weighted window recall at 24 ps; its 5.35 ps conditional timing MAE excludes 1,898 of 2,883 event windows. Its 9 ps conditional timing MAE is 2.01 ps, with 577/974 windows missed.

For node59, sampled-center spatial fraction MAE is 0.1597 and thresholded Jaccard is 0.1207 at 24 ps. These evaluate sampled at-risk centers, not full-cell phase maps.

Scaling improved the corrected JEPA E crystallization point estimates over the smaller run, but node59 does not outperform the old VICReg MACE or hand-crafted geometry baseline. Reconstruction and SIGReg loss reductions alone do not establish better predictive representations. Data volume, model width and training budget changed together, so this is not an isolated capacity ablation. No new test cohort or seed replication was introduced.

## Source artifacts

- [Node53 run](../large-node53-20260920/README.md)
- [Node59 frozen evaluation](../large-node59-20260920/CRYSTALLIZATION.md)
- [Earlier matched encoders and baselines](../v2-native-al-20260920/CRYSTALLIZATION.md)
- Definitions: `docs/metrics/neighborhood_jepa_large_v2.md` and `docs/metrics/neighborhood_crystallization_v2.md`; per-run metric exports preserve implementation identities.

Both frozen crystallization evaluations are complete. Node53 remains explicitly recorded as time-limited training.


Baseline input clarification (verified against the frozen producer): the historically named `geometry-baseline` is a **geometry + motion + bond-order descriptor baseline**, not positions-only. Its 136 descriptor inputs comprise 85 geometric packet components, 43 velocity-derived packet components, and 8 order/density/coordination components. The packet uses relative positions/velocities within 7 Angstrom with taper from 5 to 7 Angstrom; bond-order statistics use the center and its 12 nearest neighbors and their bonds. Seven temperature/time conditions are appended to both descriptor and encoder probes. The snapshot MACE/GATr encoders do not receive velocities. Therefore, this comparison is useful as a richer physical-input reference, but does not establish that hand-crafted geometry alone outperforms a learned representation on matched inputs. A positions-only descriptor ablation (93 descriptors plus conditions) has not been run in this comparison. No TDA, PTM class labels or future observations enter these descriptor features. Historical artifact names and scores are unchanged.
