# GATr internal directions along Al trajectories

Completed frozen-checkpoint evaluation on the A100 on node07. No training changes.

## What was measured

Checkpoint 1216 (`3534c3cd6065d41ca32d6ed3e77534a79429d80ad3e195c0b14beb8ce36e42eb`), the same release used in the MACE/GATr jitter comparison. 40 fixed-identity tracks, 32,040 test observations, 0–600 ps at 0.75 ps cadence; 22,381 spatial encodings across 70 snapshots. Training-only norm calibration: 420 observations. Ten held-out sources, two per temperature.

The primary displayed feature is the point-numerator triplet before the last MLP, channel 3. Other stages, all eight channels and four triplet types are also exported. This is a rotation-covariant local vector, not a physical position, velocity or crystal axis.

## Time: rapid reorientation at the saved cadence

Mean adjacent signed turn **77.8°**; **37.7%** exceed 90°, **62.6%** exceed 60°. Coverage is **99.62%** after excluding weak vectors. P1=0.177, P2=0.053. Ignoring sign still gives mean axis turn **54.8°**. Correcting for the best-fit local cage rotation gives **77.8°**.

Across all eight point-triplet channels, the mean turn ranges from about 71–82° after block 1 and 69–78° before the final MLP; this behavior is not confined to the displayed channel.

These are large frame-to-frame changes, not a stable orientation trajectory at 0.75 ps resolution. They do not prove mathematical discontinuities between frames; faster trajectory output would be needed to resolve that. Rotation equivariance is a symmetry property and does not imply temporal persistence.

| Feature | Mean turn | Axis turn | P1 | P2 |
|---|---:|---:|---:|---:|
| Block 1 · point triplet | 78.1° | 54.5° | 0.172 | 0.059 |
| Before last MLP · point triplet | 77.8° | 54.8° | 0.177 | 0.053 |
| Before last MLP · ideal bivector | 78.3° | 54.6° | 0.170 | 0.058 |
| Before last MLP · axial bivector | 81.4° | 56.2° | 0.126 | 0.023 |
| Density dipole · 7 Å | 71.8° | 52.8° | 0.264 | 0.095 |
| Density dipole · full support | 55.5° | 45.3° | 0.487 | 0.253 |
| Shape principal axis · 7 Å | 72.5° | 41.2° | 0.242 | 0.341 |

Shape-axis sign is arbitrary; its signed turn/P1 are implementation diagnostics only. Compare its axis turn/P2. Tables retain source values and 95% source-bootstrap intervals.

Among FCC→FCC adjacent pairs, the source-balanced mean turn is **85.4°**. Directional persistence is therefore particularly weak in the crystalline subset. This is consistent with a vector responding to fluctuating local asymmetry; it is not evidence of a persistent lattice axis.

## Space: measured order relative to a phase-matched shuffle

| Separation | P1 | Shuffle P1 | Excess P1 (95% interval) | Excess P2 | Nearest-80 overlap |
|---|---:|---:|---:|---:|---:|
| 0–4 Å | 0.064 | 0.000 | 0.064 [0.060, 0.067] | 0.012 | 0.68 |
| 4–8 Å | -0.005 | 0.000 | -0.005 [-0.006, -0.004] | 0.000 | 0.37 |
| 8–12 Å | -0.004 | 0.000 | -0.005 [-0.007, -0.002] | -0.001 | 0.12 |
| 12–18 Å | -0.003 | -0.000 | -0.003 [-0.006, 0.000] | -0.000 | 0.01 |
| 18–26 Å | 0.003 | 0.000 | 0.003 [-0.002, 0.008] | -0.002 | 0.00 |
| 26–40 Å | -0.000 | -0.000 | -0.000 [-0.002, 0.002] | 0.000 | 0.00 |
| 40–80 Å | -0.001 | -0.000 | -0.001 [-0.001, -0.000] | -0.000 | 0.00 |

Positive excess indicates more parallel directions (P1), or common axes irrespective of sign (P2), than the same snapshot and phase composition shuffled across positions. This does not establish crystal orientation order: the descriptors share atoms, and the sampling concentrates on two local patches. Observed directional order is weak below 4 Å and approximately absent at larger separation. Earlier block-1 point triplets have stronger nearest-neighbor alignment, still short-range. Separate FCC/FCC and unclassified/unclassified spatial tables are included. Compare density-dipole baselines and the 3D viewer.

## What these directions encode

The mean fraction of channel-axis energy in one direction is **0.67362** (1 means collinear axes across channels). P2 alignment with the 7 Å density dipole is **0.024**; with the full-support dipole **0.052**; with the local shape axis **-0.011**.

Plane-normal triplets are exactly zero in the two earlier sampled stages. The small final plane-normal output is not used by the scalar readout. 

## Numerical and readout controls

Five rigid rotations pass: maximum relative vector RMS error **1.7e-05**. Repeated identical inference changes multivectors by **0**. Replacing the final multivector output with zeros leaves z128 exactly unchanged, as expected from the readout code.

| Intervention, 60 observations | Native BF16 z RMS change | Autocast disabled z RMS change |
|---|---:|---:|
| erase_all_token_vector_triplets_after_block1 | 0.000630657 | 0.000630666 |
| erase_all_token_vector_triplets_after_block2_mlp_input | 0 | 0 |
| erase_all_token_vector_triplets_after_block2_output | 0 | 0 |
| erase_all_token_vector_triplets_after_input | 0.0105954 | 0.0105954 |
| independent_atom_angle_scramble_fixed_radii | 1.33588e-05 | 1.33561e-05 |
| radii_times_1p01_fixed_support_weights | 0.000323122 | 0.000323128 |

Earlier geometric streams do influence the output, but erasing all directional triplets just before the final MLP has no detectable effect for these inputs, even with autocast disabled. Do not equate a visually interesting last-layer direction with information actively used by the descriptor. Synthetic interventions are mechanistic probes, not plausible atomic motions.

The trajectories originate from stored float16 positions. This audit measures the encoder on those actual inputs; it does not separately identify thermal motion, structural rearrangements and input-coordinate quantization as causes of angular changes.

## Implications for training

If the goal is a persistent orientational representation, first attach an explicitly used covariant readout to an earlier geometric stream and give it a physically meaningful target. Do not smooth the discarded final multivector output and assume z will improve. Use norm-aware temporal consistency, and verify that a nonzero feature remains sensitive to geometry. For crystal orientation, a vector alone cannot represent all symmetry-equivalent lattice axes; a symmetry-aware tensor or bond-order representation is a better target. Validate persistence, spatial alignment, physical sensitivity and collapse together. For the existing invariant z128, direct-z temporal regularization remains a separate objective.

## Artifacts and reproduction

- [Interactive 40-track / 140-patch explorer](explore.html)
- [Summary figure](plots/overview.png), [time traces](plots/trajectories.png), [3D spatial arrows](plots/spatial-directions.png), [norm sensitivity](plots/norm-sensitivity.png)
- [Exact metric definitions](tables/METRICS.md); all source, channel, phase, lag and distance tables in `tables/`.
- Frozen checkpoint, extraction receipts, GPU identity and interventions in `technical/`.

```bash
conda run -n pointnet-torch214 python -m src.research.gatr_equivariant --config configs/analysis/gatr_equivariant.json
conda run -n pointnet-torch214 python -m src.research.gatr_equivariant.controls --config configs/analysis/gatr_equivariant.json
```

Extraction requires node07/A100; run inside a valid allocation. The default all stage includes intervention controls and reporting. Separate stages temporal, spatial, then controls, then report are also available. Only this frozen Al checkpoint and cohort were tested. Two sources per temperature yield limited uncertainty estimates; weak-vector thresholds are diagnostics, not confidence calibration.
