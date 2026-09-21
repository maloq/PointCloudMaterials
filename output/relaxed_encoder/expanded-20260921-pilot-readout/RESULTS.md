# Preliminary crystallization readouts of expanded relaxed encoders

Fixed earlier assay cohort: 758 test windows, 8 positive by 12 ps; observation frames [64, 368]. The cohort was fixed before these runs and is not selected by relaxation completion speed. This is separate from the larger pending 15-origin evaluation. One seed; use these sparse-event results as diagnostics, not hyperparameter selection.

Each encoder is frozen. Matched linear and MLP hazard readouts use original source splits and original MD onset labels; thresholds are calibrated at 5% false-positive rate. Timing MAE includes detected event windows only; misses must be considered alongside it.

| Encoder | Readout | Event NLL | 12 ps AP | AUROC | Timing MAE (ps) | Misses / events |
|---|---|---:|---:|---:|---:|---:|
| cold-control | linear | 0.0954 | 0.1129 | 0.8606 | 1.3602 | 4/8 |
| cold-control | mlp | 0.0793 | 0.1267 | 0.8460 | 3.0897 | 2/8 |
| cold-sig-temp1 | linear | 0.0981 | 0.0953 | 0.8432 | 1.4903 | 5/8 |
| cold-sig-temp1 | mlp | 0.0801 | 0.1448 | 0.8475 | 3.0968 | 2/8 |
| conditions | linear | 0.0956 | 0.0552 | 0.7189 | 3.4781 | 5/8 |
| conditions | mlp | 0.1084 | 0.0506 | 0.6303 | 4.7673 | 6/8 |
| geometry_cold | linear | 0.0890 | 0.1965 | 0.8868 | 3.0161 | 2/8 |
| geometry_cold | mlp | 0.0739 | 0.1579 | 0.8628 | 3.0814 | 2/8 |
| geometry_hot | linear | 0.0918 | 0.1109 | 0.8241 | 3.0969 | 3/8 |
| geometry_hot | mlp | 0.0808 | 0.1573 | 0.8803 | 3.3751 | 3/8 |
| hot-control | linear | 0.0943 | 0.0791 | 0.7891 | 2.6074 | 4/8 |
| hot-control | mlp | 0.0924 | 0.0866 | 0.7272 | 4.6771 | 5/8 |
| original_geometry | linear | 0.0924 | 0.1016 | 0.8071 | 3.1833 | 3/8 |
| original_geometry | mlp | 0.0846 | 0.1048 | 0.8372 | 3.2864 | 3/8 |
| parent_cold | linear | 0.0992 | 0.0981 | 0.8452 | 1.5164 | 5/8 |
| parent_cold | mlp | 0.0790 | 0.1721 | 0.8596 | 3.1303 | 2/8 |
| parent_hot | linear | 0.0956 | 0.0728 | 0.7959 | 2.6119 | 4/8 |
| parent_hot | mlp | 0.0937 | 0.0877 | 0.7134 | 4.6803 | 5/8 |

Completed readouts: 18/18. Full horizon metrics (0.75, 3, 6, 9, 12 ps) are in readouts/tables/. Source-bootstrap NLL gains against hot-control are in comparison/tables/.
