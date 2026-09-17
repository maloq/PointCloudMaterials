# Local predictability: incremental descriptor results

Single training seed 20260919. Source intervals condition on this seed.

| Model | Horizon (ps) | Test log loss | Test Brier | Test AP |
| --- | --- | --- | --- | --- |
| linear-condition | 9 | 0.1392 | 0.0305 | 0.0426 |
| linear-condition | 48 | 0.4968 | 0.1603 | 0.2424 |
| mlp-condition | 9 | 0.1361 | 0.0304 | 0.0642 |
| mlp-condition | 48 | 0.4827 | 0.1580 | 0.3010 |
| linear-packet_H0 | 9 | 0.1181 | 0.0283 | 0.1554 |
| linear-packet_H0 | 48 | 0.4865 | 0.1570 | 0.2845 |
| mlp-packet_H0 | 9 | 0.1086 | 0.0268 | 0.2499 |
| mlp-packet_H0 | 48 | 0.4649 | 0.1504 | 0.3247 |
| linear-packet_H3 | 9 | 0.1070 | 0.0260 | 0.2813 |
| linear-packet_H3 | 48 | 0.4785 | 0.1541 | 0.3256 |
| mlp-packet_H3 | 9 | 0.0998 | 0.0247 | 0.3442 |
| mlp-packet_H3 | 48 | 0.4610 | 0.1476 | 0.3584 |
| linear-packet_H12 | 9 | 0.1045 | 0.0254 | 0.3097 |
| linear-packet_H12 | 48 | 0.4792 | 0.1544 | 0.3373 |
| mlp-packet_H12 | 9 | 0.1002 | 0.0241 | 0.3508 |
| mlp-packet_H12 | 48 | 0.4587 | 0.1457 | 0.3684 |
| linear-packet_H48 | 9 | 0.1085 | 0.0264 | 0.2872 |
| linear-packet_H48 | 48 | 0.4963 | 0.1610 | 0.3144 |
| mlp-packet_H48 | 9 | 0.1062 | 0.0255 | 0.3161 |
| mlp-packet_H48 | 48 | 0.4807 | 0.1550 | 0.3385 |
| linear-packet_repeat12 | 9 | 0.1172 | 0.0283 | 0.1644 |
| linear-packet_repeat12 | 48 | 0.4856 | 0.1566 | 0.2882 |
| mlp-packet_repeat12 | 9 | 0.1072 | 0.0269 | 0.2580 |
| mlp-packet_repeat12 | 48 | 0.4640 | 0.1487 | 0.3359 |
| linear-packet_plus_center_H0 | 9 | 0.1131 | 0.0275 | 0.1997 |
| linear-packet_plus_center_H0 | 48 | 0.4821 | 0.1551 | 0.3050 |
| mlp-packet_plus_center_H0 | 9 | 0.1062 | 0.0265 | 0.2651 |
| mlp-packet_plus_center_H0 | 48 | 0.4638 | 0.1492 | 0.3382 |
| linear-packet_plus_shell25_H0 | 9 | 0.1112 | 0.0276 | 0.2071 |
| linear-packet_plus_shell25_H0 | 48 | 0.4732 | 0.1510 | 0.3273 |
| mlp-packet_plus_shell25_H0 | 9 | 0.0812 | 0.0218 | 0.4586 |
| mlp-packet_plus_shell25_H0 | 48 | 0.4305 | 0.1365 | 0.4508 |

Native models and dense alarm/timing assays are separate stages; unfinished stages are not results.
