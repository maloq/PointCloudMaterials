# Relaxed observations and targets: matched MACE pilot

One seed; 20 encoder-training and 10 development sources. Two observed origins per source (48 and 276 ps); original independent source roles retained for frozen readouts. Limited event counts make this a pilot, not a final ranking. All readouts predict original MD onset at 0.75, 3, 6, 9 and 12 ps. Relaxed input requires per-frame preprocessing at deployment.

| Encoder/input | Readout | Event NLL | 12 ps AP | Detected timing MAE (ps) | Missed / event windows |
|---|---|---:|---:|---:|---|
| conditions | linear | 0.09563 | 0.05524039084981469 | 3.478099994597836 | 5 / 8 |
| conditions | mlp | 0.10839 | 0.05055032414543441 | 4.767287136226334 | 6 / 8 |
| geometry_cold | linear | 0.08904 | 0.19651042342287436 | 3.016103584169817 | 2 / 8 |
| geometry_cold | mlp | 0.07388 | 0.15793938511828282 | 3.0814092149092507 | 2 / 8 |
| geometry_hot | linear | 0.09178 | 0.11093524494669225 | 3.0969152147124257 | 3 / 8 |
| geometry_hot | mlp | 0.08081 | 0.15729552639167707 | 3.375141508783254 | 3 / 8 |
| hot_to_relaxed | linear | 0.09781 | 0.06790302797146022 | 2.499258402368236 | 4 / 8 |
| hot_to_relaxed | mlp | 0.09678 | 0.18303258950095477 | 4.6927266319155505 | 5 / 8 |
| instantaneous | linear | 0.09496 | 0.07439177274459018 | 2.5830773105926976 | 4 / 8 |
| instantaneous | mlp | 0.09223 | 0.08880340726692397 | 4.663248301978443 | 5 / 8 |
| original_geometry | linear | 0.09243 | 0.10158616941504582 | 3.183303283451215 | 3 / 8 |
| original_geometry | mlp | 0.08461 | 0.10479990258744881 | 3.286437784713743 | 3 / 8 |
| parent_cold | linear | 0.09922 | 0.09813302496454196 | 1.516352026527973 | 5 / 8 |
| parent_cold | mlp | 0.07897 | 0.17213039654747198 | 3.1302600285076845 | 2 / 8 |
| parent_hot | linear | 0.09559 | 0.07282046301856032 | 2.6119335498199194 | 4 / 8 |
| parent_hot | mlp | 0.09368 | 0.08772781659465478 | 4.680315703970361 | 5 / 8 |
| relaxed_to_relaxed | linear | 0.09671 | 0.1085576427886085 | 1.3599941412159453 | 4 / 8 |
| relaxed_to_relaxed | mlp | 0.07898 | 0.14388397969673614 | 3.1225354406003425 | 2 / 8 |

Positive NLL gains favor the named method against instantaneous→instantaneous. Intervals resample whole test sources within temperature; they exclude seed uncertainty. Reconstruction scores across thermal versus quenched target domains do not establish a better encoder. The original-geometry control has its historical descriptor support; paired hot/cold geometry controls use exactly the same tracked 80 atoms as these encoders.

Completed readouts: 18/18. Benchmark results are under benchmark/.
