# Relaxed observations and targets: matched MACE study

One seed; 90 encoder-training sources. Observed origins in frames: [32, 64, 80, 128, 176, 224, 272, 320, 368, 416, 464, 512, 560, 608, 656]. original independent source roles retained for frozen readouts. Limited event counts make this a pilot, not a final ranking. All readouts predict original MD onset at 0.75, 3, 6, 9 and 12 ps. Relaxed input requires per-frame preprocessing at deployment. Timeout exclusions (matched across all arms): assay_windows=0.
Excluded encoders: hot-to-cold-vic-temp01: User stopped remaining encoder training; evaluate completed checkpoints only

| Encoder/input | Readout | Event NLL | 12 ps AP | Detected timing MAE (ps) | Missed / event windows |
|---|---|---:|---:|---:|---|
| cold-control | linear | 0.18678 | 0.16502341625873926 | 2.6011443064715634 | 86 / 117 |
| cold-control | mlp | 0.16138 | 0.3430669894790383 | 2.631565555354323 | 59 / 117 |
| cold-sig-temp1 | linear | 0.18555 | 0.17587570064980157 | 2.541773441915198 | 85 / 117 |
| cold-sig-temp1 | mlp | 0.16145 | 0.33106317219489456 | 2.71145087962663 | 54 / 117 |
| cold-sig-temp3 | linear | 0.18257 | 0.2132601772627973 | 2.611152659928007 | 76 / 117 |
| cold-sig-temp3 | mlp | 0.16530 | 0.28159347286778824 | 2.489570291015434 | 49 / 117 |
| cold-vic-global01 | linear | 0.18246 | 0.17309970920150328 | 2.696835749073576 | 82 / 117 |
| cold-vic-global01 | mlp | 0.16439 | 0.29475327309567984 | 2.5309999006488 | 54 / 117 |
| cold-vic-temp001 | linear | 0.18542 | 0.17052408732694344 | 2.5090419743805112 | 83 / 117 |
| cold-vic-temp001 | mlp | 0.16326 | 0.32346685953544485 | 2.6539177010495947 | 54 / 117 |
| cold-vic-temp01 | linear | 0.18169 | 0.1915449424320827 | 2.581793785318632 | 78 / 117 |
| cold-vic-temp01 | mlp | 0.16417 | 0.29917287265820736 | 2.5255809269252056 | 57 / 117 |
| conditions | linear | 0.20558 | 0.0662676522255053 | 3.255615967186174 | 69 / 117 |
| conditions | mlp | 0.20141 | 0.08699144112619671 | 2.8493389698316514 | 99 / 117 |
| geometry_cold | linear | 0.17675 | 0.24584601427605723 | 2.772675225234974 | 48 / 117 |
| geometry_cold | mlp | 0.15533 | 0.3679800752253542 | 2.7369765554487553 | 40 / 117 |
| geometry_hot | linear | 0.18538 | 0.17719943341116376 | 2.979791013151298 | 73 / 117 |
| geometry_hot | mlp | 0.16922 | 0.2574857653389582 | 2.8006468716599255 | 55 / 117 |
| hot-control | linear | 0.20025 | 0.08647342124450838 | 3.360448127520099 | 99 / 117 |
| hot-control | mlp | 0.18466 | 0.1751139016043893 | 3.063276792025946 | 70 / 117 |
| hot-vic-temp01 | linear | 0.20074 | 0.10552894809392759 | 2.948288909296869 | 89 / 117 |
| hot-vic-temp01 | mlp | 0.18999 | 0.17526079166588823 | 3.280212023941649 | 69 / 117 |
| original_geometry | linear | 0.19194 | 0.1321622655946639 | 2.9337648231728655 | 92 / 117 |
| original_geometry | mlp | 0.17684 | 0.24404425079762276 | 2.939340518845658 | 61 / 117 |
| parent_cold | linear | 0.18626 | 0.16249207149914147 | 2.6963332120361954 | 79 / 117 |
| parent_cold | mlp | 0.16207 | 0.3089327714711374 | 2.681629486671085 | 53 / 117 |
| parent_hot | linear | 0.20018 | 0.08771546569174996 | 3.412844955105618 | 97 / 117 |
| parent_hot | mlp | 0.18767 | 0.17357092787324416 | 3.136474979772443 | 67 / 117 |

Positive NLL gains favor the named method against instantaneous→instantaneous. Intervals resample whole test sources within temperature; they exclude seed uncertainty. Reconstruction scores across thermal versus quenched target domains do not establish a better encoder. The original-geometry control has its historical descriptor support; paired hot/cold geometry controls use exactly the same tracked 80 atoms as these encoders.

Completed readouts: 28/28. Reference: hot-control.

## Encoder development diagnostics

| Run | Physical | TDA | Raw rank | Correlation rank |
|---|---:|---:|---:|---:|
| hot-control | 0.3141 | 0.2468 | 1.30 | 1.77 |
| cold-control | 0.2434 | 0.2384 | 1.17 | 1.58 |
| cold-sig-temp1 | 0.2434 | 0.2388 | 2.05 | 2.57 |
| cold-sig-temp3 | 0.2439 | 0.2397 | 4.05 | 4.51 |
| cold-vic-temp001 | 0.2446 | 0.2373 | 1.31 | 1.32 |
| cold-vic-temp01 | 0.2403 | 0.2327 | 8.45 | 8.48 |
| cold-vic-global01 | 0.2404 | 0.2326 | 8.76 | 8.75 |
| hot-vic-temp01 | 0.3133 | 0.2415 | 11.21 | 11.21 |
