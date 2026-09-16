# MACE context information recovery

Exploratory source-held-out comparison. Lower TDA MSE is better; higher increment error reduction is better. Increment readouts use embeddings at both times and are not forecasts. Frozen fusion shares one backbone; trained fusion combines two separately trained backbones. Nonlinear means a validation-selected residual MLP added to ridge, with unchanged encoder features.

| Method | Readout | Instantaneous TDA MSE | Relaxed TDA MSE | q6 change error reduction |
|---|---|---:|---:|---:|
| frozen-fusion | nonlinear | 0.040124 | 0.030912 | 20.46% |
| frozen-fusion | ridge | 0.040124 | 0.031535 | 20.81% |
| frozen-halo_center | nonlinear | 0.077041 | 0.048667 | 19.64% |
| frozen-halo_center | ridge | 0.077556 | 0.049863 | 20.00% |
| frozen-halo_inner | nonlinear | 0.041357 | 0.030018 | 2.26% |
| frozen-halo_inner | ridge | 0.041357 | 0.030925 | 4.46% |
| frozen-mean80 | nonlinear | 0.017009 | 0.034602 | 10.45% |
| frozen-mean80 | ridge | 0.017148 | 0.035753 | 11.32% |
| trained-fusion | nonlinear | 0.040177 | 0.030925 | 20.85% |
| trained-fusion | ridge | 0.040212 | 0.031326 | 21.92% |
| trained-halo_center | nonlinear | 0.078841 | 0.046358 | 20.48% |
| trained-halo_center | ridge | 0.078912 | 0.046926 | 21.17% |
| trained-halo_inner | nonlinear | 0.041476 | 0.030049 | 2.16% |
| trained-halo_inner | ridge | 0.041476 | 0.030984 | 3.26% |
| trained-mean80 | nonlinear | 0.017008 | 0.034684 | 10.42% |
| trained-mean80 | ridge | 0.017150 | 0.035616 | 11.17% |
| dual_physics | ridge | 0.040117 | 0.031440 | 21.14% |
| dual_physics | trained_head | 0.047188 | 0.032241 | 24.16% |
| dual_ssl | ridge | 0.040124 | 0.031535 | 20.81% |
| dual_ssl | trained_head | 0.048872 | 0.033198 | 25.39% |

[Full comparison](tables/comparison.csv) and [metric definitions](tables/METRICS.md).

![Cached readout comparison](plots/cached-readouts.png)
