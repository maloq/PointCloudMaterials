# Temporal-only backtracking continuation

The previous fit checkpointed at update 250. Its physical/TDA selection score
was 0.1919352710; the new run reproduced that score exactly before updating.
Weights, optimizer moments, RNG, target moments and the global schedule were
preserved. The total remains 1,492 updates / twelve epoch equivalents.

The fixed coefficient changed from 0.001 to 21, selected on three training-only
full temporal batches. The raw formula is unchanged. Reference temporal loss
fractions are 1.03–1.08%, with encoder-gradient norm ratios 9.17–9.89% relative
to the other combined objectives. The coefficient was limited by the gradient
criterion. These are calibration measurements, not a guarantee for every later
update. No dynamic gradient calibration runs during training.

Spatial updates now load/encode two views; temporal updates use three.
All 29 relevant tests passed. The first new spatial updates have exactly zero
backtracking contribution. The first temporal updates have 1.04–1.08%
contribution, with finite losses and gradients. Early steady spatial timing is
10.8 s versus about 22.0 s previously; temporal updates remain about 16.3 s.
The equal-mixture estimate is approximately 1.4x throughput; this is a small
startup sample, not a separate hardware benchmark.

The continuation is detached on H100 allocation 997799 with compact W&B logging:
https://wandb.ai/teshbek/PointCloudMaterials/runs/gatr-temporal-backtracking-0918

Raw calibration values, producer hashes and test output are in technical/.
The fit and frozen queue live in sibling gatr-temporal-backtracking-20260918
and gatr-temporal-backtracking-campaign-20260918 outputs.
