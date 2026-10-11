# Unrelaxed response clustering spaces

All 74,880 uniform training rows fit each transformation and K7 MiniBatchKMeans;
all 24,960 uniform test rows retain their fixed Al64 identities. No relaxed rows,
selection/calibration fitting, readout training or checkpoint selection occurs.
The existing all-training joint-descriptor K7 assignment is the frozen reference.
Native z128, regularized whitening of z128, and the existing predictor head's
256 normalized future-feature means are compared. Whitening uses training mean
and covariance C, with (0.95 C + 0.05 trace(C)/128 I)^(-1/2).
KMeans batch4096, n_init3, max_iter200, reassignment_ratio0; all three declared
seeds are reported without choosing the best agreement. Inference batch256,
float32, no TF32/autocast; geometry-only unrelaxed nearest80 inputs.

Primary many_to_one = sum over neural clusters of their largest descriptor
intersection, divided by N. Each cluster maps to its own majority descriptor;
several clusters may share that descriptor. Mapping is descriptive on evaluation
rows, not a learned predictor or untouched generalization score. Fixed K=7 is
essential because purity can rise with fragmentation.
one_to_one maximizes the contingency trace over permutations divided by N.
adjusted_rand and adjusted_mutual_information are sklearn chance-adjusted agreement.
occupied_clusters counts nonempty test clusters; largest_cluster_fraction is the
largest test occupancy/N. effective_clusters = exp(-sum p log p), with zero terms
omitted. stability_ari_to_first_seed compares test assignments with the first
fixed clustering seed for this same model/space (first seed has identity value1).
Seed error bars are standard deviations, not independent-source uncertainty.
rows counts test observations. Model/space/seed columns identify the treatment.
No W&B runs are opened for these frozen diagnostics. Intermediate vectors stay in
memory; checkpoints, normalization, assignments, centers and metrics are retained.
