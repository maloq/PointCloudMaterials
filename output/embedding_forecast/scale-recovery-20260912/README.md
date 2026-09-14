# H100 forecast recovery — September 12, 2026

Preparation failed on the `/home/ids` quota after 35/125 shards; the dependent training jobs never started. Recovery reclaims 25.353 GiB of completed-run training caches and converts forecast embeddings to verified float16 storage. The same two 32-epoch fits follow successful preparation.

[Diagnosis and protocol](../../../experiments/embedding_forecast_20260911/RECOVERY_20260912.md). [Cleanup result](technical/cache-cleanup-result.json). [Conversion progress](technical/embedding-conversion.log). [Conversion audit](technical/embedding-conversion.json). [Concrete restart plan](technical/restart-plan.json).

The cache remains under `/home/ids/vmorozov/training-cache/embedding-forecast-full-20260911`; results use the original `output/embedding_forecast_20260911/scale/runs` path.

Submitted: preparation **990768**, autoregressive fit **990769**, direct fit **990770**, comparison **990771**. Preparation waits for H100 resources; downstream jobs wait for successful dependencies. [Submission receipt](technical/submission.json). [Verified scheduler state](technical/restart-verification.json).
