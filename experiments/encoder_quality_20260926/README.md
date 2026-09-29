# Does the latest native MACE retain useful liquid and interface information?

**Completed:** [Eight-checkpoint results and interpretation](RESULTS.md).

Freeze the latest completed 128-channel, 128-dimensional native MACE exports:
scratch, VICReg-initialized and Epi+variance-initialized likelihood fits, each on
observed and relaxed inputs; also evaluate the two self-supervised epoch-12
checkpoints before fine-tuning. Supervised checkpoints retain their original
validation-NLL selector after epoch12 of24. Pretrained checkpoints retain fixed
epoch12 selection without crystallization labels. Physical-reconstruction
pretraining is excluded. This is an evaluation of existing encoders.

Three complementary questions:

1. Do latent distances resolve noncrystalline structure? Use the fixed Al/Ta/Zr
   static assay: liquid-order/topology retrieval and regression with density
   controls, interface/fault/nonbulk classification, and distance-matched spatial
   boundary discrimination with shuffled and collapsed controls. UMAP and K=7
   panels illustrate results; neither selects a model or defines ground truth.
2. Is the representation numerically consistent and sensitive to structural
   change? Verify rotation/permutation/recentering/periodic-image consistency,
   measure a fixed noise-amplitude sweep, and associate latent changes at0.75 ps
   with changes in geometric descriptors over all30 held-out observed sources.
   This temporal association does not establish causation or displacement matching.
3. What predictive information survives encoding? On exactly the fixed al64_v1
   all64 population, fit matched linear/MLP likelihood readouts on embeddings,
   same-domain physical descriptors and their concatenation. Include the fitted
   constant event-distribution control and train-neighbor forecasts. Descriptor
   add-back helps identify information accessible beyond the frozen embedding;
   differing predictor dimensions/capacities remain a limitation of that inference.

Prediction uses all126545 fixed observations, no row drops or resplitting.
Twenty-four readout epochs, batch256, selection by validation event NLL from
epoch12 onward, calibration on its separate role, source-paired test Brier/log-loss
intervals, and diagnostic AP. The test cohort has been inspected historically;
it is a reused benchmark, not a pristine confirmatory holdout. One trained encoder
seed per recipe cannot establish a causal training advantage.

Encoder inputs are current centered local geometry with a constant atom channel
and center indicator, nearest80 candidates,8 Å crop,5 Å edges, two message-passing
blocks and no extra halo. No motion, history, temperature or time covariates enter
any new probe. Paired observed/relaxed views and the Epi initialization reservoir
belong only to historical pretraining. Predictor inputs are recorded separately.

Predetermined interpretation: improvement in both liquid geometry and held-out
proper predictive scores supports useful local information; either alone is
insufficient. Reduced noise response is useful only alongside retained structure,
rank and predictive information. Descriptor add-back improvement means that the
tested frozen embedding/readout combination did not make all32 descriptor signals
predictively accessible. No universal weighted score or AP-based winner is used.

The current onset labels predominantly measure arrival from existing crystal.
Static Ta/Zr ordered-liquid proxies cannot prove future nuclei. Regional-emergence
prediction and dense relaxed temporal response remain explicitly untested.

[Recipe](../../configs/analysis/encoder_quality_latest_20260926.json) ·
[Definitions](../../docs/metrics/encoder_quality.md) ·
[Reproduction commands](../../docs/encoder_quality.md)
