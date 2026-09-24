# Input-noise responsiveness

[Encoder handbook](README.md) · [Definitions](../metrics/embedding_noise.md) ·
[0.75 ps trajectory analysis](embedding_dynamics.md)

The [combined results](../../output/encoder_research/noise-lag075-20260924/RESULTS.md)
add input-noise response to the matched 0.75 ps table, including Geoformer encoder
and projector exports. All original trajectory metrics remain unchanged.

The primary new number is **normalized noise RMS at sigma = 0.01 Angstrom per
coordinate**. A second number compares that displacement with the same model's
natural 0.75 ps motion at the sampled origins. Values below one mean that the
noise perturbation moves the embedding less than the sampled natural movement.
This ratio is not a causal attribution of how much MD motion is noise.

We also export response curves over four noise amplitudes, p95 responses,
sensitivity per Angstrom, repeated-input variation, cached-input replay agreement
and noncrystalline subsets. This makes unusually sensitive tails and numerical
execution effects visible alongside mean responsiveness.

Run with the existing native producers and cached MD observations:

```bash
conda activate pointnet-torch214
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m src.research.trajectory_stability.noise \
  --config configs/analysis/encoder_noise_20260924.json
```

`--stage prepare`, `infer`, `report` separate input construction, checkpoint /
descriptor inference and metric export. Use a fresh output for a new completed
report. Incomplete inference resumes only when its task and feature checksums
match. The [recipe](../../configs/analysis/encoder_noise_20260924.json) declares
the source report, noise levels, sampling seed, number of observations and draws.
No new training or molecular-dynamics simulation is required.
# Local-spacing normalization in the new experiments

The September 24 robust-onset queue specifies input perturbations as expected
**3D RMS displacement divided by the clean mean center-to-12-nearest-neighbor
distance**. The primary setting is 0.5%; evaluations also use 0.1%, 1%, and 3%.
Tables report the realized percentage as well as Å. This input percentage is
separate from normalized embedding Noise RMS, which describes the output response.
See the [exact definitions](../metrics/robust_onset.md) and
[experiment protocol](../../experiments/robust_onset_20260924/README.md).
The earlier noise table retains its original Å-based inputs and metric contract.
Recomputing this input scale on its 1,280 perturbations gives a mean local
12-neighbor distance of **2.947 Å**: its primary per-coordinate sigma of 0.01 Å
was **0.588% realized 3D RMS displacement relative to local spacing**.
