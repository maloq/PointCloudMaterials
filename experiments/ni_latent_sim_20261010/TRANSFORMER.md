# Transformer versus MLP motion transition

Question: does causal attention over the same four latent states improve
physical path responses and autonomous forecasts over the current MLP?

Only the transition architecture changes. The frozen unrelaxed Small eSEN,
280-to-16 compressor, 16-to-280 Gaussian decoder, four input observations,
5-fs neural step, 100-fs history refresh, stochastic noise scaling and all loss
weights remain the dense-path protocol. The model receives saved Q/V through
the same frozen feature construction. It remains autoregressive: its predicted
current code feeds the next step without inserting physical intermediate states.

The transformer projects each of four 16-dimensional tokens to width64, adds
learned ordered-slot embeddings, and applies two pre-LayerNorm causal attention
blocks. Each block has four heads, a width192 GELU feedforward network and
residual connections. A final LayerNorm and linear head on the current token
predict sixteen drift and sixteen noise-amplitude values. Dropout is zero;
stochasticity comes from the same explicit latent Gaussian innovations. Slot
order substitutes for ordering implicit in MLP concatenation; no timestamps,
simulation ages, temperature, species or substep-phase channels are introduced.

The transition has87,008 parameters versus90,656 for the MLP; complete trainable
models have442,912 versus446,560. Paired seeds start the shared compressor and
decoder identically. All fits use the same training-row and noise-seed streams.
This compares approximately matched capacity with different inductive biases,
not an increased history or a larger encoder.

Train two transformer seeds,20261011/20261012, for3200updates each, batch and
microbatch256, BF16 long recurrence, FP32 physical response loss, AdamW maximum
LR0.001,200-update warmup and cosine decay. Whole-path value and initial-Q/V
derivative targets cover5,10,...,100fs. The300 physical teacher groups and the
14,400-row native motion cohort are checksum-bound reused inputs, with no new
MD, filtering, source split, broad spatial replay or target generation.

The reference models are the matched `path_response` MLP seeds from the dense
study. Both architectures use native-future NLL at0.1/0.5/1ps for training and
checkpoint selection. Evaluation keeps all twenty physical-response times,
free rollouts through6ps, per-regime proper scores and stability metrics, and
the unchanged32/64-atom nucleus tasks at1/3/6ps leads. AP is diagnostic only.
The final architecture bundle retains original tables and their hashes, with
paired seed-wise transformer-minus-MLP differences. Two-seed ranges in figures
are descriptive, not confidence intervals.

Recipe: `configs/ni_latent_sim/transformer_path_response_20261011.json`.
Entry point: `python -m src.research.ni_latent_sim.workflow submit --config
configs/ni_latent_sim/transformer_path_response_20261011.json`.
Each detached lane waits for its corresponding original MLP lane to complete.
Afterward both transformer seeds can run in parallel on the same two GPUs.
This preserves the earlier endpoint/path-values/path-response comparison.
