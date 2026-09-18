# Shared-pretraining efficiency priorities

The scientific architecture/precision comparison is in the
[mixed-precision proposal](../experiments/shared_pretraining_20260918/MIXED_PRECISION_PROPOSAL.md).


1. **MACE graph/input preparation.** Recent traces spend roughly 30% of update-plus-
   input-wait time waiting for prepared batches, versus negligible GATr wait.
   Cache normalized neighbor graphs or prepare them concurrently with bounded
   worker queues. Preserve sampled identities and full-batch loss statistics.
   Eliminating this wait has a larger potential effect than MACE's measured 1.07x
   BF16 compute speedup. The timing fraction is observational, not a promised gain.
2. **Use more available VRAM for microbatches.** The assigned GPUs have about
   94–96 GiB, while current BF16 peaks are about 19–21 GiB. Profile GATr encoder
   microbatches 416/832/1024 after choosing safe precision, with a separate memory
   preflight. Keep the statistical batch fixed at 1,024 pairs. Full activation
   retention could remove the extra encoder forward only if it fits; do not
   assume it will fit from one microbatch's memory measurement.
3. **Reduce padding without changing the batch.** Bucket observations by atom
   count within each selected statistical batch, and restore original pair order
   before computing VICReg. This also limits the number of compiled shapes.
4. **Compile measured hotspots.** The installed GATr explicitly disables compilation
   around its SDPA wrapper, so wrapping the whole model in `torch.compile` is not
   sufficient evidence of acceleration. Start with scalar MLPs and repeated tensor
   operations, then address the shape/dispatch graph break in a repository-owned
   adapter. Include compilation time and warmup in the amortization calculation.
5. **Verify attention kernel dispatch.** Current GATr already calls PyTorch SDPA.
   Actual fused-kernel eligibility depends on dtype, shape and the smooth-support
   mask. Profile the selected backend and preserve the mask semantics. The
   [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
   and [performance tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
   describe backend selection and measurement. Installing another attention package
   does not by itself establish a speedup.

The measured current precision baseline is
[1.07x MACE/H100 and 1.59x GATr/RTX6000](../output/shared_pretraining/restart-b1024-lr002-bf16-20260918/PRECISION.md)
on a repeated real-data batch. All additional speedups above are candidates,
not measured results. Do not combine microbatch, precision, architecture and
objective changes into one unidentifiable comparison.
