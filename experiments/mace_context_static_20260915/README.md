# Joint MACE embeddings on static Al and Zr

Question: what local structural regimes does the selected smooth-inner + center
MACE embedding distinguish in the configured static Al and Zr snapshots?

Analyze the `dual_physics` checkpoint selected at epoch 12. Its 512 channels are
the concatenation of a smooth inner average and the tracked atom's feature from
one jointly trained backbone. Use the standard static analysis recipe and fit
seven clusters separately for each material. This is descriptive static analysis,
not a new test of forecasting or TDA generalization. Zr is an out-of-domain
geometry transfer using the fixed Al element channel and a fixed distance factor
9.192189/10.415006. Cluster IDs across materials are not aligned phase labels.

Al frames: 166, 170, 174, 175, 177, 240 ps. Zr frames: 160, 200, 240, 280, 310,
1560 ps. Keep a full two-hop halo and pool only the 5-to-7 A smooth inner region.
The same regular-grid overlap 0.5 and three-edge-layer exclusion apply to both
materials. The extra excluded layer is necessary for complete context at static
boundaries. See [operating protocol](../../docs/mace_context_static.md).

```bash
python -m src.analysis.pipeline configs/analysis/static_mace_context_al.yaml
python -m src.analysis.pipeline configs/analysis/static_mace_context_zr.yaml
```

Results: `output/mace_context_static/dual-physics-epoch12-al-20260915/` and
`output/mace_context_static/dual-physics-epoch12-zr-20260915/`.
Checkpoint export and validation:
`output/mace_context_static/dual-physics-epoch12-20260915/technical/`.

Pre-launch verification: all exported tensors match the selected checkpoint
exactly. Full-frame reuse versus individual complete-halo evaluation has relative
squared error below 1e-11 on both materials, including reordered centers and
different graph batch sizes. Full analysis findings are pending completion.
