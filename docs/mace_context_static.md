# Static Al and Zr analysis of the joint MACE checkpoint

The export recipe is `configs/analysis/mace_context_static.json`. It names the
exact `dual_physics` checkpoint selected at epoch 12 and checks its SHA-256.
`static-export` copies only its MACE state into the standard analysis checkpoint
format; the original checkpoint, optimizer and prediction heads remain intact.

```bash
python -m src.research.mace_context.run --config configs/analysis/mace_context_static.json --stage static-export
python -m src.research.mace_context.run --config configs/analysis/mace_context_static.json --stage static-verify
python -m src.analysis.pipeline configs/analysis/static_mace_context_al.yaml
python -m src.analysis.pipeline configs/analysis/static_mace_context_zr.yaml
```

The two analysis recipes retain the full `static.yaml` workflow: seven spherical
clusters, standardization, PCA (99% variance, at most 64 components), L2
normalization, UMAP/t-SNE, connected regimes, representatives and spatial plots.
Each material has its own clustering fit; cluster numbers are not aligned phases.
The six Zr frames and its 10.415006 A source radius come from
`static_multi_material.yaml`. Both materials use the static Al grid overlap of
0.5. Three edge layers are excluded, rather than two: the old Al grid includes
centers with less than the required 17 A complete message support. Original
snapshots and sample caches remain unchanged.

The Al encoder receives physical distances. The Zr transfer keeps the trained Al
element channel and multiplies physical offsets by 9.192189/10.415006. The fixed
factor is shared by every Zr frame, so temporal density differences survive. This
is a transfer test of local geometry, not validation of a chemically trained Zr
encoder. Structural displays retain physical Zr coordinates.

`src/analysis/mace_context_adapter.py` computes the exact two-hop ancestors of
each required atom in batches, then reuses its 256 scalar features across all
overlapping neighborhoods. A readout concatenates the 5-to-7 A tapered inner
average and the tracked center's features. There is no prediction head, projector,
learned feature normalization, added atom or float16 intermediate in this path.
Coordinates are translated in float64 before float32 GPU geometry. There is no
periodic box metadata in these static NPY inputs, so only interior centers are
accepted; box lengths are never guessed from coordinate extrema.

Verification checks exported tensors exactly, compares reuse against independent
complete-halo inference for Al and Zr, and changes batch sizes and center order.
Each full analyzed frame repeats direct-halo checks on six centers. The results
and boundary margins are saved in `technical/context-inference-protocol.json`;
`technical/context-inference-status.json` reports frame/node progress. The saved
inference cache uses float32 and supports subsequent analysis reruns.

Detached execution uses the existing allocation runner and tracked command
workflow. The node51 plans are `configs/mace-context/static-node51-gpu{0,1}.json`.
Each lane exposes one physical GPU as CUDA:0, including to rendering subprocesses.
Exact controller scripts, submission IDs, logs and source snapshots are retained
under `output/mace_context_static/dual-physics-epoch12-20260915/technical/`.
