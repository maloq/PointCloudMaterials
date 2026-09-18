# Static Al analysis of structural MACE/GATr–VICReg

## Active mixed GATr: frozen latest checkpoint

The temporal-backtracking run's latest available optimizer checkpoint was copied
at update 400 while training continued. Its recipe pins the copied `last.pt`
and release manifest hashes. This is an intermediate training state, not a
best-selected checkpoint, so its export has no checkpoint-selection score.

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_backtracking_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_backtracking_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_gatr_backtracking_al.yaml
```

Output: `output/structural_static/gatr-temporal-backtracking-latest-20260918T1936/`.
Inference uses node58's RTX PRO 6000 in `pointnet-torch214`. The
`shared_pretraining_mixed_v8` protocol has grouped auxiliary heads; its actual
exported encoder remains `StructuralGATr(history=False)`. Export validates the
mixed architecture revision and producer hashes, and extracts only `encoder.*`
from the immutable optimizer checkpoint. Group-normalized heads are excluded.

Static verification reconstructs a native Al example from the full release,
using its unchanged Al scale. The current mixed fit uses dynamic observations;
that static example verifies the shared input producer, not training membership.
Since latest states have no saved best-selection features, verification also
loads the exact latest encoder independently, compiles it with the training
policy and compares 64 native dynamic selection observations with the static
adapter. This checks numerical fidelity, not checkpoint quality. Every full
static frame retains its batch replay check. All six Al frames, interior
centers, clustering settings and corrected raw PCA match the earlier analysis.

## Matched Al-only v6 MACE checkpoint

The completed September 18 Al-only MACE run selects its final update 1,465,
with selection score 0.24064353108406067. Use `pointnet-torch214` and the
checkpoint's native cuEquivariance backend on the available H100:

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_mace_v6_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_mace_v6_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_mace_v6_al.yaml
```

Output: `output/structural_static/mace-vicreg-v6-step1465-al-20260918/`.
This retains the v6 GATr analysis's six snapshots, 684,723 interior centers,
seven-cluster settings, corrected float64 raw PCA and complete plot workflow.
Only the selected encoder changes. Each MACE observation reconstructs the same
full support and fixed Al scaling as training; its directed 5-model-unit edges
are generated on the unpadded, normalized local coordinates. Packed graphs have
no edges between observations. The observation's taper enters both edges and
node features exactly as in training, so atom states are not reused across
different centers. Multiscale pooling uses the trained 0–3, 5–7 and 15–17 model
unit ranges. Protected geometry/tensor operations remain FP32 and the trained
scalar operations use BF16. Only the raw z128 encoder state is exported.
The MACE encoder uses the training full-graph compiler with dynamic shapes and
preserved precision casts. Eager execution failed the saved-selection check
(maximum absolute difference about 2.52e-5), so it is not used for this report.

Verification matches every native input tensor (including packed edges),
selected checkpoint tensors, single-observation output, batch/reorder replay
and all 480 saved compiled selection states. CuEquivariance CUDA reductions
are not bitwise repeatable: inputs/weights match exactly, while MACE output
checks use rtol 2e-5 and atol 2e-6 and record repeated-native error separately.
These frames overlap training;
the clusters are descriptive groups and their IDs are specific to each fit.

## Newest Al-only v6 checkpoint

The completed September 18 Al-only run selects update 1,216 out of 1,465,
with selection score 0.22607703506946564. Its normalization differs from the
earlier broad-material run, so the selection scores do not rank the two models.
Use conda `pointnet-torch214` for this analysis:

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_v6_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_v6_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_gatr_v6_al.yaml
```

Output: `output/structural_static/gatr-vicreg-v6-step1216-al-20260918/`, with its
data on the configured WORK analysis root. The same six snapshots, 684,723
centers and analysis settings are retained. The model has the explicit
`structural_v6_conditioned_heads` architecture and uses its trained selective
BF16 policy: protected geometry, residual streams and exported z128 stay FP32;
scalar maps use the model's compensated arithmetic. Head normalization buffers
are excluded because this analysis uses only the encoder. Eager inference uses
the training contraction order and is verified against all 480 saved states
from the compiled best-checkpoint selection pass. No training is resumed.

The raw PCA diagnostic uses float64 full SVD. This avoids cancellation in
float32 covariance calculations when small structural differences sit on large
channel means. Latent summary statistics also accumulate in float64. The
standardized clustering calculation is unchanged. Pre-correction diagnostics
are preserved in the new run's `technical/pre-correction-diagnostics.tar.gz`.

The previous model and its archived outputs keep their own versioned source
and metric definitions. The protocol below records that earlier analysis.

## Original September 17 checkpoint

The RTX PRO 6000 run completed 4,096 updates. Its selected encoder is update
3,072, with source-balanced selection score 0.4045023210346699. The export
recipe pins the source SHA-256 and verifies every tensor against `best.pt`.
The source checkpoint and training state are preserved.

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_gatr_al.yaml
```

Run in conda `pointnet`. Export refuses to overwrite an existing export.
Verification and analysis use a GPU. The standard pipeline caches inference
for reruns and publishes its gallery, plots and metric tables at
`output/structural_static/gatr-vicreg-step3072-al-20260918/`.

The six Al inherent snapshots are 166, 170, 174, 175, 177 and 240 ps.
The recipe preserves `static.yaml` analysis settings: seven spherical clusters,
standardization, PCA retaining 99% variance up to 64 components, L2 normalization,
t-SNE/UMAP, connected-regime analysis, representative structures, PTM/CNA and
spatial figures. It reuses the established three-edge-layer context grid with
684,723 centers. This excludes incomplete neighborhoods at nonperiodic borders.
No box is inferred from coordinate extrema.

Inference uses the trained raw 128-channel center state. The nearest-160 samples
in the standard loader supply the sampling grid and normalized representative
displays; spatial maps retain physical center coordinates. Encoder
inference independently extracts every source atom within the trained physical
support (about 16.87 Angstrom for Al). Offsets undergo the same float64
subtraction, float32 storage and fixed scaling as training. The Al radius is
9.121389139452193 Angstrom, with reference radius 9.192189; weights taper from
15 to 17 model units. Species is the trained Al channel, time is zero, and the
trained log material scale is supplied. No scale is refitted on analysis data.
FP32 inference disables TF32. Projection and physical/TDA heads are excluded.

Verification checks exported tensors exactly, matches all seven native training
input tensors and the encoder output on a real prepared static example, and
checks batch/padding/order independence on the first analysis snapshot. Each
full analyzed frame repeats the batch-size check on six centers. Receipts are
`technical/static-verification.json` and `technical/structural-inference-protocol.json`;
`technical/structural-inference-status.json` reports extraction progress.

These static snapshots contributed to the broad training release. Results are
descriptive representation diagnostics, not held-out accuracy or independent
source evidence. Cluster IDs are learned groups, not assigned thermodynamic
phases. See the [structural-state glossary](research_glossary.md#shared-structural-pretraining)
and [training protocol](../experiments/structural_pretraining_20260917/README.md).
