# H200 handoff: packed causal MACE study

Use the **runtime v2** bundle `causal-mace-runtime-20260916.tar.zst` and its SHA256
file. It supersedes the original H200 bundle for new experiments. The original
archive remains a reproducible snapshot. The H100 pilot continues independently;
the H100 also runs a longer width-16 cohort in the isolated runtime checkout.

The bundle contains code, metric contracts, tests, environment requirements, all
six H200 recipes, the existing 150-source/1,800-window physical cache, and the
optional raw-data transfer inventory. No raw trajectories or pretrained weights
are needed for this fixed-data study. Do not rerun `causal-prepare`. Each H200
recipe uses 5,000 updates, batch_sources 8, packed GPU-resident inputs, 2,000 probe
updates and the same fixed targets/splits. Width 16 versus 32 changes tensor
capacity while retaining the exported 128-channel state.

Verify and enter the bundle before producing new files inside it:

```bash
sha256sum -c causal-mace-runtime-20260916.tar.zst.sha256
tar --zstd -xf causal-mace-runtime-20260916.tar.zst
cd causal-mace-runtime-20260916
PYTHONDONTWRITEBYTECODE=1 python scripts/project.py verify-bundle .
```

Reuse conda `pointnet`, or create a compatible environment outside the bundle with
`environments/requirements-gpu.txt`. The tested stack is Python 3.12, PyTorch
2.11.0+cu128, mace-torch 0.3.16, e3nn 0.4.4, NumPy 1.26.4, SciPy 1.17.1. Check the
local CUDA driver with an actual forward/backward; no silent precision changes.
The portable profile defaults to CPU: pass `--device cuda:0`. Initialize an import
Git snapshot for tracked execution (the bundle has no `.git`), excluding data and
outputs with its `.gitignore`.

## Prompt to give the receiving agent

Implement no architecture changes before the matched experiment. Run the existing
optimized causal MACE implementation in this checkout on this H200.

1. Read AGENTS.md, scripts/README.md, docs/mace_causal.md,
   docs/mace_causal_runtime.md, docs/metrics/mace_causal.md and this handoff. Verify
   the bundle, available CPU/RAM/VRAM and allocation deadline. Validate the relocated
   cache with `mace_causal.data.load`: 90/30/30 whole sources and 1,080/360/360 windows.
   Run the causal encoder, comparison and layout tests and the actual-graph CUDA
   benchmark. Set a fresh benchmark output in a copied benchmark recipe. Existing
   H100 timings are contended observations, not H200 throughput predictions.

2. Packing, GPU residency and vectorized evaluation are already implemented.
   Keep `runtime.encoding=packed`, `runtime.residency=device`, runtime batch 8 and
   statistical batch_sources 8 for the scientific cohort. Confirm finite outputs
   and gradient agreement at both widths. Do not change cached manifests, source
   splits, normalization or hazard reduction. Retain FP32; AMP/TF32 would need a
   separate validated numerical study. Choose one or multiple independent fits
   from measured aggregate throughput and memory; keep activation headroom.

3. Run C (current geometry+motion), D (observed atom history) and repeated_anchor
   (D trained with repeated current frames) for widths 16 and 32 and seeds
   20260916, 20260917, 20260918, using
   `configs/mace_causal/h200/width{16,32}-seed{SEED}.json`. Keep physical horizons
   0.75/3/9 ps, history 2.25 ps, radius/depth, deterministic heads and loss fixed.
   Prioritize a complete width-32 seed-20260916 triplet, then width-16 seed-20260916,
   then replication. The H100 width-16 triplet is useful context, but keep this
   study's own paired recipes/results. Report missing seeds if time expires.
   Validation selects checkpoints; test outcomes never select settings.

4. For every encoder run `causal-probe --probe-modes linear nonlinear`. Only for
   D, additionally run `causal-probe --probe-modes state_constant state_history`.
   That pair asks whether the exported D state discarded useful observed-history
   information; its absence for C/repeated_anchor is predeclared. Use the existing
   comparison-width16/32 configs, which require all common readouts and D's extra
   pair. Do not infer state sufficiency from lack of a diagnostic improvement.

5. Use maintained tracking/allocation runners to detach the queue. Set this
   server's actual node/job/deadline and preserve PIDs, logs, strict-resume state,
   normalization and source snapshots. Do not reuse node53/991900 assumptions.
   `--resume` requires the exact saved config, code and cache; raising the step
   budget or changing runtime in place is not exact resume. Read learning curves
   to assess convergence rather than assuming 5,000 steps is enough.

6. Compare decoded present/future errors with persistence, with equal weight for
   the six physical target blocks, within-low-order results, and whole-source
   paired uncertainty. Report J at its exact declared lag/population, separately
   from information quality. The existing test segment has only one distinct
   onset center/source, so hazard results do not establish onset/timing skill.
   For a direct width comparison, add a documented paired collector that permits
   exactly the width difference and verifies targets, data, seeds, budgets and
   runtime; do not disable the existing matched-recipe guard.

7. Return output/mace_causal/h200-* including technical checkpoints and paired
   physical predictions, source snapshots, tables/METRICS.md, plots and a concise
   report. Return any new tested source/config changes. Preserve failed/restart
   evidence. If more data is available, treat the fixed regular-grid plan in the
   runtime handoff as a separate cache/protocol implementation; do not combine
   data expansion and width/history ablations in one comparison.

Example commands:

```bash
python -m src.research.mace_velocity causal-train \
  --config configs/mace_causal/h200/width32-seed20260916.json --variant D --device cuda:0
python -m src.research.mace_velocity causal-probe \
  --config configs/mace_causal/h200/width32-seed20260916.json --variant D \
  --probe-modes linear nonlinear --device cuda:0
python -m src.research.mace_velocity causal-probe \
  --config configs/mace_causal/h200/width32-seed20260916.json --variant D \
  --probe-modes state_constant state_history --device cuda:0
python -m src.research.mace_causal_comparison \
  --config configs/mace_causal/h200/comparison-width32.json
```

Large VRAM is useful for input residency, wider tensor activations, larger true
batches, or several independent fits. It does not require increasing the model
size. The [runtime handoff](mace_causal_runtime.md) includes a separate 101.5 GB raw
inventory and 36,000-window candidate grid if more temporal coverage is desired.
No external server transfer has been executed.
