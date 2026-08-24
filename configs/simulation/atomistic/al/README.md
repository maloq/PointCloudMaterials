# Aluminium atomistic workflows

- `liquid_source_*`: immutable 500 K liquid preparation.
- `homogeneous_*`: physical homogeneous-crystallization workload.
- `campaign_*`: checkpointing, online-event, and analysis execution policy.
- `phase_context_*` and `phase_transition_*`: 70,304-atom phase/coexistence workflows.
- `transition_campaign_*`: persistent one-model-per-GPU queued execution for the
  70,304-atom temperature/replica grid, exact chunked MTK resume, and deferred
  CPU PTM/RDF analysis.
- `jumpy_ffs_*`: optional rare-event workflow.
- `potential_*`: scientific validation, performance measurement, and model selection.
- `active_campaign_16384_mpa.yaml`: the currently running natural-endpoint replica.
- `producer_compatibility.json`: audited producer-code checkpoint migrations.

## H100 runtime selection

The fixed 16,384-atom MPA workload was measured on one 95.8 GiB H100 NVL with
MACE 0.3.16, PyTorch 2.11, FP32 forces+stress, and TF32 explicitly disabled.
The same immutable 500 K source coordinates, MTK settings, velocity seed, and
1.2M-edge shape were used unless noted.

| Kernels | Compile mode | Skin / edge pad | Steps/s | First evaluation | Peak reserved | Result |
|---|---|---:|---:|---:|---:|---|
| CuEq | `reduce-overhead` | 0.3 A / 1.2M | 5.51 | 148.7 s | 27.54 GiB | pass |
| CuEq | `max-autotune` | 0.3 A / 1.2M | 5.11 | 256.7 s | 22.35 GiB | pass, slower |
| CuEq | `max-autotune-no-cudagraphs` | 0.3 A / 1.2M | 6.93 | 204.2 s | 22.25 GiB | pass |
| OpenEquivariance | `max-autotune-no-cudagraphs` | 0.3 A / 1.2M | 2.37 | 238.3 s | 46.01 GiB | pass, slower |
| CuEq + OpenEquivariance | `max-autotune-no-cudagraphs` | 0.3 A / 1.2M | 4.76 | 273.2 s | 20.97 GiB | pass, slower |
| CuEq | `reduce-overhead` | 0.5 A / 1.2M | 6.14 | not isolated | 27.54 GiB | pass |
| CuEq | `max-autotune-no-cudagraphs` | 0.5 A / 1.2M | 7.50 | 142.4 s, not isolated | 22.25 GiB | selected |
| CuEq | `reduce-overhead` | 0.5 A / 1.1M | 6.30 | not isolated | 25.75 GiB | tight experimental pad |

OpenEquivariance and hybrid `reduce-overhead` and `max-autotune` all fail on
their first force/stress backward pass in PyTorch's CUDA-graph pool alias check.
The no-CUDA-graph measurements above are explicit comparison modes, not silent
fallbacks. The production config keeps the 1.2M pad: the observed 0.5 A graph
reached about 1.060M edges, so the faster 1.1M result has too little margin for
a long NPT trajectory.

Re-run the auditable sweep with:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_potential_performance \
  --config configs/simulation/atomistic/al/potential_runtime_variants.yaml
```

Run the tuned nine-replica workload with one persistent model per H100:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_homogeneous_campaign \
  run --config configs/simulation/atomistic/al/campaign_16384_mpa_110ps_multiseed_h100.yaml \
  --devices 0,1

conda run -n pointnet python -m src.data_utils.synthetic.atomistic_homogeneous_campaign \
  analyze --config configs/simulation/atomistic/al/campaign_16384_mpa_110ps_multiseed_h100.yaml \
  --workers 4
```

The 70,304-atom Slurm wrappers run deferred analysis and checkpoint
visualization automatically after the final MD segment succeeds. To refresh
the images manually from the latest hash-verified checkpoint without rerunning
MACE:

```bash
conda run -n pointnet python scripts/plot_homogeneous_checkpoint.py \
  --campaign-config configs/simulation/atomistic/al/campaign_70304_mpa_130ps_source12345_seed35803.yaml \
  --include-structure-slices \
  --step-stamped
```

This writes a live dashboard and PTM structure-slice image under the campaign's
`visualizations/` directory and preserves step-stamped copies for audit.

For future homogeneous runs that need three saved coordinate/cell states per ps,
add the following top-level field to the `homogeneous_*.yaml` file while keeping
`sample_interval: 1000` as the one-ps nucleation-event definition:

```yaml
sample_interval: 1000
trajectory_samples_per_ps: 3
```

At the configured 1 fs timestep this writes the exact rational schedule
`0, 333, 667, 1000, ...` steps. The field is part of new campaign identity and
must not be added retroactively to completed or resumable campaigns.

Including each replica's configured 5,000-step equilibration and 110,000-step
measurement, the measured 7.50 steps/s implies about 38.3 H100-hours for all
nine replicas. The dynamic two-GPU queue has a five-replica critical path of
about 21.3 hours before checkpoint/I/O overhead, versus roughly 29.0 hours with
the old 5.51 steps/s runtime.

Generate the prepared 70,304-atom interface once if it is not already present.
The queued campaign content-hashes this source and will refuse to initialize
from missing or subsequently replaced files:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_generator \
  --config configs/simulation/atomistic/al/phase_context_70304_mpa.yaml
```

Then run the direct-coexistence campaign on one or two GPUs and analyze the
atomically committed raw trajectories without loading MACE:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_transition_campaign \
  run --config configs/simulation/atomistic/al/transition_campaign_70304_mpa.yaml \
  --devices 0,1

conda run -n pointnet python -m src.data_utils.synthetic.atomistic_transition_campaign \
  analyze --config configs/simulation/atomistic/al/transition_campaign_70304_mpa.yaml \
  --workers 4
```

The 70,304-atom wrapper intentionally uses uncompiled CuEq today. The completed
130 ps trajectories reached 4,115,156 directed edges with a 0.3 A skin.
Compiled execution still requires a target-specific atom/edge pad plus margin;
benchmark that full fixed shape on an H100 rather than reusing the 16k padding
ratio.

The archived 130 ps coordinates give the following full-size graph envelopes.
The pads retain approximately 10% margin and are benchmark controls, not
automatic allocation fallbacks.

| Skin | Source-frame edges | Archived maximum | Candidate pad |
|---:|---:|---:|---:|
| 0.3 A | 4,053,538 | 4,115,156 including runtime rebuilds | 4,600,000 |
| 0.4 A | 4,201,168 | 4,341,090 saved-frame scan | 4,800,000 |
| 0.5 A | 4,358,594 | 4,600,994 saved-frame scan | 5,100,000 |

Run each candidate in a fresh H100 process, smallest first. Each report uses
uncompiled CuEq as its explicit numerical reference because constructing a
70k e3nn reference can exhaust GPU memory before timing begins. There is no
backend or edge-budget fallback. The same report also times an uncompiled CuEq
variant, so `speedup_vs_baseline` is measured on the identical source and MD
workload.

```bash
for skin in 03 04 05; do
  PYTORCH_ALLOC_CONF=expandable_segments:True NVIDIA_TF32_OVERRIDE=0 \
    conda run -n pointnet python \
      -m src.data_utils.synthetic.atomistic_potential_performance \
      --config \
      configs/simulation/atomistic/al/potential_runtime_70304_cueq_nocudagraphs_skin${skin}.yaml \
    || break
done
```

Stop after any OOM or failed parity gate. Only test CUDA-graph
`reduce-overhead` in a separate process at the winning skin if its measured
peak leaves enough H100 memory.

Fixed-cell interface preparation can opt into force-only Langevin steps with
`potential.nvt_md_property_mode: forces`. Stress is then evaluated only on
thermodynamic trace frames. This changes FP32 reduction order slightly, is
restricted to exploratory use, and must use a new config/output identity
rather than extending an existing trajectory.

## Experimental BF16 comparison

The BF16 experiment uses CUDA BF16 autocast only inside the second and largest
interaction block of the uncompiled two-interaction CuEq model. Geometry,
radial features, the first interaction, products, readouts, reductions, the
checkpoint, and graph inputs remain FP32. It is a separate numerical
trajectory, never a continuation of an FP32 checkpoint. The two campaigns
reuse the completed replicas' source and velocity seeds and save three
coordinate/cell states per ps.

From a Slurm login host, submit the full-size FP32/BF16 force-and-stress parity
gate, both four-allocation replica chains, and the final matched comparison with:

```bash
scripts/submit_al_homogeneous_70304_mpa_bf16.sh
```

The parity gate records force, stress, energy, time, and peak H100 memory under
`output/synthetic_data/al_mpa_70304_bf16_validation_20260724/`. The campaigns
are released only when that gate succeeds.
