# GeoFrame factor-VAE sweep

Question: compare the fixed factor-VAE settings and GeoFrame controls implemented
by this research queue. The checkpoint selection, sweep and Hydra configuration
names remain the original experiment's explicit assumptions.

```bash
conda run -n pointnet python experiments/factor_vae_20260901/run_queue.py --help
```

The queue's arguments select its output root and execution settings. It records
run specifications, logs and results under that output root. The fixed V1 source
checkpoint is selected from
`output/detached/vicreg_geoframe_corrected_h100_20260829_220250`; inspect the recipe
and its `--help` before reproducing the sweep. Configurations remain under
`configs/` for Hydra composition. Existing run reports remain authoritative;
no experiment was rerun and no new conclusion is claimed by this relocation.

The maintained generic plan runner is `scripts/run_experiments.py`. Use a plan
for a new sweep that fits its existing training/stage model; keep this recipe to
reproduce the original checkpoint chaining and factor settings.
