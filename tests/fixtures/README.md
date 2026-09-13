# Regression inputs

These files exercise maintained implementation without keeping retired experiment
recipes in the active `configs/` tree.

- `simulation/`: fifteen historical Al protocol configs and their dependencies,
  copied from the [verified config archive](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/configs/simulation/atomistic/al/).
  Only `configs/simulation/atomistic/al/` references changed to fixture paths.
- `vicreg_geo_frame_transformer_v2.yaml` and
  `vicreg_geo_frame_multiscale_factor_vae_midpoint_no_noise.yaml`: fully composed
  Hydra configurations captured before retirement, without changing scientific
  values. They test encoder construction and batch-independent projector export.
- `shooting_multiscale.yaml`: the existing distributional shooting test input.

These are test data, not maintained simulation or training launch recipes.
