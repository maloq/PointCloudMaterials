# Symmetric context: MACE and GATr

25 structured query slots; fixed source split; one seed; predictor heads trained from scratch with frozen backbones.
GATr retains the requested original 16.87 A local support; MACE retains 7.94 A. Compare shared physical/event targets, not raw cross-encoder latent errors.

| Fit | State | Selected update | Brier | 12 ps AP | Physical MSE |
|---|---|---:|---:|---:|---:|
| mace-direct-symmetric-E36 | complete | 6013 | 0.09625 | 0.66123 | 0.83677 |
| mace-ar_mse-symmetric-E36 | complete | 6013 | 0.09416 | 0.67183 | 0.85064 |
| mace-mixture-symmetric-E36 | complete | 6013 | 0.09353 | 0.66632 | 0.86878 |
| mace-diffusion-symmetric-E36 | complete | 14603 | 0.09819 | 0.60887 | 0.85436 |
| gatr-direct-symmetric-E36 | complete | 6013 | 0.09343 | 0.63687 | 0.83734 |
| gatr-ar_mse-symmetric-E36 | complete | 6013 | 0.09285 | 0.64126 | 0.85296 |
| gatr-mixture-symmetric-E36 | complete | 7731 | 0.09216 | 0.65211 | 0.86671 |
| gatr-diffusion-symmetric-E36 | complete | 14603 | 0.09964 | 0.57176 | 0.85552 |

All forecasts remain open-loop to 96 ps. Short-horizon timing, misses, calibration and sampled-center spatial scores are in each fit’s metrics.json. 
The stencil is fixed in the simulation-box frame. Cubic rotations permute slots; arbitrary rotations require transporting the query frame. Real atom assignments are approximate and their exact offsets are inputs.
