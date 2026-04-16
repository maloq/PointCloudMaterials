# Successor-VICReg

This folder adds a minimal temporal Successor-VICReg method on top of the existing invariant VICReg encoder.

## What stays the same

- The encoder backbone is built through `src.models.encoders.factory.build_encoder`.
- The invariant latent `z_inv` still comes from the existing VICReg invariant path in `src/training_methods/contrastive_learning/vicreg.py`.
- The VICReg loss and its augmentations are reused without changing the original static or temporal VICReg codepaths.
- The original `contrastive_learning` and `temporal_ssl` methods remain runnable unchanged.

## What is new

- A separate Lightning module: `SuccessorVICRegModule`.
- A predictor head `g_phi : R^d -> R^d`.
- An optional EMA teacher encoder for future targets.
- Offline export of both:
  - raw invariant embeddings `z_inv`
  - predicted successor embeddings `hat_S`
- Offline clustering comparison for:
  - raw `z_inv`
  - successor embeddings `hat_S`
  - optional `concat[z_inv, hat_S]`

## Exact implementation choices

- The current time `t` is the first frame in each temporal window.
  This keeps the successor target aligned with frames `t+1, ..., t+H`.
- VICReg is applied to the current frame only.
  The existing VICReg loss implementation is reused directly on `x_i^t`.
- The successor target uses the mean teacher latent over the full fixed-size local neighborhood at each target frame.
- The local neighborhood is exactly the dataset neighborhood of `N=data.num_points` atoms.
  In the current temporal LAMMPS pipeline this neighborhood includes the center atom.
- If `data.model_points < data.num_points`, the online encoder still receives the cropped `model_points` input used by the base repo, while the successor target averages over the full sampled `num_points` neighborhood.
- If `successor_use_ema_teacher=true`, the teacher is an EMA copy of the encoder.
  The EMA decay is taken from `encoder.kwargs.ema_decay` when present, otherwise `0.996`.
- If `successor_use_ema_teacher=false`, teacher latents come from stop-gradient online encoder outputs.
- For exact future neighborhood means, atom latents that are missing from the current batch are fetched from a sequence-length-1 lookup dataset and encoded on demand.

## Data split policy

- Successor-VICReg uses a contiguous temporal split in its dedicated datamodule.
- `data.train_ratio` defines the leading train block.
- Optional `data.val_ratio` defines the next validation block.
- The remaining tail becomes the test block.
- This avoids random frame shuffling across temporal train/val/test splits.

## Export + analysis

- `export_successor_embeddings.py` exports `z_inv` and `hat_S` for all selected atom-time pairs.
- `analyze_successor_embeddings.py` clusters the exported embeddings and reports a future-oriented metric:
  - mean within-cluster variance of the discounted future local field
- `visualize_successor_labels.py` colors atoms by cluster/state label on selected frames from a `(frames, atoms)` label grid.

## Caveats

- If the export artifact contains only a subset of atoms, offline analysis recomputes missing neighborhood-atom latents from the checkpoint on demand.
- Training is exact but more expensive than plain VICReg because future neighborhood-atom latents may require additional neighborhood fetches.
