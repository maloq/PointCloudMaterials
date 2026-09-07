"""Matched spatiotemporal checkpoint representation measurements."""
import numpy as np
from omegaconf import OmegaConf
import torch
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule

def representation_metrics(values):
    anchor, spatial, future = (values[:, i].double() for i in range(3))
    centered = anchor - anchor.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / (len(anchor) - 1)).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities[probabilities > 0]
    cosine_drift = 1 - torch.nn.functional.cosine_similarity(centered, future - anchor.mean(0), dim=1)
    return dict(
        temporal_mse=(anchor - future).square().mean().item(),
        temporal_relative_mse=((anchor - future).square().mean() / anchor.var(0).mean()).item(),
        spatial_relative_mse=((anchor - spatial).square().mean() / anchor.var(0).mean()).item(),
        centered_temporal_cosine_distance=cosine_drift.mean().item(),
        anchor_std=anchor.std(0).mean().item(),
        effective_rank=torch.exp(-(positive * positive.log()).sum()).item(),
        participation_rank=(1 / probabilities.square().sum()).item(),
        largest_pc_fraction=probabilities[-1].item(),
    )



@torch.inference_mode()
def extract(checkpoint, manifest, root):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["hyper_parameters"])
    cfg.compile_encoder = False
    model = VICRegModule(cfg)
    # These two repository checkpoints were saved with torch.compile's wrapper.
    state = {key.replace("encoder._orig_mod.", "encoder."): value for key, value in payload["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    output = {}
    for material in ("Al", "Mg", "Ta"):
        enc, proj, lag_arrays = [], [], []
        for shard in manifest["shards"]:
            if shard["split"] != "val" or shard["material"] != material:
                continue
            views = np.load(root / shard["views"], mmap_mode="r")
            pairs = np.load(root / shard["pairs"], mmap_mode="r")
            count = 768 if material == "Ta" else 128
            selected = np.linspace(0, len(views) - 1, count, dtype=np.int64)
            lag_arrays.append(pairs[selected, 3])
            for start in range(0, count, 128):
                points = torch.from_numpy(views[selected[start:start + 128]].copy()).cuda()
                encoded = model.encoder_io.encode(points.flatten(0, 1))
                features = model._shared_invariant(encoded.invariant, encoded.equivariant)
                projected = model.vicreg.project_features(features)
                enc.append(features.reshape(-1, 3, 128).cpu())
                proj.append(projected.reshape(-1, 3, 128).cpu())
        output[material] = dict(encoder=torch.cat(enc), projector=torch.cat(proj), lags=np.concatenate(lag_arrays))
    del model
    torch.cuda.empty_cache()
    return output



