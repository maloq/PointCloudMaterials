#!/usr/bin/env python3
"""Compare original/fine-tuned projector clustering on identical static samples."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import adjusted_rand_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.vis_tools.latent_analysis_vis import compute_kmeans_labels


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args(argv).root
    checkpoints = json.loads((root / "temporal_comparison.json").read_text())["checkpoints"]
    outputs = {}
    torch.set_float32_matmul_precision("high")
    for name, checkpoint in checkpoints.items():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(payload["hyper_parameters"])
        cfg.compile_encoder = False
        model = VICRegModule(cfg)
        model.load_state_dict({k.replace("encoder._orig_mod.", "encoder."): v for k, v in payload["state_dict"].items()}, strict=True)
        model.cuda().eval()
        for material in ("Al", "Mg", "Ta"):
            print(name, material, flush=True)
            cache = root / "data_cache" / material
            metadata = json.loads((cache / "metadata.json").read_text())
            embeddings = []
            for shard in metadata["shards"]:
                points = np.load(cache / shard["samples_path"], mmap_mode="r")
                for start in range(0, len(points), 256):
                    pc = torch.from_numpy(points[start:start + 256].copy()).cuda()
                    embeddings.append(model(pc)[0].cpu().numpy())
            z = np.concatenate(embeddings)
            labels, info = compute_kmeans_labels(z, 7, random_state=123, method="spherical_kmeans",
                                                l2_normalize=True, standardize=True, pca_variance=.99,
                                                pca_max_components=64, return_info=True)
            np.savez_compressed(root / f"{name}_{material}_static_embeddings.npz", projected=z, labels=labels)
            selected = {key: info[key] for key in ("silhouette_cosine", "silhouette_euclidean", "calinski_harabasz", "davies_bouldin", "pca_components", "cluster_counts")}
            outputs.setdefault(material, {})[name] = selected
        del model
        torch.cuda.empty_cache()
    for material in outputs:
        old = np.load(root / f"baseline_{material}_static_embeddings.npz")["labels"]
        new = np.load(root / f"finetuned_{material}_static_embeddings.npz")["labels"]
        outputs[material]["cluster_agreement_ari"] = adjusted_rand_score(old, new)
        # Compare uncompiled inference with the pipeline on the same cached order.
        pipeline = root / material / "analysis_inference_cache.npz"
        if pipeline.exists():
            expected = np.load(pipeline)["inv_latents"]
            actual = np.load(root / f"finetuned_{material}_static_embeddings.npz")["projected"]
            outputs[material]["compiled_vs_uncompiled_max_absolute_error"] = float(np.abs(actual - expected).max())
    (root / "matched_static_clustering.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
