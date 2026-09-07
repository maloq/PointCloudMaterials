#!/usr/bin/env python3
"""Evaluate baseline and best/final VICReg/VISReg on identical cached samples."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from src.analysis.spatiotemporal import extract, representation_metrics
from src.training_methods.spatiotemporal import write_json
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.vis_tools.latent_analysis_vis import compute_kmeans_labels


@torch.inference_mode()
def static_metrics(checkpoint, cache_root, output, name):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["hyper_parameters"])
    cfg.compile_encoder = False
    model = VICRegModule(cfg)
    model.load_state_dict({k.replace("encoder._orig_mod.", "encoder."): v for k, v in payload["state_dict"].items()}, strict=True)
    model.cuda().eval()
    results = {}
    for material in ("Al", "Mg", "Ta"):
        cache = cache_root / material
        metadata = json.loads((cache / "metadata.json").read_text())
        features, projections = [], []
        for shard in metadata["shards"]:
            points = np.load(cache / shard["samples_path"], mmap_mode="r")
            for start in range(0, len(points), 256):
                pc = model._prepare_model_input(torch.from_numpy(points[start:start + 256].copy()).cuda())
                encoded = model.encoder_io.encode(pc)
                z = model._shared_invariant(encoded.invariant, encoded.equivariant)
                features.append(z.cpu().numpy())
                projections.append(model.vicreg.project_features(z).cpu().numpy())
        arrays = {"encoder": np.concatenate(features), "projector": np.concatenate(projections)}
        results[material] = {}
        for representation in ("encoder", "projector"):
            labels, info = compute_kmeans_labels(arrays[representation], 7, random_state=123, method="spherical_kmeans",
                                                l2_normalize=True, standardize=True, pca_variance=.99,
                                                pca_max_components=64, return_info=True)
            arrays[f"{representation}_labels"] = labels
            results[material][representation] = {key: info[key] for key in (
                "silhouette_cosine", "silhouette_euclidean", "calinski_harabasz", "davies_bouldin", "pca_components", "cluster_counts")}
        np.savez_compressed(output / f"{name}_{material}_static_embeddings.npz", **arrays)
    del model
    torch.cuda.empty_cache()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--static-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.root / "comparison"
    out.mkdir(exist_ok=True)
    cfg = OmegaConf.load(args.root / "vicreg/config.yaml")
    cache = Path(cfg.data.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    checkpoints = {"baseline": str((REPOSITORY / cfg.init_from_checkpoint).resolve())}
    audits = {}
    for objective in ("vicreg", "visreg"):
        run = args.root / objective
        status = json.loads((run / "status.json").read_text())
        if status["state"] != "complete":
            raise RuntimeError(f"Cannot compare unfinished run: {run}: {status}")
        audits[objective] = json.loads((run / "checkpoint_audit.json").read_text())
        checkpoints[f"{objective}_best"] = status["best_checkpoint"]
        checkpoints[f"{objective}_final"] = str(run / "final.ckpt")
    torch.set_float32_matmul_precision("high")
    results = {}
    for name, checkpoint in checkpoints.items():
        print(f"Evaluating {name}: {checkpoint}", flush=True)
        values = extract(checkpoint, manifest, cache)
        results[name] = {"temporal": {}, "static": {}}
        arrays = {}
        for material, data in values.items():
            results[name]["temporal"][material] = {}
            for representation in ("encoder", "projector"):
                arrays[f"{material}_{representation}"] = data[representation].numpy()
                results[name]["temporal"][material][representation] = {
                    label: representation_metrics(data[representation][selection])
                    for label, selection in (("all", slice(None)), ("0.1ps", data["lags"] == 1), ("0.5ps", data["lags"] == 5))
                }
            arrays[f"{material}_lags"] = data["lags"]
        np.savez_compressed(out / f"{name}_temporal_embeddings.npz", **arrays)
        results[name]["static"] = static_metrics(checkpoint, args.static_cache, out, name)
        write_json(out / "metrics.json", dict(checkpoints=checkpoints, checkpoint_audit=audits, results=results))

    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    names = list(checkpoints)
    x = np.arange(3)
    for row, representation in enumerate(("encoder", "projector")):
        for i, name in enumerate(names):
            temporal = [results[name]["temporal"][m][representation]["all"] for m in ("Al", "Mg", "Ta")]
            for col, key in enumerate(("temporal_relative_mse", "effective_rank")):
                axes[row, col].bar(x + (i - 2) * .16, [m[key] for m in temporal], width=.16, label=name)
            axes[row, 2].bar(x + (i - 2) * .16, [results[name]["static"][m][representation]["silhouette_cosine"] for m in ("Al", "Mg", "Ta")], width=.16, label=name)
        for col, title in enumerate(("Temporal MSE / variance ↓", "Effective rank", "Static cosine silhouette ↑")):
            axes[row, col].set(xticks=x, xticklabels=["Al", "Mg", "Ta"], title=f"{representation}: {title}")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, objective in zip(axes, ("vicreg", "visreg")):
        rows = [json.loads(line) for line in (args.root / objective / "epoch_metrics.jsonl").read_text().splitlines()]
        ax.plot([r["epoch"] + 1 for r in rows], [r["metrics"]["loss/val"] for r in rows])
        ax.axvline(audits[objective]["best"]["epoch"] + 1, linestyle="--", color="tab:orange", label="Best checkpoint")
        ax.set(title=f"{objective.upper()} (own objective scale)", xlabel="Epoch", ylabel="Validation objective")
        ax.legend()
    figure.tight_layout()
    figure.savefig(out / "validation_curves.png", dpi=180)
    plt.close(figure)

    lines = ["# GFv2 spatial + temporal VICReg versus VISReg", "",
             f"Both runs use the same original GFv2 initialization, balanced Al/Mg/Ta cache, {cfg.epochs} epochs, training batch size {cfg.batch_size}, validation batch size {getattr(cfg, 'spatiotemporal_validation_batch_size', cfg.batch_size)}, seed 123 and fixed material-mixed validation batches. Spatial and temporal pair losses have equal weight. VISReg uses the existing GFv2 ablation settings: lambda 0.4, 4096 random projections, scale/shape/center coefficients 1/0.5/0.1; VICReg uses coefficients 25/25/1. Objective values have different scales and are not compared directly.", "",
             "Evaluation uses the same 768 temporal pairs per material and the same 12,000 Al / 12,000 Mg / 10,000 Ta static neighborhoods for every checkpoint, with float32 uncompiled inference. Lower relative temporal MSE indicates less drift; effective rank and static clustering help detect lost structure. A high silhouette alone does not establish physically correct clusters.", "",
             "Validation reuses source trajectories and is not an independent generalization test. Al/Mg source coordinates have float16 quantization. This is a single-seed comparison with shared hyperparameters, not separate objective tuning.", "",
             "![Comparison](comparison.png)", "", "![Validation](validation_curves.png)", ""]
    for representation in ("encoder", "projector"):
        lines.extend([f"## {representation.capitalize()}", "", "| Checkpoint | Material | Temporal MSE | Relative MSE | Effective rank | Static silhouette |", "|---|---|---:|---:|---:|---:|"])
        for name in names:
            for material in ("Al", "Mg", "Ta"):
                t = results[name]["temporal"][material][representation]["all"]
                s = results[name]["static"][material][representation]
                lines.append(f"| {name} | {material} | {t['temporal_mse']:.5f} | {t['temporal_relative_mse']:.4f} | {t['effective_rank']:.2f} | {s['silhouette_cosine']:.4f} |")
        lines.append("")
    lines.extend(["## Checkpoint audit", "", "| Run | Best epoch (1-based) | Last step | Final step |", "|---|---:|---:|---:|"])
    for name, audit in audits.items():
        lines.append(f"| {name} | {audit['best']['epoch'] + 1} | {audit['last']['global_step']} | {audit['final']['global_step']} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"Comparison complete: {out / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
