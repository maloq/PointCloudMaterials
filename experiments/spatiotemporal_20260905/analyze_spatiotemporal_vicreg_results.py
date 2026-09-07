#!/usr/bin/env python3
"""Audit saved checkpoints and compare temporal drift and latent dimensionality."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.spatiotemporal import extract, representation_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.experiment_root
    out = root / "post_training"
    status = json.loads((root / "training/status.json").read_text())
    cfg = OmegaConf.load(root / "training/config.yaml")
    manifest = json.loads((root / "views/manifest.json").read_text())
    checkpoints = {"baseline": str(Path(cfg.init_from_checkpoint).resolve()), "finetuned": status["best_checkpoint"]}
    torch.set_float32_matmul_precision("high")
    results = {}
    for name, path in checkpoints.items():
        print(f"Evaluating {name}: {path}", flush=True)
        values = extract(path, manifest, root / "views")
        results[name] = {}
        arrays = {}
        for material, data in values.items():
            results[name][material] = {}
            for representation in ("encoder", "projector"):
                arrays[f"{material}_{representation}"] = data[representation].numpy()
                results[name][material][representation] = {"all": representation_metrics(data[representation])}
                for lag in (1, 5):
                    results[name][material][representation][f"{lag / 10:.1f}ps"] = representation_metrics(data[representation][data["lags"] == lag])
            arrays[f"{material}_lags"] = data["lags"]
        np.savez_compressed(out / f"{name}_probe_embeddings.npz", **arrays)
    audit = {}
    for path in (Path(status["best_checkpoint"]), root / "training/last.ckpt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        audit[path.name] = dict(epoch=payload["epoch"], global_step=payload["global_step"])
    report = dict(checkpoints=checkpoints, checkpoint_audit=audit, metrics=results,
                  note="Same fixed 768 samples per material, all validation branches, float32 inference. Validation reuses source trajectories. Global float16 position quantization affects Al/Mg.")
    (out / "temporal_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [json.loads(line) for line in (root / "training/epoch_metrics.jsonl").read_text().splitlines()]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot([r["epoch"] + 1 for r in rows], [r["metrics"]["loss/val"] for r in rows])
    axes[0].axvline(4, color="tab:orange", linestyle="--", label="Saved best")
    axes[0].set(xlabel="Epoch (1-based)", ylabel="Validation VICReg", title="Validation minimum occurred early")
    axes[0].legend()
    x = np.arange(3)
    for i, representation in enumerate(("encoder", "projector"), 1):
        for shift, name, label in ((-.18, "baseline", "Original GFv2"), (.18, "finetuned", "Fine-tuned, epoch 4")):
            axes[i].bar(x + shift, [results[name][m][representation]["all"]["temporal_relative_mse"] for m in ("Al", "Mg", "Ta")], width=.36, label=label)
        axes[i].set(xticks=x, xticklabels=["Al", "Mg", "Ta"], ylabel="Temporal MSE / latent variance", title=representation.capitalize())
        axes[i].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "training_and_temporal_stability.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
