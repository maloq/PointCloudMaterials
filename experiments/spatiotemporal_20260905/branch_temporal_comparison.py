"""Check temporal gains within source branches using the saved matched probes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.spatiotemporal import representation_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    cfg = OmegaConf.load(root / "vicreg/config.yaml")
    manifest = json.loads((Path(cfg.data.cache_dir) / "manifest.json").read_text())
    matched = json.loads((root / "comparison/metrics.json").read_text())
    results = {}
    for name in matched["checkpoints"]:
        results[name] = {}
        with np.load(root / f"comparison/{name}_temporal_embeddings.npz") as arrays:
            for material in ("Al", "Mg", "Ta"):
                start = 0
                results[name][material] = {}
                for shard in manifest["shards"]:
                    if shard["split"] != "val" or shard["material"] != material:
                        continue
                    # extract() saves 128 pairs per Al/Mg branch, 768 for Ta.
                    count = 768 if material == "Ta" else 128
                    lags = arrays[f"{material}_lags"][start:start + count]
                    branch = {}
                    for representation in ("encoder", "projector"):
                        z = torch.from_numpy(arrays[f"{material}_{representation}"][start:start + count])
                        branch[representation] = {
                            label: representation_metrics(z[selection])
                            for label, selection in (("all", slice(None)), ("0.1ps", lags == 1), ("0.5ps", lags == 5))
                        }
                    results[name][material][shard["snapshot"]] = branch
                    start += count
                assert start == len(arrays[f"{material}_lags"]), (material, start)
    (root / "comparison/branch_temporal_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Temporal stability within each source branch", "",
             "Variance and effective rank are computed separately within each source branch. This checks whether pooled relative-MSE gains are explained only by increased separation between sources. The matched samples are the same as the main comparison; they remain from the source trajectories used for training.", ""]
    for representation in ("encoder", "projector"):
        lines.extend([f"## {representation.capitalize()}", "",
                      "| Material | Source | Baseline relative MSE | VICReg best | VISReg best | Baseline rank | VICReg rank | VISReg rank |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for material in ("Al", "Mg", "Ta"):
            for source in results["baseline"][material]:
                values = [results[name][material][source][representation]["all"] for name in ("baseline", "vicreg_best", "visreg_best")]
                numbers = [v["temporal_relative_mse"] for v in values] + [v["effective_rank"] for v in values]
                lines.append(f"| {material} | {source} | " + " | ".join(f"{n:.4f}" for n in numbers) + " |")
        lines.append("")
    (root / "comparison/BRANCH_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(root / "comparison/BRANCH_RESULTS.md")


if __name__ == "__main__":
    main()
