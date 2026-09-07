"""Control dimensionality when comparing physical drift and structure readouts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.smooth_temporal_encoder_20260905.evaluate import structure_fit, spectrum
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = ROOT / cfg["output"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((output / "data/manifest.json").read_text())
    metadata = {split:dict(np.load(output / "embeddings" / f"{split}_metadata.npz")) for split in cfg["splits"]}
    report = {}
    for name in ("density_pca", "power_mlp", "mace_product", "geoframe_vicreg"):
        z = {split:torch.tensor(np.load(output / "embeddings" / f"{name}_{split}.npy"), device="cuda") for split in cfg["splits"]}
        mean = z["train"].mean(0)
        centered = (z["train"]-mean).double()
        _, eigenvectors = torch.linalg.eigh(centered.T@centered)
        basis = eigenvectors.flip(-1).float()
        report[name] = {}
        for dim in (2, 4, 8, 16, 32, 128):
            projected = {split:(v-mean)@basis[:, :dim] for split,v in z.items()}
            metrics, _, _ = structure_fit(projected["train"], metadata["train"]["labels"], projected["val"], metadata["val"]["labels"], projected["test"], metadata["test"]["labels"])
            by_material = {}
            for mi, material in enumerate(("Al", "Mg", "Ta")):
                steps = []
                offset = 0
                scale = projected["train"][metadata["train"]["material"] == mi].var(0).sum().sqrt().item()
                for shard in manifest["shards"]:
                    if shard["split"] != "test":
                        continue
                    n = shard["centers"]*shard["frames"]
                    values = projected["test"][offset:offset+n].reshape(shard["centers"], shard["frames"], dim)
                    offset += n
                    if shard["material"] == material:
                        # Match the complete-history origin range of other rolling states.
                        values = values[:, cfg["temporal"]["history_frames"]-1:]
                        steps.append(torch.linalg.vector_norm(torch.diff(values, dim=1), dim=-1).flatten().cpu().numpy()/scale)
                by_material[material] = dict(p95_scaled_step=float(np.quantile(np.concatenate(steps), .95)),
                                             **spectrum(projected["test"][metadata["test"]["material"] == mi]))
            report[name][str(dim)] = dict(structure=metrics, by_material=by_material)
        print(f"Dimension controls: {name}", flush=True)
    write_json(output / "rank_controls.json", dict(results=report,
               protocol="PCA fitted only on training embeddings, without whitening. Identical dimensions do not imply identical effective rank; both are reported. PTM probes refit at each dimension using training labels and validation alpha selection."))


if __name__ == "__main__":
    main()
