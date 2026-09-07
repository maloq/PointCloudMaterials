"""Full six-snapshot Al evaluation at the existing 772,953 analysis centers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, balanced_accuracy_score, adjusted_rand_score
import torch
from src.models.encoders.smooth_density import SmoothDensity
from src.vis_tools.latent_analysis_vis import compute_kmeans_labels
from experiments.smooth_temporal_encoder_20260905.evaluate import load_models, load_geoframe, power_pca, spectrum
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def full_ptm(points, threshold):
    from ovito.data import DataCollection, Particles
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    particles = Particles(count=len(points))
    particles.create_property("Position", data=points)
    data = DataCollection()
    data.objects.append(particles)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(rmsd_cutoff=threshold))
    result = pipeline.compute()
    return np.asarray(result.particles["Structure Type"]).copy()


@torch.inference_mode()
def run(cfg, output):
    directory = output / "full_static_Al"
    directory.mkdir(exist_ok=False)
    cache = ROOT / cfg["full_static_cache"]
    metadata = json.loads((cache / "metadata.json").read_text())
    source_manifest = json.loads(Path(cfg["views_manifest"]).read_text())
    radius = next(s["radius"] for s in source_manifest["sources"] if s["material"] == "Al") * cfg["cutoff_factor"]
    scaling, models = load_models(cfg, output)
    density = SmoothDensity(**cfg["density"]).cuda()
    geoframe = load_geoframe(cfg)
    names = ["density_pca", "power_mlp", "mace_product", "geoframe_vicreg"]
    embeddings = {name:open_memmap(directory / f"{name}.npy", mode="w+", dtype=np.float32,
                                  shape=(metadata["total_samples"], 128)) for name in names}
    all_coords = open_memmap(directory / "coords.npy", mode="w+", dtype=np.float32, shape=(metadata["total_samples"], 3))
    labels = np.empty(metadata["total_samples"], dtype=np.int32)
    source_ids = np.empty(metadata["total_samples"], dtype=np.int32)
    frames, offset = [], 0
    for source_index, shard in enumerate(metadata["shards"]):
        started = time.monotonic()
        coords = np.load(cache / shard["coords_path"])
        source = next(s for s in metadata["request"]["sources"] if s["name"] == shard["source"])
        path = ROOT / source["files"][0]["path"]
        points = np.load(path)
        tree = cKDTree(points, balanced_tree=False)
        whole_labels = full_ptm(points, cfg["ptm_rmsd_cutoff"])
        margin = float("inf")
        maximum_support = 0
        for start in range(0, len(coords), cfg["inference_batch_size"]):
            centers = coords[start:start+cfg["inference_batch_size"]]
            distances, ids = tree.query(centers, k=cfg["candidate_neighbors"]+2, workers=4)
            if distances[:, 0].max() > 1e-5:
                raise RuntimeError(f"Static centers no longer match source atoms: {path}")
            margin = min(margin, float(distances[:, -1].min()-radius))
            if margin <= 0:
                raise RuntimeError(f"Static radius support exceeds candidate capacity: {path}")
            maximum_support = max(maximum_support, int((distances[:, 1:-1]<radius).sum(1).max()))
            x = torch.tensor((points[ids[:, 1:-1]]-centers[:, None])/radius, device="cuda", dtype=torch.float32)
            q = density(x)
            features = {"density_pca":power_pca(density, q, scaling)}
            features.update({name:model.forward_moments(q) for name,model in models.items()})
            gf_cloud = torch.cat((x.new_zeros(len(x), 1, 3), x[:, :79]*cfg["cutoff_factor"]), 1)
            features["geoframe_vicreg"] = geoframe.encoder.forward_features(gf_cloud)
            section = slice(offset+start, offset+start+len(centers))
            for name, value in features.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite static embedding: {name}/{path}/{start}")
                embeddings[name][section] = value.cpu().numpy()
            labels[section] = whole_labels[ids[:, 0]]
            source_ids[section] = source_index
            all_coords[section] = centers
        frame = dict(source=shard["source"], count=len(coords), offset=offset, radius_A=radius,
                     minimum_candidate_margin_A=margin, maximum_support=maximum_support,
                     path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     elapsed_seconds=time.monotonic()-started)
        frames.append(frame)
        offset += len(coords)
        write_json(directory / "coverage.json", dict(expected=metadata["total_samples"], processed=offset, frames=frames))
        print(f"Full static Al {shard['source']}: {len(coords):,} centers, {frame['elapsed_seconds']:.1f}s", flush=True)
    if offset != metadata["total_samples"]:
        raise RuntimeError(f"Incomplete static coverage: {offset}/{metadata['total_samples']}")
    for value in embeddings.values():
        value.flush()
    all_coords.flush()
    np.savez(directory / "metadata.npz", ptm_labels=labels, source_ids=source_ids)
    del geoframe
    report = {}
    for name in names:
        values = np.asarray(embeddings[name])
        probe = torch.load(output / "embeddings" / f"{name}_structure_probe.pt", map_location="cuda", weights_only=True)
        predictions = []
        for start in range(0, len(values), 8192):
            z = torch.tensor(values[start:start+8192], device="cuda")
            x = torch.cat(((z-probe["mean"])/probe["std"], z.new_ones(len(z), 1)), 1).double()
            predictions.append(probe["classes"][(x@probe["coefficients"]).argmax(1)].cpu().numpy())
        predicted = np.concatenate(predictions)
        np.save(directory / f"{name}.predicted_ptm.npy", predicted)
        # Reuse the repository's full-data spherical clustering protocol.
        clusters, info = compute_kmeans_labels(values, 7, random_state=123, method="spherical_kmeans",
                                               standardize=True, l2_normalize=True, pca_variance=.99,
                                               pca_max_components=64, return_info=True)
        np.save(directory / f"{name}.clusters.npy", clusters)
        selected = {key:info[key] for key in ("silhouette_cosine", "silhouette_euclidean", "davies_bouldin", "calinski_harabasz", "pca_components", "cluster_counts")}
        report[name] = dict(**selected, ptm_macro_f1=float(f1_score(labels, predicted, average="macro", zero_division=0)),
                            ptm_balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                            cluster_ptm_ari=float(adjusted_rand_score(labels, clusters)),
                            **spectrum(torch.tensor(values, device="cuda")))
        write_json(directory / "metrics.json", report)
        print(f"Full static metrics {name}: {report[name]}", flush=True)
    write_json(directory / "protocol.json", dict(total_samples=offset, frames=frames,
               structures={str(k):int(v) for k,v in zip(*np.unique(labels, return_counts=True))},
               fitting="All 772,953 rows used in spherical k-means; repository sampled internal clustering diagnostics.",
               readout="Class-balanced ridge PTM probes trained only on ordinary-MD training data, alpha chosen on its validation split.",
               geoframe="Recomputed raw VICReg encoder on the nearest 80 atoms at the same saved centers, with fixed Al normalization from the temporal source manifest. Earlier static reports used projector outputs and per-snapshot normalization; their scores are not identical-protocol comparisons."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = ROOT / cfg["output"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status = dict(state="running", pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "full_static_status.json", status)
    try:
        run(cfg, output)
        status.update(state="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "full_static_status.json", status)


if __name__ == "__main__":
    main()
