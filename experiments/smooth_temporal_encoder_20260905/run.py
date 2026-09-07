"""Feature preparation and spatial training for the smooth temporal encoder pilot."""
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
from omegaconf import OmegaConf
import torch
from src.models.encoders.smooth_density import DensityEncoder, SmoothDensity
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def feature_cache(cfg, output, manifest):
    density = SmoothDensity(**cfg["density"]).cuda().eval()
    started = time.monotonic()
    with torch.inference_mode():
        for index, shard in enumerate(manifest["shards"]):
            source = output / "data" / shard["stem"]
            clouds = np.load(str(source)+".clouds.npy", mmap_mode="r")
            c, t, _, n, _ = clouds.shape
            values = open_memmap(str(source)+".moments.npy", mode="w+", dtype=np.float32,
                                 shape=(c, t, 3, cfg["density"]["radial_channels"], (cfg["density"]["max_ell"]+1)**2))
            flat = clouds.reshape(c*t, 2, n, 3)
            result = values.reshape(c*t, *values.shape[2:])
            generator = torch.Generator(device="cuda").manual_seed(cfg["seed"]+index)
            for start in range(0, len(flat), cfg["inference_batch_size"]):
                points = torch.tensor(flat[start:start+cfg["inference_batch_size"]], device="cuda")
                noise = torch.randn(points[:, 0].shape, device="cuda", generator=generator)
                noise = (noise * cfg["jitter_std_A"]).clamp(-cfg["jitter_clip_A"], cfg["jitter_clip_A"])/shard["radius_A"]
                augmented = torch.cat((points, (points[:, 0]+noise)[:, None]), 1)
                q = density(augmented.flatten(0, 1)).reshape(len(points), 3, *values.shape[-2:])
                if not torch.isfinite(q).all():
                    raise FloatingPointError(f"Non-finite density: {shard['stem']} rows {start}")
                result[start:start+len(points)] = q.cpu().numpy()
            values.flush()
            print(f"Density cache: {shard['stem']} ({c*t} observations)", flush=True)
    write_json(output / "features_status.json", dict(state="complete", elapsed_seconds=time.monotonic()-started,
               descriptor="Radial Gaussian x spherical-harmonic density; power spectrum is SOAP-like, not DScribe SOAP.",
               view_order=["anchor", "spatial", "bounded_jitter"], cached_augmentation_note="One fixed jitter realization per observation."))


def load_split(output, manifest, split):
    arrays, metadata = [], {key:[] for key in ("material", "source", "frame", "center_id", "labels", "nonaffine_A2")}
    material_index = {"Al":0, "Mg":1, "Ta":2}
    for index, shard in enumerate(manifest["shards"]):
        if shard["split"] != split:
            continue
        prefix = output / "data" / shard["stem"]
        q = np.load(str(prefix)+".moments.npy", mmap_mode="r")
        arrays.append(q.reshape(-1, *q.shape[2:]))
        saved = np.load(str(prefix)+".metadata.npz")
        shape = q.shape[:2]
        metadata["material"].append(np.full(shape, material_index[shard["material"]], dtype=np.int64).ravel())
        metadata["source"].append(np.full(shape, index, dtype=np.int64).ravel())
        metadata["frame"].append(np.broadcast_to(saved["frames"], shape).ravel())
        metadata["center_id"].append(np.broadcast_to(saved["center_ids"][:, None], shape).ravel())
        metadata["labels"].append(saved["labels"].ravel())
        metadata["nonaffine_A2"].append(saved["nonaffine_A2"].ravel())
    return np.concatenate(arrays), {key:np.concatenate(values) for key, values in metadata.items()}


def make_loss(dim):
    cfg = OmegaConf.create(dict(vicreg_enabled=True, vicreg_weight=1., vicreg_embed_dim=dim,
                               vicreg_projector_mode="identity", vicreg_sim_coeff=25.,
                               vicreg_std_coeff=25., vicreg_cov_coeff=1., vicreg_std_target=1.,
                               vicreg_std_eps=1e-4, vicreg_drop_ratio=0., vicreg_jitter_std=0.))
    return VICRegLoss.from_config(cfg, input_dim=dim).cuda()


def paired_loss(loss, features, spatial_weight):
    anchor, spatial, jitter = features.unbind(1)
    # Call the repository raw objective: the outer public helper sanitizes NaNs,
    # which is inappropriate for this research run's explicit finite-loss gate.
    jitter_loss, _ = loss._loss(anchor, jitter)
    spatial_loss, _ = loss._loss(anchor, spatial)
    value = (jitter_loss + spatial_weight * spatial_loss)/(1+spatial_weight)
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite spatial VICReg loss")
    return value


@torch.no_grad()
def fitted_scaling(density, train):
    powers = torch.cat([density.power(batch[:, 0]) for batch in train.split(4096)])
    mean, std = powers.mean(0), powers.std(0)
    floor = .01 * std.median()
    std = std.clamp_min(floor)
    moment_scale = train[:, 0].square().mean((0, 2)).sqrt()
    # A fixed basis and target space, fit exclusively on training anchors.
    standardized = (powers - mean)/std
    covariance = standardized.double().T @ standardized.double()/(len(powers)-1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    return dict(power_mean=mean, power_std=std, moment_scale=moment_scale,
                pca_vectors=vectors[:, -128:].flip(-1).float(), pca_eigenvalues=eigenvalues[-128:].flip(-1).float())


def construct(cfg, kind, scaling):
    model = DensityEncoder(SmoothDensity(**cfg["density"]), kind, **cfg["encoder"]).cuda()
    for key in ("power_mean", "power_std", "moment_scale"):
        getattr(model, key).copy_(scaling[key])
    return model


@torch.inference_mode()
def encode_moments(model, moments, batch_size=4096):
    return torch.cat([model.forward_moments(batch) for batch in moments.split(batch_size)])


def train_spatial(cfg, output, manifest):
    train_np, train_meta = load_split(output, manifest, "train")
    val_np, val_meta = load_split(output, manifest, "val")
    train = torch.tensor(train_np, device="cuda")
    val = torch.tensor(val_np, device="cuda")
    del train_np, val_np
    density = SmoothDensity(**cfg["density"]).cuda()
    scaling = fitted_scaling(density, train)
    torch.save({key:value.cpu() for key, value in scaling.items()}, output / "scaling.pt")
    settings = cfg["training"]
    objective = make_loss(cfg["encoder"]["latent_dim"])
    summary = []
    for kind in ("power_mlp", "mace_product"):
        for seed in settings["seeds"]:
            torch.manual_seed(seed)
            model = construct(cfg, kind, scaling)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"], fused=True)
            run = output / f"{kind}_seed{seed}"
            run.mkdir(exist_ok=False)
            started, best, best_epoch = time.monotonic(), float("inf"), -1
            batch_size = settings["batch_size"]
            # Fixed mixed validation batches, identical for all candidates.
            validation_order = torch.randperm(len(val), device="cuda", generator=torch.Generator(device="cuda").manual_seed(100))
            history = []
            for epoch in range(settings["epochs"]):
                model.train()
                order = torch.randperm(len(train), device="cuda")
                current_lr = settings["learning_rate"] * .5 * (1+np.cos(np.pi*epoch/settings["epochs"]))
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                loss_sum, count = 0., 0
                epoch_start = time.monotonic()
                for ids in order.split(batch_size):
                    if len(ids) < 2:
                        raise RuntimeError("VICReg minibatch contains fewer than two samples")
                    q = train[ids]
                    z = model.forward_moments(q.flatten(0, 1)).reshape(len(q), 3, -1)
                    value = paired_loss(objective, z, settings["spatial_weight"])
                    optimizer.zero_grad(set_to_none=True)
                    value.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += value.item()*len(ids)
                    count += len(ids)
                model.eval()
                val_sum = 0.
                with torch.no_grad():
                    for ids in validation_order.split(batch_size):
                        q = val[ids]
                        z = model.forward_moments(q.flatten(0, 1)).reshape(len(q), 3, -1)
                        val_sum += paired_loss(objective, z, settings["spatial_weight"]).item()*len(ids)
                val_loss = val_sum/len(val)
                record = dict(epoch=epoch, train_loss=loss_sum/count, val_loss=val_loss,
                              seconds=time.monotonic()-epoch_start, learning_rate=current_lr, gradient_norm=float(norm))
                history.append(record)
                with (run / "epochs.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False)+"\n")
                payload = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), kind=kind,
                               seed=seed, epoch=epoch, config=cfg, val_loss=val_loss)
                torch.save(payload, run / "last.pt")
                if val_loss < best:
                    best, best_epoch = val_loss, epoch
                    torch.save(payload, run / "best.pt")
                write_json(run / "status.json", dict(state="running", **record, best_epoch=best_epoch))
                print(f"{kind}/{seed}: epoch {epoch+1}/{settings['epochs']}, train={record['train_loss']:.4f}, val={val_loss:.4f}, {record['seconds']:.2f}s", flush=True)
            torch.save(payload, run / "final.pt")
            result = dict(kind=kind, seed=seed, best_epoch=best_epoch, best_val_loss=best,
                          last_epoch=epoch, elapsed_seconds=time.monotonic()-started,
                          parameters=sum(p.numel() for p in model.parameters()), checkpoint=str(run / "best.pt"))
            write_json(run / "status.json", dict(state="complete", **result))
            summary.append(result)
            write_json(output / "training_summary.json", summary)
            del optimizer, model
    write_json(output / "training_data.json", dict(train_samples=len(train), val_samples=len(val),
               train_ptm_counts={str(k):int(v) for k,v in zip(*np.unique(train_meta["labels"], return_counts=True))},
               val_ptm_counts={str(k):int(v) for k,v in zip(*np.unique(val_meta["labels"], return_counts=True))},
               objective="VICReg directly on exported 128D encoder output; spatial weight .25, bounded-jitter weight 1; no temporal attraction."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("features", "train", "spatial"), required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = ROOT / cfg["output"]
    manifest = json.loads((output / "data/manifest.json").read_text())
    if manifest["state"] != "complete":
        raise RuntimeError("Radius data preparation is incomplete")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status = dict(state="running", stage=args.stage, pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "spatial_status.json", status)
    try:
        if args.stage in ("features", "spatial"):
            feature_cache(cfg, output, manifest)
        if args.stage in ("train", "spatial"):
            train_spatial(cfg, output, manifest)
        status.update(state="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "spatial_status.json", status)


if __name__ == "__main__":
    main()
