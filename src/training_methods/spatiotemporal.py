#!/usr/bin/env python3
"""Train GFv2 with local progress records and a fixed temporal stability probe."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from src.models import EncoderAdapter
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.training_methods.trainer import train_model


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def stability_probe(model, cache_dir):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    was_training = model.training
    model.eval()
    encoder = model.encoder._orig_mod if model._compile_encoder else model.encoder
    adapter = EncoderAdapter(encoder)
    collected = {material: [] for material in ("Al", "Mg", "Ta")}
    for shard in manifest["shards"]:
        if shard["split"] != "val":
            continue
        views = np.load(cache_dir / shard["views"], mmap_mode="r")
        pairs = np.load(cache_dir / shard["pairs"], mmap_mode="r")
        count = 768 if shard["material"] == "Ta" else 128
        selected = np.linspace(0, len(views) - 1, count, dtype=np.int64)
        chunks = []
        for start in range(0, count, 128):
            data = torch.from_numpy(views[selected[start:start + 128]].astype(np.float32)).to(model.device)
            batch_size = data.shape[0]
            encoded = adapter.encode(data.flatten(0, 1))
            features = model._shared_invariant(encoded.invariant, encoded.equivariant)
            projected = model.vicreg.project_features(features)
            chunks.append((features.reshape(batch_size, 3, -1).cpu(), projected.reshape(batch_size, 3, -1).cpu()))
        collected[shard["material"]].append((torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks]), pairs[selected, 3]))
    results = {}
    for material, chunks in collected.items():
        lags = np.concatenate([c[2] for c in chunks])
        report = {"samples": len(lags)}
        for representation, column in (("encoder", 0), ("projector", 1)):
            features = torch.cat([c[column] for c in chunks]).float()
            metrics = {"anchor_std": features[:, 0].std(dim=0).mean().item()}
            for lag in (1, 5):
                a, t = features[lags == lag, 0], features[lags == lag, 2]
                mse = (a - t).square().mean()
                metrics[f"temporal_{lag / 10:.1f}ps_mse"] = mse.item()
                metrics[f"temporal_{lag / 10:.1f}ps_relative_mse"] = (mse / a.var(dim=0).mean().clamp_min(1.e-8)).item()
            report[representation] = metrics
        results[material] = report
    model.train(was_training)
    return results


class ProgressCheckpoint(ModelCheckpoint):
    def __init__(self, root):
        super().__init__(dirpath=root, filename="GFv2-spatiotemporal-{epoch:03d}",
                         monitor="loss/val", mode="min", save_top_k=1, save_last=True)
        self.root = root
        self.started = time.monotonic()

    def progress(self, trainer, *, state):
        elapsed = time.monotonic() - self.started
        completed = trainer.global_step
        write_json(self.root / "status.json", dict(
            state=state, updated_at=datetime.now(timezone.utc).isoformat(),
            epoch=trainer.current_epoch, global_step=completed, max_epochs=trainer.max_epochs,
            elapsed_seconds=elapsed, seconds_per_step=elapsed / completed if completed else None,
            best_checkpoint=self.best_model_path,
        ))

    def on_fit_start(self, trainer, pl_module):
        probe = stability_probe(pl_module, Path(pl_module.hparams.data.cache_dir))
        write_json(self.root / "baseline_stability.json", probe)
        print("Baseline temporal stability:", json.dumps(probe), flush=True)
        self.progress(trainer, state="training")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.global_step == 1 or trainer.global_step % 20 == 0:
            self.progress(trainer, state="training")

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if not trainer.sanity_checking:
            metrics = {key: value.item() for key, value in trainer.callback_metrics.items() if value.numel() == 1}
            with (self.root / "epoch_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(epoch=trainer.current_epoch, step=trainer.global_step, metrics=metrics)) + "\n")
            self.progress(trainer, state="training")

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        # This Lightning version updates save_last only when a top-k file is
        # saved. Explicitly retain the latest epoch even when validation worsens.
        trainer.save_checkpoint(self.root / "last.ckpt")
        if (trainer.current_epoch + 1) % 10 == 0:
            trainer.save_checkpoint(self.root / f"epoch-{trainer.current_epoch + 1:03d}.ckpt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True, help="Hydra configuration name under configs/.")
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(version_base=None, config_dir=str(REPOSITORY / "configs")):
        cfg = compose(config_name=args.config_name)
    OmegaConf.save(cfg, args.run_dir / "config.yaml", resolve=True)
    (args.run_dir / ".hydra").mkdir()
    OmegaConf.save(cfg, args.run_dir / ".hydra/config.yaml", resolve=True)
    torch.set_float32_matmul_precision("high")
    checkpoint = ProgressCheckpoint(args.run_dir)
    try:
        trainer, model, _, _ = train_model(cfg, VICRegModule, run_dir=str(args.run_dir), checkpoint_callbacks=[checkpoint], run_test=False)
        trainer.save_checkpoint(args.run_dir / "final.ckpt")
        audit = {}
        for name, path in (("best", Path(checkpoint.best_model_path)), ("last", args.run_dir / "last.ckpt"), ("final", args.run_dir / "final.ckpt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            audit[name] = dict(path=str(path), epoch=payload["epoch"], global_step=payload["global_step"])
            if name != "best" and payload["global_step"] != trainer.global_step:
                raise RuntimeError(f"Stale {name} checkpoint: {audit[name]}, expected step {trainer.global_step}")
        write_json(args.run_dir / "checkpoint_audit.json", audit)
        write_json(args.run_dir / "final_checkpoint_stability.json", dict(checkpoint=str(args.run_dir / "final.ckpt"), metrics=stability_probe(model, Path(cfg.data.cache_dir))))
        best = torch.load(checkpoint.best_model_path, map_location=model.device, weights_only=False)
        model.load_state_dict(best["state_dict"], strict=True)
        after = stability_probe(model, Path(cfg.data.cache_dir))
        write_json(args.run_dir / "best_checkpoint_stability.json", dict(checkpoint=checkpoint.best_model_path, metrics=after))
        checkpoint.progress(trainer, state="complete")
    except BaseException as error:
        write_json(args.run_dir / "status.json", dict(state="failed", error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
