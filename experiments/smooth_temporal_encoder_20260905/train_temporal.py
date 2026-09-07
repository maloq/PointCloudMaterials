"""Matched GRU and learned tensor-memory experiments on the prepared pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.models.encoders.smooth_density import TensorTransport, motion_matrix
from src.temporal_vamp.smooth_state import SmoothTemporalState
from experiments.smooth_temporal_encoder_20260905.prepare import write_json
from experiments.smooth_temporal_encoder_20260905.evaluate import load_models, structure_fit, spectrum
from experiments.smooth_temporal_encoder_20260905.run import load_split, make_loss


def temporal_data(cfg, output, manifest, epsilon):
    transport = TensorTransport(cfg["density"]["max_ell"]).cuda().float()
    h = cfg["temporal"]["history_frames"]
    result = {}
    for split in cfg["splits"]:
        q, meta = load_split(output, manifest, split)
        q = torch.tensor(q[:, 0], device="cuda")
        raw = torch.tensor(np.load(output / "embeddings" / f"mace_product_{split}.npy"), device="cuda")
        target = dict(np.load(output / "history" / f"{split}.targets.npz"))
        cross, rows, all_rows, shard_slices = [], [], [], []
        offset = 0
        for shard in manifest["shards"]:
            if shard["split"] != split:
                continue
            c, t = shard["centers"], shard["frames"]
            cross.append(np.load(output / "data" / (shard["stem"]+".cross.npy")).reshape(-1, 3, 3))
            end = np.arange(h-1, t)
            index = (offset+np.arange(c)[:, None, None]*t+end[None, :, None]-np.arange(h-1, -1, -1)[None, None, :])
            valid = end+max(cfg["temporal"]["horizons_frames"]) < t
            rows.append(index[:, valid].reshape(-1, h))
            all_rows.append(index.reshape(-1, h))
            shard_slices.append(dict(stem=shard["stem"], centers=c, frames=len(end)))
            offset += c*t
        if len(np.concatenate(rows)) != len(target["current"]):
            raise RuntimeError(f"History ordering mismatch: {split}")
        matrices = {}
        cross = torch.tensor(np.concatenate(cross), device="cuda")
        if not torch.isfinite(cross).all():
            raise FloatingPointError(f"Non-finite observed motion cross-moment: {split}")
        for method in ("kabsch", "smooth"):
            chunks = [transport.matrices(motion_matrix(chunk, method, epsilon)) for chunk in cross.split(4096)]
            matrices[method] = [torch.cat([chunk[ell] for chunk in chunks]) for ell in range(cfg["density"]["max_ell"]+1)]
        result[split] = dict(q=q, raw=raw, rows=torch.tensor(np.concatenate(rows), device="cuda"),
                             all_rows=torch.tensor(np.concatenate(all_rows), device="cuda"), matrices=matrices,
                             target=target, meta=meta, shard_slices=shard_slices)
    mean, std = result["train"]["raw"].mean(0), result["train"]["raw"].std(0)
    std = std.clamp_min(.01*std.median())
    target_mean = result["train"]["target"]["current"].mean(0)
    target_std = result["train"]["target"]["current"].std(0)
    for split, data in result.items():
        data["raw"] = (data["raw"]-mean)/std
        data["future"] = torch.tensor((data["target"]["future"]-target_mean)/target_std, device="cuda")
        data["current"] = torch.tensor((data["target"]["current"]-target_mean)/target_std, device="cuda")
        data["material"] = torch.tensor(data["target"]["material"], device="cuda")
    return result, mean, std, target_mean, target_std


def forward_batch(model, data, ids):
    rows = data["rows"][ids]
    if model.mode == "gated_kabsch":
        matrices = [block[rows] for block in data["matrices"]["kabsch"]]
    elif model.mode == "gated_smooth":
        matrices = [block[rows] for block in data["matrices"]["smooth"]]
    else:
        matrices = []
    return model(data["q"][rows], data["raw"][rows], matrices, data["material"][ids])


@torch.inference_mode()
def predict(model, data, batch_size):
    model.eval()
    future, state = [], []
    for ids in torch.arange(len(data["rows"]), device="cuda").split(batch_size):
        f, z, _ = forward_batch(model, data, ids)
        future.append(f)
        state.append(z[:, -1])
    return torch.cat(future), torch.cat(state)


def run(cfg, output):
    manifest = json.loads((output / "data/manifest.json").read_text())
    forecast = json.loads((output / "forecast_metrics.json").read_text())
    # Reuse the regularization chosen by the fixed-filter validation experiment.
    selected = forecast["selected_by_validation"]["smooth"]
    epsilon = float(selected.split("_eps")[1].split("_tau")[0])
    settings = dict(**cfg["temporal_training"], smooth_epsilon=epsilon, seeds=cfg["training"]["seeds"])
    write_json(output / "temporal_training_config.json", settings)
    scaling, spatial = load_models(cfg, output)
    data, mean, std, target_mean, target_std = temporal_data(cfg, output, manifest, epsilon)
    variance_loss = make_loss(32)
    report = []
    for mode in settings["modes"]:
        for seed in settings["seeds"]:
            torch.manual_seed(seed)
            model = SmoothTemporalState(spatial["mace_product"], mean, std, mode=mode).cuda()
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"], weight_decay=settings["weight_decay"], fused=True)
            directory = output / "temporal_models" / f"{mode}_seed{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            best, best_epoch, started = float("inf"), -1, time.monotonic()
            for epoch in range(settings["epochs"]):
                epoch_start = time.monotonic()
                model.train()
                order = torch.randperm(len(data["train"]["rows"]), device="cuda")
                rate = settings["learning_rate"]*.5*(1+np.cos(np.pi*epoch/settings["epochs"]))
                for group in optimizer.param_groups:
                    group["lr"] = rate
                total, count = 0., 0
                for ids in order.split(settings["batch_size"]):
                    future, states, current = forward_batch(model, data["train"], ids)
                    pred_loss = (future-data["train"]["future"][ids]).square().mean()
                    state = states[:, -1]
                    loss = (pred_loss + settings["current_reconstruction_weight"]*(current-data["train"]["current"][ids, :8]).square().mean()
                            + settings["state_variance_weight"]*variance_loss._variance_loss(state)
                            + settings["state_covariance_weight"]*variance_loss._covariance_loss(state)
                            + settings["state_update_weight"]*(states[:, 2:]-states[:, 1:-1]).square().mean())
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite temporal loss: {mode}/{seed}/{epoch}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
                    optimizer.step()
                    total += pred_loss.item()*len(ids)
                    count += len(ids)
                predicted, _ = predict(model, data["val"], settings["batch_size"])
                validation = (predicted-data["val"]["future"]).square().mean().item()
                record = dict(epoch=epoch, train_prediction_mse=total/count, validation_prediction_mse=validation,
                              seconds=time.monotonic()-epoch_start)
                with (directory / "epochs.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False)+"\n")
                payload = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                               mode=mode, seed=seed, config=cfg, temporal_config=settings,
                               target_mean=target_mean, target_std=target_std, validation_mse=validation)
                torch.save(payload, directory / "last.pt")
                if validation < best:
                    best, best_epoch = validation, epoch
                    torch.save(payload, directory / "best.pt")
                write_json(directory / "status.json", dict(state="running", **record, best_epoch=best_epoch))
                print(f"Temporal {mode}/{seed}: {epoch+1}/{settings['epochs']}, val={validation:.5f}, {record['seconds']:.2f}s", flush=True)
            torch.save(payload, directory / "final.pt")
            best_payload = torch.load(directory / "best.pt", map_location="cuda", weights_only=False)
            model.load_state_dict(best_payload["state_dict"])
            predictions, states = {}, {}
            for split in cfg["splits"]:
                predictions[split], states[split] = predict(model, data[split], settings["batch_size"])
                np.save(directory / f"{split}.state.npy", states[split].cpu().numpy())
            np.save(directory / "test.forecast.npy", predictions["test"].cpu().numpy())
            probe_metrics, probe, _ = structure_fit(states["train"], data["train"]["target"]["labels"],
                    states["val"], data["val"]["target"]["labels"], states["test"], data["test"]["target"]["labels"])
            torch.save(probe, directory / "structure_probe.pt")
            metrics = dict(mode=mode, seed=seed, best_epoch=best_epoch, validation_mse=best,
                           elapsed_seconds=time.monotonic()-started, trainable_parameters=sum(p.numel() for p in params),
                           structure=probe_metrics, by_material={})
            for mi, material in enumerate(("Al", "Mg", "Ta")):
                mask = data["test"]["target"]["material"] == mi
                train_mask = data["train"]["target"]["material"] == mi
                mse = (predictions["test"][mask]-data["test"]["future"][mask]).square().mean((0, 2))
                persistence = (data["test"]["current"][mask, None]-data["test"]["future"][mask]).square().mean((0, 2))
                conditional_mean = data["train"]["future"][train_mask].mean(0)
                mean_mse = (conditional_mean-data["test"]["future"][mask]).square().mean((0, 2))
                metrics["by_material"][material] = dict(mse=mse.cpu().tolist(),
                      skill_vs_persistence=(1-mse/persistence).cpu().tolist(), skill_vs_material_mean=(1-mse/mean_mse).cpu().tolist(),
                      **spectrum(states["test"][mask]))
            # States on every test observation with a complete five-frame past.
            all_states = []
            test = data["test"]
            with torch.inference_mode():
                for rows in test["all_rows"].split(settings["batch_size"]):
                    method = {"gated_kabsch":"kabsch", "gated_smooth":"smooth"}.get(mode)
                    matrices = [] if method is None else [block[rows] for block in test["matrices"][method]]
                    material = torch.tensor(test["meta"]["material"], device="cuda")[rows[:, -1]]
                    _, values, _ = model(test["q"][rows], test["raw"][rows], matrices, material)
                    all_states.append(values[:, -1].cpu().numpy())
            all_states = np.concatenate(all_states)
            np.save(directory / "test.all_rolling_states.npy", all_states)
            position = 0
            distances = {m:[] for m in ("Al", "Mg", "Ta")}
            for shard in test["shard_slices"]:
                count = shard["centers"]*shard["frames"]
                z = all_states[position:position+count].reshape(shard["centers"], shard["frames"], 32)
                position += count
                material = shard["stem"].split("_")[0]
                mi = ("Al", "Mg", "Ta").index(material)
                scale = states["train"][data["train"]["target"]["material"] == mi].var(0).sum().sqrt().item()
                distances[material].append(np.linalg.norm(np.diff(z, axis=1), axis=-1).ravel()/scale)
            for material, values in distances.items():
                metrics["by_material"][material]["p95_scaled_step"] = float(np.quantile(np.concatenate(values), .95))
            write_json(directory / "metrics.json", metrics)
            write_json(directory / "status.json", dict(state="complete", best_epoch=best_epoch, elapsed_seconds=metrics["elapsed_seconds"]))
            report.append(metrics)
            write_json(output / "neural_temporal_metrics.json", report)
            del model, optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = ROOT / cfg["output"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status = dict(state="running", pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "neural_temporal_status.json", status)
    try:
        run(cfg, output)
        status.update(state="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "neural_temporal_status.json", status)


if __name__ == "__main__":
    main()
