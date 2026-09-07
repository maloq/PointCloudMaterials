"""Matched continuity, structure and finite-history forecast pilot measurements."""
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
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import torch
from src.models.encoders.smooth_density import SmoothDensity, TensorTransport, motion_matrix
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from experiments.smooth_temporal_encoder_20260905.prepare import write_json
from experiments.smooth_temporal_encoder_20260905.run import construct, encode_moments, load_split


def load_models(cfg, output):
    scaling = {key:value.cuda() for key,value in torch.load(output / "scaling.pt", weights_only=True).items()}
    records = json.loads((output / "training_summary.json").read_text())
    models, selected = {}, {}
    for kind in ("power_mlp", "mace_product"):
        record = min((r for r in records if r["kind"] == kind), key=lambda r:r["best_val_loss"])
        model = construct(cfg, kind, scaling)
        payload = torch.load(record["checkpoint"], map_location="cuda", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        models[kind] = model.eval().requires_grad_(False)
        selected[kind] = record
    write_json(output / "selected_models.json", selected)
    return scaling, models


def load_geoframe(cfg):
    payload = torch.load(cfg["geoframe_checkpoint"], map_location="cpu", weights_only=False)
    settings = OmegaConf.create(payload["hyper_parameters"])
    settings.compile_encoder = False
    model = VICRegModule(settings)
    model.load_state_dict({key.replace("encoder._orig_mod.", "encoder."):value for key,value in payload["state_dict"].items()}, strict=True)
    return model.cuda().eval().requires_grad_(False)


def power_pca(density, q, scaling):
    return ((density.power(q)-scaling["power_mean"])/scaling["power_std"]) @ scaling["pca_vectors"]


@torch.inference_mode()
def extract(cfg, output, manifest, scaling, models):
    density = SmoothDensity(**cfg["density"]).cuda()
    geoframe = load_geoframe(cfg)
    directory = output / "embeddings"
    directory.mkdir(exist_ok=True)
    for split in cfg["splits"]:
        q_np, metadata = load_split(output, manifest, split)
        q = torch.tensor(q_np[:, 0], device="cuda")
        del q_np
        outputs = {"density_pca":power_pca(density, q, scaling).cpu().numpy()}
        for kind, model in models.items():
            outputs[kind] = encode_moments(model, q).cpu().numpy()
        values = []
        for shard in manifest["shards"]:
            if shard["split"] != split:
                continue
            raw = np.load(output / "data" / (shard["stem"]+".clouds.npy"), mmap_mode="r")
            flat = raw.reshape(-1, *raw.shape[2:])
            for start in range(0, len(flat), cfg["inference_batch_size"]):
                cloud = torch.tensor(flat[start:start+cfg["inference_batch_size"], 0, :79], device="cuda") * cfg["cutoff_factor"]
                points = torch.cat((cloud.new_zeros(len(cloud), 1, 3), cloud), 1)
                values.append(geoframe.encoder.forward_features(points).cpu().numpy())
        outputs["geoframe_vicreg"] = np.concatenate(values)
        for kind, values in outputs.items():
            np.save(directory / f"{kind}_{split}.npy", values)
        np.savez(directory / f"{split}_metadata.npz", **metadata)
        print(f"Extracted matched static embeddings: {split}, {len(q)}", flush=True)
    del geoframe


def spectrum(z):
    z = z.double()
    c = z-z.mean(0)
    eig = torch.linalg.eigvalsh(c.T@c/(len(c)-1)).clamp_min(0)
    p = eig/eig.sum()
    positive = p[p>0]
    return dict(effective_rank=float(torch.exp(-(positive*positive.log()).sum())),
                total_variance=float(eig.sum()), largest_pc_fraction=float(p[-1]))


def structure_fit(train, labels, val, val_labels, test, test_labels):
    classes, counts = np.unique(labels, return_counts=True)
    y = torch.tensor(labels, device="cuda")
    classes_t = torch.tensor(classes, device="cuda")
    target = (y[:, None] == classes_t).double()
    weight = (target @ torch.tensor(len(y)/(len(classes)*counts), device="cuda", dtype=torch.float64)).sqrt()
    mean, std = train.mean(0), train.std(0)
    std = std.clamp_min(std.median()*.01)
    x = torch.cat(((train-mean)/std, torch.ones(len(train), 1, device="cuda")), 1).double()
    xv = torch.cat(((val-mean)/std, torch.ones(len(val), 1, device="cuda")), 1).double()
    xt = torch.cat(((test-mean)/std, torch.ones(len(test), 1, device="cuda")), 1).double()
    gram = (x*weight[:, None]).T@(x*weight[:, None])
    rhs = (x*weight[:, None]).T@(target*weight[:, None])
    best = None
    for alpha in (.1, 10., 1000.):
        ridge = torch.eye(gram.shape[0], device="cuda", dtype=torch.float64)*alpha
        ridge[-1, -1] = 0
        coefficients = torch.linalg.solve(gram+ridge, rhs)
        pv = classes[(xv@coefficients).argmax(1).cpu().numpy()]
        score = f1_score(val_labels, pv, average="macro", zero_division=0)
        if best is None or score > best[0]:
            best = (score, alpha, coefficients)
    predictions = classes[(xt@best[2]).argmax(1).cpu().numpy()]
    saved = dict(mean=mean.cpu(), std=std.cpu(), coefficients=best[2].cpu(), classes=torch.tensor(classes))
    metrics = dict(alpha=best[1], validation_macro_f1=float(best[0]),
                   test_macro_f1=float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
                   test_balanced_accuracy=float(balanced_accuracy_score(test_labels, predictions)),
                   classes=classes.tolist(), confusion=confusion_matrix(test_labels, predictions, labels=classes).tolist())
    return metrics, saved, predictions


def structural_analysis(cfg, output):
    if json.loads((output / "labels_status.json").read_text())["state"] != "complete":
        raise RuntimeError("Corrected PTM labels must finish before structural analysis")
    data_manifest = json.loads((output / "data/manifest.json").read_text())
    # Refresh evaluation metadata after the independently run assay correction.
    for split in cfg["splits"]:
        _, metadata = load_split(output, data_manifest, split)
        np.savez(output / "embeddings" / f"{split}_metadata.npz", **metadata)
    metadata = {split:dict(np.load(output / "embeddings" / f"{split}_metadata.npz")) for split in cfg["splits"]}
    report = {}
    for name in ("density_pca", "power_mlp", "mace_product", "geoframe_vicreg"):
        z = {split:torch.tensor(np.load(output / "embeddings" / f"{name}_{split}.npy"), device="cuda") for split in cfg["splits"]}
        metrics, probe, predictions = structure_fit(z["train"], metadata["train"]["labels"], z["val"], metadata["val"]["labels"], z["test"], metadata["test"]["labels"])
        torch.save(probe, output / "embeddings" / f"{name}_structure_probe.pt")
        by_material = {}
        for mi, material in enumerate(("Al", "Mg", "Ta")):
            mask = metadata["test"]["material"] == mi
            by_material[material] = dict(**spectrum(z["test"][mask]),
                ptm_macro_f1=float(f1_score(metadata["test"]["labels"][mask], predictions[mask], average="macro", zero_division=0)))
        report[name] = dict(structure=metrics, by_material=by_material)
        print(f"Structure {name}: macro F1={metrics['test_macro_f1']:.4f}", flush=True)
    write_json(output / "structure_metrics.json", report)
    # Correct stale assay counts in the training bookkeeping; no labels entered its loss.
    training = json.loads((output / "training_data.json").read_text())
    for split in ("train", "val"):
        training[f"{split}_ptm_counts"] = {str(k):int(v) for k,v in zip(*np.unique(metadata[split]["labels"], return_counts=True))}
    training["ptm_correction"] = "Evaluation labels recomputed after removing OVITO only_selected; model training unchanged."
    write_json(output / "training_data.json", training)


@torch.inference_mode()
def continuity(cfg, output, scaling, models):
    root = ROOT / cfg["continuity_root"]
    endpoints = torch.tensor(np.load(root / "inputs.npz")["endpoints"][:, :, 1:], device="cuda")/cfg["cutoff_factor"]
    refinement = dict(np.load(root / "vicreg_best_refinement.npz"))
    records = json.loads((root / "paths.json").read_text())
    density = SmoothDensity(**cfg["density"]).cuda()
    changes = {name:[] for name in ("density_pca", *models)}
    for iteration in range(len(refinement["iteration"])):
        alpha = torch.tensor(np.stack((refinement["left"][iteration], refinement["right"][iteration]), 1), device="cuda")
        points = endpoints[:, 0, None] + alpha[:, :, None, None]*(endpoints[:, 1, None]-endpoints[:, 0, None])
        q = density(points.flatten(0, 1))
        z = {"density_pca":power_pca(density, q, scaling)}
        z.update({name:model.forward_moments(q) for name, model in models.items()})
        for name, value in z.items():
            value = value.reshape(len(endpoints), 2, -1)
            changes[name].append((value[:, 1]-value[:, 0]).norm(dim=-1).cpu().numpy())
    arrays = {name:np.asarray(values) for name,values in changes.items()}
    arrays["geoframe_vicreg"] = refinement["native_jump"][:, :, 0]
    arrays["input_rms_A"] = refinement["input_rms_A"]
    np.savez(output / "continuity.npz", **arrays)
    report = {}
    for name in arrays:
        if name == "input_rms_A":
            continue
        train = np.load(output / "embeddings" / f"{name}_train.npy")
        material_ids = np.load(output / "embeddings/train_metadata.npz")["material"]
        report[name] = {}
        for mi, material in enumerate(("Al", "Mg", "Ta")):
            mask = np.array([r["material"] == material for r in records])
            scale = np.sqrt(train[material_ids == mi].var(0).sum())
            report[name][material] = dict(median_L2_at_iteration10=float(np.median(arrays[name][10, mask])),
                median_variance_scaled_L2_at_iteration10=float(np.median(arrays[name][10, mask])/scale),
                median_step10_over_step0=float(np.median(arrays[name][10, mask]/arrays[name][0, mask])))
    write_json(output / "continuity_metrics.json", dict(results=report,
               protocol="Replay selected GFv2 frame-boundary refinements on fixed 256-candidate finite clouds. New encoder uses a smooth radius and all in-support candidates, GF reference retains its original fixed patches. This is continuity evidence, not an isolated receptive-field comparison."))
    print("Completed existing frame-boundary continuity probes", flush=True)


@torch.inference_mode()
def history_features(cfg, output, manifest, scaling, models):
    directory = output / "history"
    directory.mkdir(exist_ok=True)
    h = cfg["temporal"]["history_frames"]
    density = SmoothDensity(**cfg["density"]).cuda()
    transport = TensorTransport(cfg["density"]["max_ell"]).cuda().float()
    static_offset = {split:0 for split in cfg["splits"]}
    buckets = {split:{} for split in cfg["splits"]}
    meta = {split:{key:[] for key in ("current", "future", "material", "frame", "labels", "nonaffine_A2", "center_id", "source")} for split in cfg["splits"]}
    for source_index, shard in enumerate(manifest["shards"]):
        split = shard["split"]
        prefix = output / "data" / shard["stem"]
        q = torch.tensor(np.load(str(prefix)+".moments.npy")[:, :, 0], device="cuda")
        c, t, r, d = q.shape
        cross = torch.tensor(np.load(str(prefix)+".cross.npy"), device="cuda")
        saved = dict(np.load(str(prefix)+".metadata.npz"))
        # All candidate states observe exactly the same h actual frames.
        end = torch.arange(h-1, t, device="cuda")
        ids = end[:, None] - torch.arange(h-1, -1, -1, device="cuda")
        windows = q[:, ids].reshape(-1, h, r, d)
        rows = torch.arange(c*t, device="cuda").reshape(c, t)[:, ids].reshape(-1, h)
        start = static_offset[split]
        static_offset[split] += c*t
        static = {name:torch.tensor(np.load(output / "embeddings" / f"{name}_{split}.npy", mmap_mode="r")[start:start+c*t], device="cuda")
                  for name in ("density_pca", "power_mlp", "mace_product", "geoframe_vicreg")}
        states = {name:values[rows[:, -1]] for name,values in static.items()}
        states["mace_linear_history"] = static["mace_product"][rows].flatten(1)
        for tau in cfg["temporal"]["tau_ps"]:
            a = np.exp(-cfg["temporal"]["cadence_ps"]/tau)
            z = static["mace_product"][rows[:, 0]]
            for k in range(1, h):
                z = a*z+(1-a)*static["mace_product"][rows[:, k]]
            states[f"invariant_ema_tau{tau}"] = z
        methods = [("untransported", 0.) , ("kabsch", 0.)] + [("smooth", e) for e in cfg["temporal"]["epsilon"]]
        for method, epsilon in methods:
            matrices = None
            if method != "untransported":
                # CUDA's batched eigensolver rejects the 46,464-matrix Ta shard
                # in this environment. Bound the solve batch explicitly.
                matrix = torch.cat([motion_matrix(chunk, method, epsilon)
                                    for chunk in cross.flatten(0, 1).split(4096)])
                matrices = transport.matrices(matrix)
            for tau in cfg["temporal"]["tau_ps"]:
                a = np.exp(-cfg["temporal"]["cadence_ps"]/tau)
                memory = windows[:, 0]
                for k in range(1, h):
                    if matrices is not None:
                        memory = transport(memory, [block[rows[:, k]] for block in matrices])
                    memory = a*memory+(1-a)*windows[:, k]
                states[f"{method}_eps{epsilon}_tau{tau}"] = encode_moments(models["mace_product"], memory)
        # Equal forecast origin set at all horizons; leave future frames unused by states.
        valid = (end+max(cfg["temporal"]["horizons_frames"]) < t).cpu().numpy()
        for name, value in states.items():
            a = value.reshape(c, len(end), -1).cpu().numpy()
            buckets[split].setdefault(name, []).append(a[:, valid].reshape(-1, a.shape[-1]))
            # Save all rolling states for temporal drift/transition analysis.
            np.save(directory / f"{shard['stem']}.{name}.npy", a)
        pc = static["density_pca"].reshape(c, t, -1)[..., :32]
        current_ids = end[torch.tensor(valid, device="cuda")]
        meta[split]["current"].append(pc[:, current_ids].flatten(0, 1).cpu().numpy())
        meta[split]["future"].append(torch.stack([pc[:, current_ids+lag] for lag in cfg["temporal"]["horizons_frames"]], 2).flatten(0, 1).cpu().numpy())
        shape = (c, len(current_ids))
        time_ids = current_ids.cpu().numpy()
        meta[split]["material"].append(np.full(shape, ("Al", "Mg", "Ta").index(shard["material"])).ravel())
        meta[split]["source"].append(np.full(shape, source_index).ravel())
        meta[split]["frame"].append(np.broadcast_to(saved["frames"][time_ids], shape).ravel())
        meta[split]["center_id"].append(np.broadcast_to(saved["center_ids"][:, None], shape).ravel())
        meta[split]["labels"].append(saved["labels"][:, time_ids].ravel())
        meta[split]["nonaffine_A2"].append(saved["nonaffine_A2"][:, time_ids].ravel())
        print(f"History states: {shard['stem']}", flush=True)
    for split in cfg["splits"]:
        for name, chunks in buckets[split].items():
            np.save(directory / f"{split}.{name}.npy", np.concatenate(chunks))
        np.savez(directory / f"{split}.targets.npz", **{key:np.concatenate(chunks) for key,chunks in meta[split].items()})
    write_json(directory / "manifest.json", dict(state="complete", candidates=list(buckets["train"]),
               target="First 32 training-fit density-power PCA coordinates, fixed across every candidate.",
               history_frames=h, direct_horizons_ps=[lag*.1 for lag in cfg["temporal"]["horizons_frames"]],
               note="Fixed-length rolling histories initialized at their first observation; no future coordinates used for state construction. No autonomous rollout claim."))


def ridge_forecasts(cfg, output):
    directory = output / "history"
    manifest = json.loads((directory / "manifest.json").read_text())
    targets = {split:dict(np.load(directory / f"{split}.targets.npz")) for split in cfg["splits"]}
    ymean = torch.tensor(targets["train"]["current"].mean(0), device="cuda")
    ystd = torch.tensor(targets["train"]["current"].std(0), device="cuda")
    y = {split:(torch.tensor(targets[split]["future"], device="cuda")-ymean)/ystd for split in cfg["splits"]}
    present = {split:(torch.tensor(targets[split]["current"], device="cuda")-ymean)/ystd for split in cfg["splits"]}
    report = {}
    for name in manifest["candidates"]:
        x = {split:torch.tensor(np.load(directory / f"{split}.{name}.npy"), device="cuda") for split in cfg["splits"]}
        mean, std = x["train"].mean(0), x["train"].std(0)
        std = std.clamp_min(std.median()*.01)
        for split in x:
            material = torch.nn.functional.one_hot(torch.tensor(targets[split]["material"], device="cuda"), 3)
            x[split] = torch.cat(((x[split]-mean)/std, material, torch.ones(len(x[split]), 1, device="cuda")), 1).double()
        gram = x["train"].T@x["train"]
        rhs = x["train"].T@y["train"].flatten(1).double()
        best = None
        for alpha in cfg["temporal"]["ridge_alpha"]:
            diagonal = torch.eye(gram.shape[0], device="cuda", dtype=torch.float64)*alpha
            diagonal[-1, -1] = 0
            coefficient = torch.linalg.solve(gram+diagonal, rhs)
            prediction = (x["val"]@coefficient).reshape_as(y["val"])
            score = (prediction-y["val"]).square().mean().item()
            if best is None or score < best[0]:
                best = score, alpha, coefficient
        predicted = (x["test"]@best[2]).reshape_as(y["test"])
        metrics = dict(alpha=best[1], validation_mse=best[0], by_material={})
        for mi, material in enumerate(("Al", "Mg", "Ta")):
            mask = targets["test"]["material"] == mi
            mse = (predicted[mask]-y["test"][mask]).square().mean((0, 2))
            persistence = (present["test"][mask, None]-y["test"][mask]).square().mean((0, 2))
            train_mask = targets["train"]["material"] == mi
            material_mean = y["train"][train_mask].mean(0)
            mean_mse = (material_mean-y["test"][mask]).square().mean((0, 2))
            metrics["by_material"][material] = dict(mse=mse.cpu().tolist(), persistence_mse=persistence.cpu().tolist(),
                  material_mean_mse=mean_mse.cpu().tolist(), skill_vs_material_mean=(1-mse/mean_mse).cpu().tolist(),
                  skill_vs_persistence=(1-mse/persistence).cpu().tolist())
        report[name] = metrics
        torch.save(dict(mean=mean.cpu(), std=std.cpu(), coefficient=best[2].cpu(), target_mean=ymean.cpu(),
                        target_std=ystd.cpu()), directory / f"{name}.ridge.pt")
        print(f"Forecast {name}: validation MSE {best[0]:.5f}", flush=True)
    # Family hyperparameters are chosen using validation only.
    families = {"static": [name for name in report if name in ("density_pca", "power_mlp", "mace_product", "geoframe_vicreg")],
                "invariant_ema":[name for name in report if name.startswith("invariant_ema")],
                "untransported":[name for name in report if name.startswith("untransported")],
                "kabsch":[name for name in report if name.startswith("kabsch")],
                "smooth":[name for name in report if name.startswith("smooth")],
                "linear_history":["mace_linear_history"]}
    selected = {family:min(names, key=lambda name:report[name]["validation_mse"]) for family,names in families.items()}
    write_json(output / "forecast_metrics.json", dict(results=report, selected_by_validation=selected,
               horizons_ps=[lag*.1 for lag in cfg["temporal"]["horizons_frames"]],
               note="Direct linear forecasts of common fixed 32D density embedding. Ridge and family settings selected on validation; test used only for reporting."))


def temporal_metrics(cfg, output, manifest):
    history_manifest = json.loads((output / "history/manifest.json").read_text())
    metadata_train = np.load(output / "embeddings/train_metadata.npz")
    names = history_manifest["candidates"]
    report = {}
    h = cfg["temporal"]["history_frames"]
    for name in names:
        by_material = {}
        train_history = np.load(output / "history" / f"train.{name}.npy")
        train_ids = np.load(output / "history/train.targets.npz")["material"]
        for mi, material in enumerate(("Al", "Mg", "Ta")):
            stable_steps, changing_steps, all_steps, clouds = [], [], [], []
            train_events = []
            for shard in manifest["shards"]:
                if shard["material"] == material and shard["split"] == "train":
                    a = np.load(output / "data" / (shard["stem"]+".metadata.npz"))["nonaffine_A2"][:, 1:]
                    train_events.append(a.ravel())
            low, high = np.quantile(np.concatenate(train_events), [.5, .9])
            scale = train_history[train_ids == mi].var(0).sum()
            for shard in manifest["shards"]:
                if shard["material"] != material or shard["split"] != "test":
                    continue
                values = np.load(output / "history" / f"{shard['stem']}.{name}.npy")
                change = np.linalg.norm(np.diff(values.astype(np.float64), axis=1), axis=-1)/np.sqrt(scale)
                physical = np.load(output / "data" / (shard["stem"]+".metadata.npz"))["nonaffine_A2"][:, h:]
                all_steps.append(change.ravel())
                stable_steps.append(change[physical<=low])
                changing_steps.append(change[physical>=high])
                clouds.append(values.reshape(-1, values.shape[-1]))
            a, b = np.concatenate(stable_steps), np.concatenate(changing_steps)
            # Count availability explicitly; a source can have no events above its training threshold.
            by_material[material] = dict(p95_scaled_step=float(np.quantile(np.concatenate(all_steps), .95)),
                low_nonaffine_count=len(a), high_nonaffine_count=len(b),
                low_nonaffine_mean_step=float(a.mean()) if len(a) else None,
                high_nonaffine_mean_step=float(b.mean()) if len(b) else None,
                **spectrum(torch.tensor(np.concatenate(clouds), device="cuda")))
        report[name] = by_material
    write_json(output / "temporal_metrics.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("extract", "continuity", "structure", "history", "forecast", "temporal", "remaining", "all"), required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = ROOT / cfg["output"]
    manifest = json.loads((output / "data/manifest.json").read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status = dict(state="running", stage=args.stage, pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "evaluation_status.json", status)
    try:
        scaling, models = load_models(cfg, output)
        if args.stage == "all":
            stages = ["extract", "continuity", "structure", "history", "forecast", "temporal"]
        elif args.stage == "remaining":
            stages = ["history", "forecast", "temporal"]
        else:
            stages = [args.stage]
        for stage in stages:
            status["stage"] = stage
            write_json(output / "evaluation_status.json", status)
            if stage == "extract":
                extract(cfg, output, manifest, scaling, models)
            elif stage == "continuity":
                continuity(cfg, output, scaling, models)
            elif stage == "structure":
                structural_analysis(cfg, output)
            elif stage == "history":
                history_features(cfg, output, manifest, scaling, models)
            elif stage == "forecast":
                ridge_forecasts(cfg, output)
            elif stage == "temporal":
                temporal_metrics(cfg, output, manifest)
        status.update(state="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "evaluation_status.json", status)


if __name__ == "__main__":
    main()
