"""Matched temporal readouts, protocol verification, and motion diagnostics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.metrics import f1_score
import torch
from src.models.encoders.smooth_density import motion_matrix
from experiments.smooth_temporal_encoder_20260905.evaluate import structure_fit
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def verify_protocol(cfg, output, manifest):
    report = dict(splits={}, center_pools_disjoint=True, all_targets_match_observed_future=True)
    for split in cfg["splits"]:
        base = dict(np.load(output / "embeddings" / f"{split}_metadata.npz"))
        target = dict(np.load(output / "history" / f"{split}.targets.npz"))
        pca = np.load(output / "embeddings" / f"density_pca_{split}.npy")[:, :32]
        lookup = {(int(s), int(c), int(t)):i for i,(s,c,t) in enumerate(zip(base["source"], base["center_id"], base["frame"]))}
        origins = list(zip(target["source"], target["center_id"], target["frame"]))
        current = np.array([lookup[(int(s), int(c), int(t))] for s,c,t in origins])
        if not np.array_equal(pca[current], target["current"]):
            raise RuntimeError(f"Current target order mismatch: {split}")
        for j,lag in enumerate(cfg["temporal"]["horizons_frames"]):
            future = np.array([lookup[(int(s), int(c), int(t+lag))] for s,c,t in origins])
            if not np.array_equal(pca[future], target["future"][:, j]):
                raise RuntimeError(f"Future target order/lag mismatch: {split}/{lag}")
        report["splits"][split] = dict(origins=len(current), first_origin_ps=float(target["frame"].min()*.1),
               last_origin_ps=float(target["frame"].max()*.1), last_target_ps=float((target["frame"].max()+max(cfg["temporal"]["horizons_frames"]))* .1))
    for material, snapshot in {(s["material"], s["snapshot"]) for s in manifest["shards"]}:
        pools = [np.load(output / "data" / (s["stem"]+".metadata.npz"))["center_ids"] for s in manifest["shards"] if s["material"]==material and s["snapshot"]==snapshot]
        for i in range(len(pools)):
            for j in range(i):
                if len(np.intersect1d(pools[i], pools[j])):
                    raise RuntimeError(f"Center-ID split overlap: {material}/{snapshot}")
    write_json(output / "protocol_verification.json", report)


def motion_diagnostics(cfg, output, manifest):
    report = {}
    for material in ("Al", "Mg", "Ta"):
        angles, condition = [], []
        for shard in manifest["shards"]:
            if shard["split"] != "test" or shard["material"] != material:
                continue
            cross = np.load(output / "data" / (shard["stem"]+".cross.npy"))[:, 1:].reshape(-1, 3, 3)
            for values in torch.tensor(cross, device="cuda").split(4096):
                r = motion_matrix(values, "kabsch", 0.)
                angles.append(torch.rad2deg(torch.acos(((r.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))).cpu().numpy())
                singular = torch.linalg.svdvals(values.double())
                condition.append((singular[:, -1]/singular[:, 0]).cpu().numpy())
        a, c = np.concatenate(angles), np.concatenate(condition)
        report[material] = dict(rotation_median_deg=float(np.median(a)), rotation_p95_deg=float(np.quantile(a,.95)),
                                cross_moment_sigma_min_over_max_min=float(c.min()), cross_moment_sigma_min_over_max_median=float(np.median(c)))
    write_json(output / "motion_diagnostics.json", report)


def event_metrics(predicted, actual, cadence):
    delays, misses, anticipations, events = [], 0, 0, 0
    for p,y in zip(predicted, actual):
        for t in range(3, len(y)-5):
            if y[t] == y[t-1] or not np.all(y[t-3:t] == y[t-1]) or not np.all(y[t:t+3] == y[t]):
                continue
            events += 1
            anticipations += int(p[t-1] == y[t])
            matches = [k for k in range(5) if p[t+k] == y[t] and p[t+k+1] == y[t]]
            if matches:
                delays.append(matches[0]*cadence)
            else:
                misses += 1
    return events, misses, anticipations, delays


def temporal_readouts(cfg, output, manifest):
    meta = {split:dict(np.load(output / "history" / f"{split}.targets.npz")) for split in cfg["splits"]}
    selected = json.loads((output / "forecast_metrics.json").read_text())["selected_by_validation"]
    names = ["density_pca", "mace_product", "geoframe_vicreg", "mace_linear_history"] + [selected[k] for k in ("invariant_ema", "untransported", "kabsch", "smooth")]
    report = {}
    h = cfg["temporal"]["history_frames"]
    for name in names:
        states = {split:torch.tensor(np.load(output / "history" / f"{split}.{name}.npy"), device="cuda") for split in cfg["splits"]}
        metrics, probe, _ = structure_fit(states["train"], meta["train"]["labels"], states["val"], meta["val"]["labels"], states["test"], meta["test"]["labels"])
        torch.save(probe, output / "history" / f"{name}.structure_probe.pt")
        report[name] = dict(structure=metrics)
    neural = json.loads((output / "neural_temporal_metrics.json").read_text())
    for row in neural:
        report[f"{row['mode']}_seed{row['seed']}"] = dict(structure=row["structure"])
    for name in report:
        if "_seed" in name:
            directory = output / "temporal_models" / name
            all_states = np.load(directory / "test.all_rolling_states.npy", mmap_mode="r")
            probe = torch.load(directory / "structure_probe.pt", map_location="cuda", weights_only=True)
        else:
            probe = torch.load(output / "history" / f"{name}.structure_probe.pt", map_location="cuda", weights_only=True)
        offset = 0
        counts = [0, 0, 0]
        delays = []
        prediction_arrays, labels = [], []
        for shard in manifest["shards"]:
            if shard["split"] != "test":
                continue
            c, t = shard["centers"], shard["frames"]-h+1
            if "_seed" in name:
                values = all_states[offset:offset+c*t].reshape(c, t, -1)
                offset += c*t
            else:
                values = np.load(output / "history" / f"{shard['stem']}.{name}.npy")
            z = torch.tensor(values.reshape(-1, values.shape[-1]), device="cuda")
            x = torch.cat(((z-probe["mean"])/probe["std"], z.new_ones(len(z),1)),1).double()
            prediction = probe["classes"][(x@probe["coefficients"]).argmax(1)].cpu().numpy().reshape(c,t)
            label = np.load(output / "data" / (shard["stem"]+".metadata.npz"))["labels"][:, h-1:]
            event_count, missed, anticipated, delay = event_metrics(prediction, label, .1)
            counts = [a+b for a,b in zip(counts,(event_count, missed, anticipated))]
            delays.extend(delay)
            prediction_arrays.append(prediction.ravel())
            labels.append(label.ravel())
        report[name]["all_rolling_macro_f1"] = float(f1_score(np.concatenate(labels), np.concatenate(prediction_arrays), average="macro", zero_division=0))
        report[name]["persistent_ptm_changes"] = dict(events=counts[0], missed_within_0_5ps=counts[1],
                  predicted_new_label_before_event=counts[2], median_delay_ps=float(np.median(delays)) if delays else None,
                  mean_delay_ps=float(np.mean(delays)) if delays else None)
    write_json(output / "temporal_readouts.json", dict(results=report,
               protocol="PTM probes fit only matched forecast-origin training states. Events require three unchanged PTM labels before and after a change; detection requires two consecutive predicted new labels. This is a diagnostic of structural-assay response, not ground-truth phase-transition timing."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = ROOT / cfg["output"]
    manifest = json.loads((output / "data/manifest.json").read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    verify_protocol(cfg, output, manifest)
    motion_diagnostics(cfg, output, manifest)
    temporal_readouts(cfg, output, manifest)
    print("Protocol, motion and temporal readout assessment complete", flush=True)


if __name__ == "__main__":
    main()
