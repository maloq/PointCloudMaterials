"""Relate local geometry to the completed, paired relaxed-context forecasts.

This is a secondary, descriptive bridge: two encoders, four paired heads, and
whole simulation sources as uncertainty units. It never loads context tensors
onto a GPU, retrains a predictor, or treats heads as independent encoder fits.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.data.structural_pretraining.prepare import file_hash
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from .metrics import fit_transform, neighbor_metrics, participation


DOMAINS = ("observed", "relaxed")
METHODS = ("direct", "ar_mse", "mixture", "diffusion")
ORDER_NAMES = ("q4", "q6", "w4", "w6", "qbar6", "coherence", "density", "coordination")


def _checked(path, expected):
    path = Path(path)
    actual = file_hash(path)
    if actual != expected:
        raise ValueError(f"Latest bridge input checksum differs: {path}: {actual} != {expected}")
    return path


def _csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effective_rank(values):
    """Shared covariance participation ratio; a singleton stays undefined."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return None
    return participation(x)["rank"]


def source_moments(values, sources):
    """Training-only equal-source population moments."""
    values = np.asarray(values, np.float64)
    unique = np.unique(sources)
    means = np.stack([values[sources == sid].mean(0) for sid in unique])
    mean = means.mean(0)
    variance = np.stack([((values[sources == sid] - mean) ** 2).mean(0) for sid in unique]).mean(0)
    keep = variance > variance.max() * 1e-12
    if not keep.any():
        raise ValueError("Latest bridge has no training variation")
    return mean, np.sqrt(variance), keep


def retrieve(train_z, test_z, train_y, test_y, train_source, test_source, train_temp, test_temp, k):
    """Same-temperature training neighbors, shared physical targets and baseline.

    Scaling is fitted on training sources only. The random-neighbor denominator
    samples a training source equally, then a row within source. Nearest-neighbor
    selection itself uses the actual training observation population.
    """
    tz, qz, zm = fit_transform(train_z, test_z, mode="standardized")
    ty, qy, ym = fit_transform(train_y, test_y, mode="standardized")
    reconstruction = np.empty(len(qz))
    neighbor = np.empty(len(qz))
    random = np.empty(len(qz))
    for temp in np.unique(test_temp):
        candidates = np.flatnonzero(train_temp == temp)
        queries = np.flatnonzero(test_temp == temp)
        if len(candidates) < k:
            raise ValueError(f"Only {len(candidates)} training candidates at {temp} K for k={k}")
        mean, sd, _ = source_moments(ty[candidates], train_source[candidates])
        random[queries] = (((qy[queries] - mean) ** 2) + sd ** 2).mean(1)
        result = neighbor_metrics(tz[candidates], qz[queries], ty[candidates], qy[queries],
                                  train_source[candidates], test_source[queries], k=k)
        neighbors = ty[candidates[result["neighbor_indices"]]]
        reconstruction[queries] = ((neighbors.mean(1) - qy[queries]) ** 2).mean(1)
        neighbor[queries] = result["neighbor_target_mse"]
    return dict(reconstruction=reconstruction, neighbor=neighbor, random=random,
                embedding_channels=tz.shape[1], target_channels=ty.shape[1],
                feature_transform=zm, target_transform=ym)


def paired_association(x, y, temperatures, draws, seed):
    """Descriptive correlation after temperature centering, source bootstrap.

    Inputs have one paired difference per source. Recenter within each bootstrap
    temperature stratum; retain paired x/y source indices together.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    temperatures = np.asarray(temperatures)
    if len(x) < 4:
        return dict(spearman=None, lower=None, upper=None, valid_draws=0)
    groups = [np.flatnonzero(temperatures == t) for t in np.unique(temperatures)]

    def statistic(ix):
        a, b, t = x[ix].copy(), y[ix].copy(), temperatures[ix]
        for temperature in np.unique(t):
            mask = t == temperature
            a[mask] -= a[mask].mean()
            b[mask] -= b[mask].mean()
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(spearmanr(a, b).statistic)

    estimate = statistic(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    values = [statistic(np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])) for _ in range(draws)]
    values = [v for v in values if v is not None and np.isfinite(v)]
    interval = np.quantile(values, [.025, .975]).tolist() if values else [None, None]
    return dict(spearman=estimate, lower=interval[0], upper=interval[1], valid_draws=len(values))


def load_cohort(latest_root):
    """Read only current-center features and exact original row keys."""
    latest_root = resolve_path(str(latest_root))
    plan_path = latest_root / "technical/plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["pretraining_test_and_calibration_overlap"] or plan["relaxed_encoder"]["protected_overlap"]:
        raise ValueError("Latest bridge encoder ancestry overlaps a protected source role")
    reference = json.loads(resolve_path(plan["reuse_config"]["reference_plan"]).read_text())
    observed_cache = resolve_path(reference["structured_config"]["context_cache"])
    relaxed_cache = resolve_path(plan["reuse_config"]["context_cache"])
    corpus_cache = resolve_path(plan["config"]["cache"])
    parts = {key: [] for key in ("rows", "role", "temperature", "atom_id", "frame", "order", "label", "event", "observed", "relaxed", "cold_information")}
    receipts = []
    for source in plan["sources"]:
        sid = source["id"]
        folder = corpus_cache / str(sid)
        receipt = json.loads((folder / "complete.json").read_text())
        if receipt["identity"] != plan["cache_identity"]:
            raise ValueError(f"Original population identity differs for source {sid}")
        arrays = {}
        for name in ("risk", "order", "labels", "onset", "atom_ids"):
            path = _checked(folder / f"{name}.npy", receipt["hashes"][f"{name}.npy"])
            arrays[name] = np.load(path, mmap_mode="r")
        np.testing.assert_array_equal(arrays["atom_ids"], source["center_atom_ids"])
        ai, ci = np.where(arrays["risk"])
        frame = np.asarray(plan["anchors"])[ai]
        keep = np.array([str(f) in plan["observed_histories"][str(sid)] for f in frame])
        ai, ci, frame = ai[keep], ci[keep], frame[keep]
        old = observed_cache / str(sid)
        old_receipt = json.loads((old / "complete.json").read_text())
        if old_receipt["identity"] != reference["structured_identity"]:
            raise ValueError(f"Observed context identity differs for source {sid}")
        old_path = _checked(old / "mace_center.npy", old_receipt["files"]["mace_center.npy"])
        hot = np.load(old_path, mmap_mode="r")
        cold_folder = relaxed_cache / str(sid)
        cold_receipt = json.loads((cold_folder / "complete.json").read_text())
        if cold_receipt["identity"] != plan["structured_identity"]:
            raise ValueError(f"Relaxed context identity differs for source {sid}")
        cold_path = _checked(cold_folder / "observations.npz", cold_receipt["sha256"])
        with np.load(cold_path) as cold:
            lookup = {int(f): i for i, f in enumerate(cold["frames"])}
            fi = np.asarray([lookup[int(f)] for f in frame])
            parts["relaxed"].append(cold["features"][fi, ci, 0])
            parts["cold_information"].append(cold["information"][fi, ci])
        parts["observed"].append(np.asarray(hot[frame // 4, ci]))
        parts["rows"].append(np.column_stack((np.full(len(ai), sid), ai, ci)))
        parts["role"].append(np.full(len(ai), source.get("validation_role", source["split"])))
        parts["temperature"].append(np.full(len(ai), source["temperature_K"]))
        parts["atom_id"].append(np.asarray(arrays["atom_ids"][ci]))
        parts["frame"].append(frame)
        parts["order"].append(np.asarray(arrays["order"][ci, frame]))
        parts["label"].append(np.asarray(arrays["labels"][ci, frame]))
        parts["event"].append(np.minimum(arrays["onset"][ci] - frame - 1, 128))
        receipts.append(dict(source=sid, observed_sha256=old_receipt["files"]["mace_center.npy"],
                             relaxed_sha256=cold_receipt["sha256"], population_identity=receipt["identity"]))
    data = {key: np.concatenate(values) for key, values in parts.items()}
    if np.isin(data["label"], [1, 2, 3]).any() or (data["event"] < 0).any():
        raise ValueError("Latest cohort violates its original noncrystalline at-risk criterion")
    if data["observed"].shape != data["relaxed"].shape or data["observed"].shape[1] != 128:
        raise ValueError("Unexpected latest feature schema")
    if not all(np.isfinite(data[key]).all() for key in ("observed", "relaxed", "order", "cold_information")):
        raise ValueError("Nonfinite latest bridge observations")
    return plan, data, dict(plan_sha256=file_hash(plan_path), sources=receipts)


def run(config, root):
    """Write paired source tables, exact cohort and a method-matched scatter."""
    root = Path(root)
    technical = root / "technical/latest"
    technical.mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(exist_ok=True)
    (root / "plots").mkdir(exist_ok=True)
    latest_root = resolve_path(str(config["latest_root"]))
    plan, data, provenance = load_cohort(latest_root)
    k = int(config.get("latest_neighbors", 31))
    train, test = np.flatnonzero(data["role"] == "train"), np.flatnonzero(data["role"] == "test")
    sources = data["rows"][:, 0]
    if set(sources[train]) & set(sources[test]):
        raise ValueError("Latest bridge train/test source overlap")
    expected = json.loads((latest_root / "technical/population.json").read_text())
    for role, record in expected.items():
        if role == "training_sources_by_temperature":
            continue
        mask = data["role"] == role
        if int(mask.sum()) != record["windows"] or len(np.unique(sources[mask])) != record["sources"]:
            raise ValueError(f"Latest bridge population differs for {role}")
    retrieval = {}
    embedding_rows = []
    for domain in DOMAINS:
        result = retrieve(data[domain][train], data[domain][test], data["order"][train], data["order"][test],
                          sources[train], sources[test], data["temperature"][train], data["temperature"][test], k)
        retrieval[domain] = result
        for sid in np.unique(sources[test]):
            mask = sources[test] == sid
            ids = test[mask]
            embedding_rows.append(dict(domain=domain, source=int(sid), temperature=float(data["temperature"][ids[0]]),
                windows=len(ids), current_ptm_other_fraction=float((data["label"][ids] == 0).mean()),
                effective_rank=effective_rank(data[domain][ids]), order_effective_rank=effective_rank(data["order"][ids]),
                qbar6_variance=float(np.var(data["order"][ids, 4].astype(float))),
                physical_reconstruction_mse=float(result["reconstruction"][mask].mean()),
                physical_neighbor_mse=float(result["neighbor"][mask].mean()),
                random_neighbor_mse=float(result["random"][mask].mean()),
                physical_neighbor_ratio=float(result["neighbor"][mask].mean() / result["random"][mask].mean()),
                neighbors=k, retained_embedding_channels=result["embedding_channels"], retained_order_channels=result["target_channels"]))
    by_embedding = {(r["domain"], r["source"]): r for r in embedding_rows}
    forecasts = []
    forecast_provenance = {}
    queue = json.loads((latest_root / "technical/queue.json").read_text())
    if {(s["observation_domain"], s["method"]) for s in queue} != {(d, m) for d in DOMAINS for m in METHODS} or len(queue) != 8:
        raise ValueError("Expected exactly eight matched forecast fits")
    for spec in queue:
        folder = latest_root / "technical/runs" / spec["name"]
        if json.loads((folder / "status.json").read_text())["state"] != "complete":
            raise ValueError(f"Forecast incomplete: {spec['name']}")
        with np.load(folder / "predictions.npz") as p:
            np.testing.assert_array_equal(p["test_indices"], test)
            np.testing.assert_array_equal(p["test_event"], data["event"][test])
            cdf = np.asarray(p["test_cdf"], float)
        if cdf.shape != (len(test), 128) or not np.isfinite(cdf).all() or (np.diff(cdf, axis=1) < -1e-6).any():
            raise ValueError(f"Invalid forecast CDF: {spec['name']}")
        if (cdf < 0).any() or (cdf > 1).any():
            raise ValueError(f"Out-of-range CDF: {spec['name']}")
        event = data["event"][test]
        brier = ((cdf - (np.arange(128)[None] >= event[:, None])) ** 2).mean(1)
        reported = json.loads((folder / "metrics.json").read_text())
        replay = np.mean([brier[sources[test] == sid].mean() for sid in np.unique(sources[test])])
        np.testing.assert_allclose(replay, reported["dense_integrated_brier"], rtol=5e-7, atol=1e-8,
                                   err_msg=f"Dense Brier replay: {spec['name']}")
        risk = np.clip(cdf[:, 15], 1e-7, 1 - 1e-7)
        truth = event < 16
        logloss = -(truth * np.log(risk) + ~truth * np.log1p(-risk))
        for sid in np.unique(sources[test]):
            mask = sources[test] == sid
            forecasts.append(dict(domain=spec["observation_domain"], method=spec["method"], fit=spec["name"],
                source=int(sid), windows=int(mask.sum()), dense_brier=float(brier[mask].mean()),
                brier12=float(((cdf[mask, 15] - truth[mask]) ** 2).mean()), logloss12=float(logloss[mask].mean())))
        forecast_provenance[spec["name"]] = dict(predictions_sha256=file_hash(folder / "predictions.npz"),
                                                checkpoint_sha256=file_hash(folder / "best.pt"))
    by_forecast = {(r["domain"], r["method"], r["source"]): r for r in forecasts}
    deltas, associations = [], []
    for method in METHODS:
        for sid in np.unique(sources[test]):
            hot, cold = [by_embedding[d, int(sid)] for d in DOMAINS]
            hp, cp = [by_forecast[d, method, int(sid)] for d in DOMAINS]
            deltas.append(dict(method=method, source=int(sid), temperature=hot["temperature"], windows=hot["windows"],
                physical_neighbor_improvement=hot["physical_neighbor_ratio"] - cold["physical_neighbor_ratio"],
                physical_reconstruction_improvement=hot["physical_reconstruction_mse"] - cold["physical_reconstruction_mse"],
                rank_change=None if hot["effective_rank"] is None or cold["effective_rank"] is None else cold["effective_rank"] - hot["effective_rank"],
                dense_brier_improvement=hp["dense_brier"] - cp["dense_brier"],
                brier12_improvement=hp["brier12"] - cp["brier12"], logloss12_improvement=hp["logloss12"] - cp["logloss12"]))
        rows = [r for r in deltas if r["method"] == method]
        for x in ("physical_neighbor_improvement", "physical_reconstruction_improvement", "rank_change"):
            for y in ("dense_brier_improvement", "brier12_improvement", "logloss12_improvement"):
                valid = [r for r in rows if r[x] is not None and np.isfinite(r[x]) and np.isfinite(r[y])]
                associations.append(dict(method=method, geometry_metric=x, forecast_metric=y, sources=len(valid),
                    **paired_association([r[x] for r in valid], [r[y] for r in valid], [r["temperature"] for r in valid],
                                         int(config["bootstrap_draws"]), int(config["seed"]))))
    _csv(root / "tables/latest_embedding_by_source.csv", embedding_rows)
    _csv(root / "tables/latest_forecast_by_source.csv", forecasts)
    _csv(root / "tables/latest_paired_deltas.csv", deltas)
    _csv(root / "tables/latest_associations.csv", associations)
    np.savez_compressed(technical / "cohort.npz", **data, test_indices=test, train_indices=train,
                        **{f"{d}_{k}": v for d, result in retrieval.items() for k, v in result.items() if isinstance(v, np.ndarray)})
    provenance.update(forecasts=forecast_provenance, checkpoint_observed=plan["checkpoint_sha256"],
                      checkpoint_relaxed=plan["relaxed_encoder"]["sha256"], rows=len(sources), test_rows=len(test),
                      transforms={d: {k: retrieval[d][k] for k in ("feature_transform", "target_transform")} for d in DOMAINS},
                      caveat="Two encoders and four paired heads; source associations are descriptive, not independent encoder evidence. Context forecasts also use descriptor histories and spatial observations. At-risk excludes current/recent FCC/HCP/BCC; Other is not definitive liquid. Targets are shared original-MD order8; no cold-target advantage. Uncertainty conditions on fitted encoders, heads, training reference and one seed.")
    (technical / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    snapshot_metric_docs(root, "liquid_geometry")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        rows = [r for r in deltas if r["method"] == method]
        sc = ax.scatter([r["physical_neighbor_improvement"] for r in rows], [r["dense_brier_improvement"] for r in rows],
                        c=[r["temperature"] for r in rows], cmap="viridis", vmin=400, vmax=520, s=30)
        ax.axhline(0, color="0.7", lw=.7); ax.axvline(0, color="0.7", lw=.7)
        ax.set_title(method); ax.set_xlabel("Improvement in physical neighbor ratio"); ax.set_ylabel("Improvement in forecast Brier")
    fig.suptitle("Relaxed − observed comparison: one point per simulation source\nPositive values favor relaxed; four heads share the same two encoders", fontsize=11)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="Temperature (K)", fraction=.03, pad=.03)
    fig.savefig(root / "plots/latest_geometry_forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return dict(rows=len(sources), test_windows=len(test), test_sources=len(np.unique(sources[test])), encoders=2, heads_per_encoder=4,
                directory=str(technical), associations=associations)
