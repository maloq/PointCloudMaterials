"""Matched early/late diagnostics using immutable historical encoder producers.

This arm concerns the original shared-pretraining_v2 snapshot runs. It is an
Al-native dynamic *all-phase* assay, not a liquid-only crystallization benchmark.
Native anchor/future identity is verified before any temporal metric is exported.
"""
from __future__ import annotations

from collections import defaultdict
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from src.data.structural_pretraining.prepare import file_hash, save_json
from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import resolve_path
from .metrics import conditional_rank, fit_physical_metric, participation


def sampled_rows(manifest, train_limit, selection_limit, seed):
    """Outcome-blind, approximately source-balanced native-Al sampling."""
    if not 2 <= train_limit <= 2048 or not 2 <= selection_limit <= 1024:
        raise ValueError("Checkpoint assay limits are train2..2048, selection2..1024")
    groups = {role: defaultdict(list) for role in ("train", "selection")}
    index = 0
    for record in manifest["shards"]:
        role = record["task"]["split"]
        if (record["material"] == "Al" and not record["static"]
                and record["source"].startswith("native_") and role in groups):
            groups[role][record["source"]].extend(range(index, index + record["anchors"]))
        index += record["anchors"]
    result = {}
    for role, limit in (("train", train_limit), ("selection", selection_limit)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(role == "selection")]))
        sources = sorted(groups[role])
        if len(sources) < 2:
            raise ValueError(f"Insufficient independent native-Al sources for {role}")
        queues = {source: rng.permutation(groups[role][source]).tolist() for source in sources}
        selected = []
        while len(selected) < min(limit, sum(map(len, groups[role].values()))):
            for source in sources:
                if queues[source] and len(selected) < limit:
                    selected.append(queues[source].pop())
        result[role] = sorted(selected)
    train_sources = set(groups["train"])
    if train_sources & set(groups["selection"]):
        raise ValueError("Training and selection sources overlap")
    source_by_id = {s["id"]: s for s in manifest["sources"]}
    train_roots = {source_by_id[s]["lineage"] for s in train_sources}
    selected_roots = {source_by_id[s]["lineage"] for s in groups["selection"]}
    if train_roots & selected_roots:
        raise ValueError("Training and selection sources share simulation ancestry")
    return result


def _native_extract():
    """Standalone extraction body, executed only under the frozen native src."""
    import json
    import resource
    from pathlib import Path
    import sys
    import time
    import numpy as np
    import torch
    from src.data.structural_pretraining.batches import Release, collate, move
    from src.data.structural_pretraining.prepare import file_hash
    from src.models.encoders.structural import StructuralModel

    record_path, stage = sys.argv[1:]
    record = json.loads(Path(record_path).read_text())
    if Path.cwd().resolve() != Path(record["producer_code"]).resolve():
        raise ValueError("Native extraction must run from its checkpoint producer")
    for name, expected in record["producer_files"].items():
        path = Path(name) if Path(name).is_absolute() else Path.cwd() / name
        if file_hash(path) != expected:
            raise ValueError(f"Historical producer changed: {path}")
    if file_hash(Path(record["release"]) / "manifest.json") != record["manifest_sha256"]:
        raise ValueError("Historical release manifest changed")
    models = []
    for checkpoint in record["checkpoints"]:
        if file_hash(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {checkpoint['path']}")
        state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        if state["identity"] != record["identity"] or state["step"] != checkpoint["step"]:
            raise ValueError("Historical checkpoint identity or step differs")
        model = StructuralModel(record["architecture"])
        model.load_state_dict(state["model"], strict=True)
        models.append(model.eval())
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    torch.set_num_threads(2)
    release = Release(record["release"])
    release.max_graph_bytes = 512 * 2**20
    if stage == "preflight":
        sample_indices = [record["indices"][role][0] for role in ("train", "selection")]
        for view in ("anchor", "future"):
            samples = [release.observation(i, view, False, record["architecture"] == "mace")
                       for i in sample_indices]
            batch = collate(samples, record["architecture"])
            for key, value in batch.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Native preflight nonfinite batch field: {key}")
        print(json.dumps(dict(state="preflight_passed", architecture=record["architecture"],
                              rows={k: len(v) for k, v in record["indices"].items()},
                              native_anchor_future_collation=True)), flush=True)
        return
    if stage != "extract":
        raise ValueError(stage)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Historical checkpoint extraction requires an allocated CUDA GPU")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.85, record["memory_limit_GiB"] * 2**30 / total))
    indices = record["indices"]["train"] + record["indices"]["selection"]
    roles = np.asarray(["train"] * len(record["indices"]["train"]) +
                       ["selection"] * len(record["indices"]["selection"]))
    used = {release.rows[i][0] for i in indices}
    for shard in release.manifest["shards"]:
        if shard["task"]["id"] not in used:
            continue
        for key, expected in shard["hashes"].items():
            path = release.root / "shards" / shard["task"]["id"] / f"{key}.npy"
            if file_hash(path) != expected:
                raise ValueError(f"Native observation array changed: {path}")
    sources, atoms, steps, lags, temperatures = [], [], [], [], []
    target, future_target = [], []
    source_records = {s["id"]: s for s in release.manifest["sources"]}
    for index in indices:
        name, row, shard = release.rows[index]
        a = release.arrays[name]
        current, following = map(int, a["views"][row, [2, 3]])
        if current < 0 or following < 0:
            raise ValueError("Missing native anchor or successor")
        atom = int(a["center_ids"][current])
        if int(a["center_ids"][following]) != atom:
            raise ValueError("Native future is not the same atom")
        delta_steps = int(a["steps"][following]) - int(a["steps"][current])
        lag = delta_steps * source_records[shard["source"]]["timestep_fs"] / 1000.
        if lag <= 0 or not np.isclose(lag, float(a["times"][3] - a["times"][2]), rtol=0, atol=1e-9):
            raise ValueError("Native future timestep and recorded physical lag disagree")
        sources.append(shard["source"])
        atoms.append(atom)
        steps.append(int(a["steps"][current]))
        lags.append(lag)
        temperatures.append(record["temperatures"][shard["source"]])
        target.append(np.asarray(a["physical"][current]))
        future_target.append(np.asarray(a["physical"][following]))
    common = dict(index=np.asarray(indices), role=roles, source=np.asarray(sources),
                  atom=np.asarray(atoms), step=np.asarray(steps), lag_ps=np.asarray(lags),
                  temperature=np.asarray(temperatures), physical=np.asarray(target),
                  future_physical=np.asarray(future_target))
    for model, checkpoint in zip(models, record["checkpoints"], strict=True):
        model = model.cuda()
        destination = Path(checkpoint["features"])
        result = {}
        started = time.monotonic()
        with torch.inference_mode():
            for view, suffix in (("anchor", ""), ("future", "_future")):
                encoded, projected = [], []
                for start in range(0, len(indices), record["batch_size"]):
                    batch_indices = indices[start:start + record["batch_size"]]
                    samples = [release.observation(i, view, False, record["architecture"] == "mace")
                               for i in batch_indices]
                    batch = move(collate(samples, record["architecture"]), "cuda:0")
                    # The original v2 producer used eager FP32, without autocast.
                    z = model.encoder(batch)
                    q = model.projector(z)
                    if z.shape != (len(samples), 128) or q.shape != (len(samples), 64):
                        raise ValueError("Historical encoder/projector export shape changed")
                    if not torch.isfinite(z).all() or not torch.isfinite(q).all():
                        raise FloatingPointError("Nonfinite historical embeddings")
                    encoded.append(z.cpu().numpy())
                    projected.append(q.cpu().numpy())
                result["encoder" + suffix] = np.concatenate(encoded)
                result["projector" + suffix] = np.concatenate(projected)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".building.npz")
        np.savez(temporary, **common, **result)
        temporary.replace(destination)
        print(json.dumps(dict(state="extracted", stage=checkpoint["stage"], step=checkpoint["step"],
                              seconds=time.monotonic() - started, features=str(destination))), flush=True)
        model.cpu()
        torch.cuda.empty_cache()


def preflight(config, root):
    """Freeze inputs and validate exact historical schemas on CPU before queuing."""
    import torch
    root = Path(root).resolve()
    technical = root / "technical" / "checkpoints"
    technical.mkdir(parents=True, exist_ok=True)
    settings = config.get("checkpoint_sampling", {})
    sampling = dict(train=int(settings.get("train", 2048)),
                    selection=int(settings.get("selection", 1024)),
                    seed=int(settings.get("seed", 20260922)))
    records = []
    source_code = "import sys\nsys.path.insert(0, '.')\n" + inspect.getsource(_native_extract) + "\n_native_extract()\n"
    extraction_script = technical / "native_extract.py"
    if extraction_script.exists() and extraction_script.read_text() != source_code:
        raise ValueError("Checkpoint extraction implementation changed in an existing run")
    extraction_script.write_text(source_code)
    for pair in config.get("checkpoint_pairs", []):
        if pair["architecture"] not in ("mace", "gatr"):
            raise ValueError(f"Unsupported checkpoint architecture: {pair['architecture']}")
        folder = technical / pair["name"]
        folder.mkdir(exist_ok=True)
        producer = resolve_path(pair["producer_code"]).resolve()
        checkpoints, identities = [], []
        for stage in ("early", "late"):
            original = resolve_path(pair[stage]).resolve()
            path = folder / f"{stage}.pt"
            expected = file_hash(original)
            if not path.exists():
                shutil.copy2(original, path)
            if file_hash(path) != expected:
                raise ValueError(f"Checkpoint source changed since freeze: {original}")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            identities.append(saved["identity"])
            checkpoints.append(dict(stage=stage, path=str(path), source=str(original),
                                    sha256=expected, step=int(saved["step"]),
                                    features=str(folder / f"{stage}.npz")))
        identity = identities[0]
        if identities[1] != identity or checkpoints[0]["step"] >= checkpoints[1]["step"]:
            raise ValueError(f"{pair['name']} is not a chronological same-identity pair")
        native = identity["config"]
        if (identity["protocol"] != "shared_pretraining_v2" or native["phase"] != "structural"
                or native["architecture"] != pair["architecture"] or native["history_frames"] != 1):
            raise ValueError("This diagnostic implements only native v2 structural snapshots")
        release = resolve_path(native["release"]).resolve()
        manifest = json.loads((release / "manifest.json").read_text())
        if manifest["identity"] != identity["data"]:
            raise ValueError("Checkpoint data identity differs from its native release")
        indices = sampled_rows(manifest, **{k + "_limit": sampling[k] for k in ("train", "selection")}, seed=sampling["seed"])
        cohort = json.loads(resolve_path(manifest["config"]["cohort"]).read_text())
        temperatures = {f"native_{s['id']}": s["temperature_K"] for s in cohort["sources"]}
        record = dict(name=pair["name"], architecture=pair["architecture"], producer_code=str(producer),
                      identity=identity, producer_files=identity["implementation"]["files"],
                      release=str(release), manifest_sha256=file_hash(release / "manifest.json"),
                      indices=indices, temperatures=temperatures, checkpoints=checkpoints,
                      batch_size=int(settings.get("batch_size", 4)),
                      memory_limit_GiB=float(settings.get("memory_limit_GiB", 32)),
                      native_precision="eager_fp32", extractor_sha256=file_hash(extraction_script))
        if records:
            previous = json.loads(records[0].read_text())
            if previous["manifest_sha256"] != record["manifest_sha256"] or previous["indices"] != indices:
                raise ValueError("Checkpoint architectures must use the identical native population")
        if not 1 <= record["batch_size"] <= 8 or not 0 < record["memory_limit_GiB"] <= 40:
            raise ValueError("Historical inference bound is batch1..8 and GPUmemory<=40GiB")
        path = folder / "record.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError(f"Frozen checkpoint assay changed: {pair['name']}")
        save_json(path, record)
        with (folder / "preflight.log").open("w") as log:
            subprocess.run([sys.executable, str(extraction_script), str(path), "preflight"],
                           cwd=producer, env={**os.environ, "PYTHONPATH": str(producer)},
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        records.append(path)
    return records


def _source_mean(values, source):
    return float(np.mean([np.mean(values[source == s]) for s in np.unique(source)]))


def summarize(features, alpha=1.):
    """Training-fitted linear information and native same-atom drift diagnostics."""
    train, selection = features["role"] == "train", features["role"] == "selection"
    if min(train.sum(), selection.sum()) < 2:
        raise ValueError("Checkpoint assay needs training and held-out selection rows")
    source = features["source"][selection]
    blocks = dict(radial=(0, 32), pair=(32, 64), angular=(64, 80), moments=(80, 85))
    result = {}
    for space in ("encoder", "projector"):
        z = features[space]
        lag_difference = features[space + "_future"][selection] - z[selection]
        train_trace = participation(z[train])["trace"]
        # 2*train trace is expected squared separation of independent training states.
        drift = np.square(lag_difference).sum(1)
        rank = participation(z[selection])
        result[space] = dict(selection_rank=rank, train_rank=participation(z[train]),
                             same_atom_drift_squared=_source_mean(drift, source),
                             normalized_same_atom_drift=None if train_trace == 0 else _source_mean(drift, source)/(2*train_trace),
                             per_temperature={str(k): v for k, v in conditional_rank(
                                 z[selection], features["temperature"][selection]).items()})
        for target in ("physical", "future_physical"):
            y = features[target]
            _, predicted, fit = fit_physical_metric(z[train], y[train], z[selection], alpha=alpha)
            scale = np.asarray(fit["target_scale"])
            mean = np.asarray(fit["target_center"])
            actual = (y[selection] - mean) / scale
            baseline = np.square(actual)
            errors = np.square(predicted - actual)
            output = {}
            for block, (start, stop) in blocks.items():
                model_mse = _source_mean(errors[:, start:stop].mean(1), source)
                constant_mse = _source_mean(baseline[:, start:stop].mean(1), source)
                output[block] = dict(mse=model_mse, train_constant_mse=constant_mse,
                                     skill=None if constant_mse == 0 else 1-model_mse/constant_mse)
                if target == "future_physical":
                    persistence = ((features["physical"][selection]-y[selection])/scale)**2
                    persistence_mse = _source_mean(persistence[:, start:stop].mean(1), source)
                    output[block]["persistence_mse"] = persistence_mse
                    output[block]["gain_over_persistence"] = None if persistence_mse == 0 else 1-model_mse/persistence_mse
            result[space][target] = output
    return result


def run(config, root):
    """Run optional checkpoint pairs after CPU preflight; sequential GPU subprocesses."""
    root = Path(root).resolve()
    records = preflight(config, root)
    if not records:
        return {}
    all_results = {}
    timeout = int(config.get("checkpoint_sampling", {}).get("timeout_seconds", 7200))
    if not 60 <= timeout <= 14400:
        raise ValueError("Checkpoint pair timeout must be between60 and14400seconds")
    script = root / "technical/checkpoints/native_extract.py"
    for path in records:
        record = json.loads(path.read_text())
        with (path.parent / "inference.log").open("w") as log:
            subprocess.run([sys.executable, str(script), str(path), "extract"],
                           cwd=record["producer_code"],
                           env={**os.environ, "PYTHONPATH": record["producer_code"]},
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        baseline = None
        for checkpoint in record["checkpoints"]:
            with np.load(checkpoint["features"]) as archive:
                features = dict(archive)
            population = {k: features[k] for k in ("index", "role", "source", "atom", "step", "lag_ps", "physical", "future_physical")}
            if baseline is not None:
                for key in population:
                    np.testing.assert_array_equal(population[key], baseline[key], err_msg=f"Early/late {key} differs")
            baseline = population
            metrics = summarize(features, alpha=float(config.get("checkpoint_sampling", {}).get("ridge_alpha", 1.)))
            name = f"{record['name']}-{checkpoint['stage']}"
            all_results[name] = dict(step=checkpoint["step"], features_sha256=file_hash(checkpoint["features"]),
                                     train_n=int((features["role"] == "train").sum()),
                                     selection_n=int((features["role"] == "selection").sum()),
                                     lag_ps=np.unique(features["lag_ps"]).tolist(), metrics=metrics)
            write_metric_table(metrics, root, family="liquid_geometry", name=f"checkpoint-{name}")
    save_json(root / "technical/checkpoints/results.json", all_results)
    _plot_endpoints(all_results, records, root)
    return all_results


def _plot_endpoints(results, records, root):
    """Display endpoints only; no unobserved intermediate trajectory is implied."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [("Selection participation rank", lambda m: m["selection_rank"]["rank"]),
              ("Current physical skill versus training constant", lambda m: np.mean([v["skill"] for v in m["physical"].values()])),
              ("Future physical skill versus training constant", lambda m: np.mean([v["skill"] for v in m["future_physical"].values()])),
              ("Same-atom squared drift / (2 × train covariance trace)", lambda m: m["normalized_same_atom_drift"])]
    for path in records:
        name = json.loads(path.read_text())["name"]
        for space, marker in (("encoder", "o"), ("projector", "s")):
            endpoints = [results[f"{name}-{stage}"] for stage in ("early", "late")]
            for axis, (title, read) in zip(axes.flat, panels, strict=True):
                values = [read(e["metrics"][space]) for e in endpoints]
                if any(v is None for v in values):
                    continue
                axis.scatter([e["step"] for e in endpoints], values, marker=marker,
                             label=f"{name} {space}")
                axis.set_title(title, fontsize=10)
                axis.set_xlabel("Training update")
                axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Matched historical checkpoints: Al native dynamics, all phases\n"
                 "Training-fitted probes; whole-source selection holdout; two endpoints per run")
    fig.tight_layout()
    plots = Path(root) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / "checkpoint-endpoints.png", dpi=160)
    plt.close(fig)
