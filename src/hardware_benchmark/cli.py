"""One command, explicit component selection, immutable run directories."""
import argparse
import json
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_table
from .common import REPO, metadata
from .cpu import CPUSettings
from .gpu import GPUSettings
from .storage import StorageSettings


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("storage", "cpu", "gpu", "all"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/benchmarks/hardware.json")
    parser.add_argument("--output", type=Path, help="New run directory; refuses an existing path")
    parser.add_argument("--storage-dir", type=Path, default=REPO,
                        help="Existing directory on the filesystem to measure (default: repository root)")
    parser.add_argument("--lammps", default="lmp", help="LAMMPS executable name or path")
    parser.add_argument("--launcher", default="", help="CPU launch prefix, e.g. 'mpiexec -n 8' or 'srun -n 8'; no shell")
    parser.add_argument("--mpi-ranks", type=int)
    parser.add_argument("--threads", type=int, help="LAMMPS OpenMP threads per rank")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workloads", nargs="+", choices=("pointnet", "mace", "forecast"))
    parser.add_argument("--smoke", action="store_true", help="Tiny functional check; not representative performance")
    return parser.parse_args(argv)


def settings_for(args):
    config = json.loads(args.config.read_text())
    if set(config) != {"storage", "cpu", "gpu"}:
        raise ValueError("Benchmark config must contain exactly storage, cpu, gpu sections")
    settings = dict(storage=StorageSettings(**config["storage"]), cpu=CPUSettings(**config["cpu"]),
                    gpu=GPUSettings(**config["gpu"]))
    if args.smoke:
        settings["storage"] = replace(settings["storage"], size_gib=0.008, block_mib=1,
                                      batch_size=4, batches=4, repeats=1)
        settings["cpu"] = replace(settings["cpu"], cells=4, warmup=2, steps=3, repeats=1)
        settings["gpu"] = replace(settings["gpu"], pointnet_batch=2, mace_batch=2, forecast_batch=2,
                                  mace_channels=8, history_steps=3, future_steps=2, forecast_width=32,
                                  warmup=1, steps=2, repeats=1)
    for option in ("mpi_ranks", "threads"):
        if getattr(args, option) is not None:
            settings["cpu"] = replace(settings["cpu"], **{option: getattr(args, option)})
    if args.workloads is not None:
        settings["gpu"] = replace(settings["gpu"], workloads=tuple(args.workloads))
    return settings


def _export(root, results):
    metrics = {}
    for component, result in results.items():
        if component == "gpu":
            metrics[component] = {name: item["metrics"] for name, item in result["workloads"].items()}
        else:
            metrics[component] = result["metrics"]
    write_metric_table(metrics, root, family="hardware_benchmark", name="hardware")


def main(argv=None):
    args = arguments(argv)
    components = ("storage", "cpu", "gpu") if args.suite == "all" else (args.suite,)
    from . import storage, cpu, gpu
    runners = dict(storage=storage, cpu=cpu, gpu=gpu)
    settings = settings_for(args)
    for name in components:
        runners[name].validate(settings[name])
    # Fail before expensive work if the reviewed calculation hashes have drifted.
    check_metric_docs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = (args.output or REPO / "output/hardware_benchmark" / stamp).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    technical = root / "technical"
    technical.mkdir()
    resolved = dict(suite=args.suite, smoke=args.smoke, settings={k: asdict(v) for k, v in settings.items()},
                    storage_dir=str(args.storage_dir),
                    lammps=args.lammps, launcher=args.launcher, device=args.device)
    write_json(technical / "config.json", resolved)
    report = dict(schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
                  status="running", smoke=args.smoke, metadata=metadata(), results={})
    write_json(technical / "results.json", report)
    (root / "README.md").write_text(
        "# Synthetic hardware benchmark\n\n"
        + ("This is a functional smoke run, not a hardware performance comparison.\n\n" if args.smoke else "")
        + "See `tables/hardware.csv` and `tables/METRICS.md` for results and exact definitions. "
        "`technical/results.json` retains status, raw timings, hardware/software metadata and checks. "
        "`technical/config.json` records all workload sizes and launch settings. "
        "No datasets or checkpoints are used.\n")
    try:
        for name in components:
            if name == "storage":
                result = storage.run(settings[name], args.storage_dir)
            elif name == "cpu":
                result = cpu.run(settings[name], technical, args.lammps, args.launcher)
            else:
                result = gpu.run(settings[name], args.device)
            report["results"][name] = result
            write_json(technical / "results.json", report)
            _export(root, report["results"])
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", failed_component=name, error=f"{type(error).__name__}: {error}")
        (technical / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(technical / "results.json", report)
        print(f"Benchmark {report['status']}: {root}", flush=True)
    return 0
