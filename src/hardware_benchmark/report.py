"""One consistent final table, printed and saved with workload/hardware context."""
import csv
import json

from src.experiment_runner.metric_docs import snapshot_metric_docs

STORAGE_LABELS = {
    "write_generate_hash_fsync": "Storage: generation + hash + write + fsync",
    "write_syscalls_fsync": "Storage: write calls + fsync (diagnostic)",
    "sequential_eviction_advised": "Storage: sequential read, eviction advised",
    "sequential_warm": "Storage: sequential read, warm",
    "mmap_eviction_advised": "Storage: random mmap, eviction advised",
    "mmap_warm": "Storage: random mmap, warm",
}
GPU_LABELS = {"pointnet": "GPU: PointNet", "mace": "GPU: MACE", "forecast": "GPU: forecast transformer"}


def _row(key, label, unit, metrics, rate_key, details):
    # A range of rates reverses the order of the duration extrema.
    return dict(benchmark=key, label=label, workload=details, trials=metrics["trials"], unit=unit,
                median=metrics[rate_key], minimum=metrics["units_per_trial"] / metrics["seconds_max"],
                maximum=metrics["units_per_trial"] / metrics["seconds_min"], ms_per_update=None)


def summary_rows(report, config):
    rows = []
    settings = config["settings"]
    storage = report["results"].get("storage")
    if storage is not None:
        for name, metrics in storage["metrics"].items():
            is_mmap = name.startswith("mmap_")
            unit, rate = ("clouds/s", "clouds_per_second") if is_mmap else ("MiB/s", "MiB_per_second")
            details = f"{storage['file_bytes'] / 2**30:.6f} GiB; float16; {storage['shape'][1]} points/cloud"
            if is_mmap:
                details += f"; batch={settings['storage']['batch_size']}; {settings['storage']['batches']} batches"
            rows.append(_row(f"storage.{name}", STORAGE_LABELS[name], unit, metrics, rate, details))
    for name, result in report["results"].get("cpu", {}).get("runs", {}).items():
        values = result["settings"]
        details = (f"{result['atoms']:,} atoms; {values['mpi_ranks']} MPI ranks x {values['threads']} threads; "
                   f"{values['steps']} steps/trial; warmup={values['warmup']}; lj/cut")
        rows.append(_row(f"cpu.{name}", "CPU: LAMMPS", "MD steps/s", result["metrics"], "steps_per_second", details))
    for name, result in report["results"].get("gpu", {}).get("workloads", {}).items():
        values = settings["gpu"]
        details = f"batch={result['batch_size']}"
        if name == "forecast":
            details += (f"; history/future={values['history_steps']}/{values['future_steps']}; "
                        f"dim={values['embedding_dim']}; width={values['forecast_width']}")
        else:
            details += f"; {values['points']} points; 2 views/example"
        if name == "mace":
            details += f"; channels={values['mace_channels']}; " + ("cuEquivariance" if values["mace_accelerated"] else "e3nn")
        row = _row(f"gpu.{name}", GPU_LABELS[name], "examples/s", result["metrics"], "examples_per_second", details)
        row["ms_per_update"] = result["metrics"]["step_ms"]
        rows.append(row)
    return rows


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(report, config, rows):
    meta = report["metadata"]
    lines = ["# Hardware benchmark results", "", f"Run: {report['run_id']}; status: **{report['status']}**.",
             f"Started: {report['started_at']}; finished: {report['finished_at']}.",
             f"CPU: {meta['cpu_model']}; {len(meta['cpu_affinity'])} allowed logical CPUs. Host: `{meta['host']}`.",
             f"Python: {meta['python_version']}; NumPy: {meta['packages']['numpy']}."]
    storage = report["results"].get("storage")
    if storage is not None:
        lines.append(f"Storage target: `{storage['target']}`; exact file size: {storage['file_bytes']:,} bytes.")
        fs = storage["filesystem"]
        description = "unavailable (see metadata)."
        if fs.get("returncode") == 0:
            description = "; ".join(f"{item['fstype']} on {item['source']} mounted at {item['target']}"
                                    for item in json.loads(fs["stdout"])["filesystems"])
        lines.append("Filesystem: " + _cell(description))
    for name, result in report["results"].get("cpu", {}).get("runs", {}).items():
        lines.append(f"CPU {name}: {_cell(result['version'])}; launcher: `{_cell(result['launcher'])}`.")
    gpu = report["results"].get("gpu")
    if gpu is not None:
        values = config["settings"]["gpu"]
        driver = gpu["driver"]
        versions = ", ".join(sorted(set(driver["stdout"].splitlines()))) if driver.get("returncode") == 0 else "unavailable"
        lines += [f"GPU: {gpu['name']}; PyTorch: {gpu['torch_version']}; CUDA: {gpu['cuda_version']}; driver: {versions}.",
                  f"GPU timing: FP32; matmul={values['matmul_precision']}; eager; AdamW; "
                  f"{values['steps']} updates/trial; warmup={values['warmup']}; host threads={values['host_threads']}."]
        if "mace" in gpu["workloads"]:
            lines.append(f"MACE software: mace-torch={meta['packages']['mace-torch']}; e3nn={meta['packages']['e3nn']}; "
                         f"cuEquivariance-torch={meta['packages']['cuequivariance-torch']}.")
    if report["smoke"]:
        lines += ["", "**SMOKE RUN: functional validation only, not a hardware performance measurement.**"]
    if report["status"] == "failed":
        lines += ["", f"**INCOMPLETE: {report['failed_component']} failed.** {_cell(report['error'])}"]
    lines += ["", "Rates use median trial duration. Ranges are observed min–max throughput, not confidence intervals.",
              "GPU examples count original inputs, not paired views; updates include forward, backward and AdamW.", "",
              "| Benchmark | Workload | Trials | Median throughput | Min–max throughput | ms/update |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        latency = "—" if row["ms_per_update"] is None else f"{row['ms_per_update']:.2f}"
        lines.append(f"| {row['label']} | {_cell(row['workload'])} | {row['trials']} | {row['median']:,.2f} {row['unit']} | "
                     f"{row['minimum']:,.2f}–{row['maximum']:,.2f} {row['unit']} | {latency} |")
    if not rows:
        lines += ["", "No measurements completed."]
    lines += ["", "Buffered eviction-advised reads are not guaranteed cold-device reads. Warm reads include caching.",
              "The primary write result includes generation and hashing; write-call-only throughput is diagnostic.",
              "Background load is not controlled. Host load averages at start (1/5/15 min): " + _cell(meta["load_average"]) + ".",
              "", "Full precision: `tables/summary.csv`. Raw trials and hardware/software details: `technical/results.json`.",
              "Resolved settings: `technical/config.json`. Definitions and implementation hashes: "
              "`tables/METRICS.md` and `technical/metric-contract.json`."]
    return "\n".join(lines) + "\n"


def export(root, report, config):
    snapshot_metric_docs(root, "hardware_benchmark")
    rows = summary_rows(report, config)
    fields = ("benchmark", "label", "workload", "trials", "unit", "median", "minimum", "maximum", "ms_per_update")
    with (root / "tables/summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    text = render(report, config, rows)
    (root / "RESULTS.md").write_text(text)
    return text
