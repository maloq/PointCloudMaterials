"""Measurement validity, isolation and actual synthetic model training."""
import json
import csv
import shlex
from dataclasses import asdict
import numpy as np
import pytest

from src.hardware_benchmark import cpu, storage
from src.hardware_benchmark.cli import arguments, cpu_plan, main
from src.hardware_benchmark.common import summarize
from src.hardware_benchmark.gpu import GPUSettings, build_task, _check_training


def test_median_throughput_is_not_mean_of_trial_rates():
    result = summarize([1.0, 2.0, 9.0], 100, "items")
    assert result["items_per_second"] == 50
    assert result["seconds_samples"] == [1.0, 2.0, 9.0]
    with pytest.raises(ValueError, match="positive finite"):
        summarize([float("nan")], 100, "items")


def test_storage_materializes_exact_float16_values_and_only_removes_owned_files(tmp_path):
    sentinel = tmp_path / "existing-data.bin"
    sentinel.write_bytes(b"research data")
    settings = storage.StorageSettings(size_gib=0.0001, points=8, batch_size=3, batches=2, repeats=2)
    result = storage.run(settings, tmp_path)
    records = result["shape"][0]
    values = np.random.default_rng(settings.seed).uniform(-9.2, 9.2, (records, 8, 3)).astype("<f2")
    indices = np.random.default_rng(settings.seed + 1).integers(records, size=(2, 3))
    expected = sum(float(values[index].astype(np.float32).sum(dtype=np.float64)) for index in indices)
    assert result["sampled_checksum"] == expected
    assert result["file_bytes"] == values.nbytes
    assert result["metrics"]["mmap_warm"]["units_per_trial"] == 6
    assert result["metrics"]["write_generate_hash_fsync"]["trials"] == 2
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"research data"


def test_storage_cleanup_on_failed_read(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("injected I/O failure")
    monkeypatch.setattr(storage, "_stream_read", fail)
    with pytest.raises(OSError, match="injected I/O"):
        storage.run(storage.StorageSettings(size_gib=0.0001, repeats=1), tmp_path)
    assert not list(tmp_path.iterdir())


def test_storage_detects_corruption(tmp_path, monkeypatch):
    class WrongDigest:
        def hexdigest(self):
            return "corrupt"
    monkeypatch.setattr(storage.hashlib, "file_digest", lambda *a: WrongDigest())
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        storage.run(storage.StorageSettings(size_gib=0.0001, repeats=1), tmp_path)
    assert not list(tmp_path.iterdir())


def loop_text(settings):
    procs, atoms = settings.mpi_ranks * settings.threads, 4 * settings.cells**3
    return "\n".join(f"Loop time of {seconds} on {procs} procs for {steps} steps with {atoms} atoms"
                     for seconds, steps in [(99, settings.warmup)] + [(2, settings.steps)] * settings.repeats)


def test_lammps_excludes_warmup_and_enforces_actual_execution_shape():
    settings = cpu.CPUSettings(mpi_ranks=2, threads=4)
    text = loop_text(settings)
    assert cpu.parse_loop_times(text, settings) == [2.0] * 3
    for broken in (text.replace("8 procs", "1 procs"), text.replace("32000 atoms", "31999 atoms"),
                   text.replace("2 on", "nan on"), text + "\n500 32000 nan -4 0 -4 0\n"):
        with pytest.raises(RuntimeError):
            cpu.parse_loop_times(broken, settings)
    with pytest.raises(RuntimeError, match="Expected 4"):
        cpu.parse_loop_times("", settings)


def test_cpu_failed_subprocess_is_reported_with_diagnostics(tmp_path):
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="LAMMPS exited"):
        main(["cpu", "--smoke", "--lammps", "/bin/false", "--output", str(root)])
    report = json.loads((root / "technical/results.json").read_text())
    assert report["status"] == "failed"
    assert report["failed_component"] == "cpu"
    assert (root / "technical/failure.txt").is_file()
    assert (root / "technical/lammps.in").is_file()
    assert "**INCOMPLETE: cpu failed.**" in (root / "RESULTS.md").read_text()
    assert "No measurements completed." in (root / "RESULTS.md").read_text()


def test_cli_exports_frozen_contract_and_refuses_overwrite(tmp_path, capsys):
    root = tmp_path / "results"
    argv = ["storage", "--smoke", "--storage-dir", str(tmp_path), "--output", str(root)]
    assert main(argv) == 0
    assert json.loads((root / "technical/results.json").read_text())["status"] == "complete"
    assert (root / "tables/hardware.csv").is_file()
    assert "POSIX_FADV_DONTNEED" in (root / "tables/METRICS.md").read_text()
    assert json.loads((root / "technical/metric-contract.json").read_text())["family"] == "hardware_benchmark"
    printed = capsys.readouterr().out
    final_table = (root / "RESULTS.md").read_text()
    assert final_table in printed
    assert "**SMOKE RUN" in final_table
    assert "Median throughput | Min–max throughput | ms/update" in final_table
    with (root / "tables/summary.csv").open() as stream:
        rows = {row["benchmark"]: row for row in csv.DictReader(stream)}
    raw = json.loads((root / "technical/results.json").read_text())["results"]["storage"]["metrics"]
    assert float(rows["storage.sequential_warm"]["median"]) == raw["sequential_warm"]["MiB_per_second"]
    assert rows["storage.mmap_warm"]["unit"] == "clouds/s"
    assert rows["storage.sequential_warm"]["ms_per_update"] == ""
    with pytest.raises(FileExistsError):
        main(argv)


@pytest.mark.parametrize("name", ["pointnet", "mace", "forecast"])
def test_actual_models_train_on_synthetic_inputs_without_cuda_or_checkpoints(name):
    import torch
    settings = GPUSettings(points=8, pointnet_batch=2, mace_batch=2, forecast_batch=2,
                           mace_channels=4, embedding_dim=8, history_steps=3, future_steps=2,
                           forecast_width=16)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, loss_fn, detail = build_task(name, settings, torch.device("cpu"))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        before = [p.detach().clone() for p in model.parameters()]
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        _check_training(model, [loss.detach()], name)
        assert detail["batch_size"] == 2
        assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        if name in ("pointnet", "mace"):
            for part in ("encoder", "objective"):
                assert any(p.grad is not None and p.grad.abs().sum().item() > 0
                           for p in model[part].parameters())
    finally:
        torch.set_num_threads(previous_threads)


def test_gpu_never_substitutes_cpu(monkeypatch):
    import torch
    from src.hardware_benchmark.gpu import run
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        run(GPUSettings(), "cuda:0")


def test_radius_graph_excludes_self_edges_when_distance_rounding_is_nonzero(monkeypatch):
    import torch
    from src.hardware_benchmark.gpu import radius_edges
    distances = torch.tensor([[[0.001, 2.4, 4.8], [2.4, 0.002, 2.4], [4.8, 2.4, 0.003]]])
    monkeypatch.setattr(torch, "cdist", lambda *args: distances)
    edges, counts = radius_edges(torch.zeros(1, 3, 3), 4.0)
    assert counts.tolist() == [4]
    assert edges[0].tolist() == [[0, 1], [1, 0], [1, 2], [2, 1]]


@pytest.mark.parametrize("settings", [cpu.CPUSettings(cells=1), cpu.CPUSettings(threads=0)])
def test_invalid_lammps_config_rejected_before_launch(settings):
    with pytest.raises(ValueError):
        cpu.input_text(settings)


def test_summary_preserves_example_counts_and_reverses_duration_range():
    from src.hardware_benchmark.report import summary_rows
    # Two updates, 128 original examples/update, and uneven trial durations.
    metrics = summarize([2.0, 4.0, 8.0], 256, "examples")
    metrics["step_ms"] = 2000.0
    gpu_settings = GPUSettings(pointnet_batch=128, steps=2)
    report = dict(results=dict(gpu=dict(workloads=dict(pointnet=dict(batch_size=128, metrics=metrics)))))
    rows = summary_rows(report, dict(settings=dict(gpu=asdict(gpu_settings))))
    assert len(rows) == 1
    row = rows[0]
    assert row["unit"] == "examples/s"
    assert (row["median"], row["minimum"], row["maximum"]) == (64, 32, 128)
    assert row["ms_per_update"] == 2000
    assert "batch=128" in row["workload"] and "2 views/example" in row["workload"]


def test_one_invocation_keeps_cpu_cases_and_logs_separate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(8)))
    called = []

    def fake_run(settings, technical, executable, launcher):
        called.append((settings.mpi_ranks, technical, launcher))
        (technical / "lammps.log").write_text("retained case log")
        return dict(metrics=summarize([1, 2, 4], settings.steps, "steps"),
                    atoms=4 * settings.cells**3, version="LAMMPS fixture", launcher=shlex.split(launcher))

    monkeypatch.setattr(cpu, "run", fake_run)
    root = tmp_path / "sweep"
    main(["cpu", "--smoke", "--cpu-ranks", "1", "2", "--output", str(root)])
    assert [(rank, launcher) for rank, _, launcher in called] == [(1, ""), (2, "mpiexec -n 2")]
    assert called[0][1] != called[1][1]
    assert all((folder / "lammps.log").is_file() for _, folder, _ in called)
    raw = json.loads((root / "technical/results.json").read_text())
    assert raw["status"] == "complete"
    assert set(raw["results"]["cpu"]["runs"]) == {"r1_t1", "r2_t1"}
    with (root / "tables/summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["unit"] == "MD steps/s" and row["ms_per_update"] == "" for row in rows)
    assert "1 MPI ranks x 1 threads" in rows[0]["workload"]
    assert "2 MPI ranks x 1 threads" in rows[1]["workload"]
    assert capsys.readouterr().out.count("# Hardware benchmark results") == 1


def test_cpu_sweep_checks_capacity_and_expands_explicit_launcher(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(8)))
    args = arguments(["cpu", "--cpu-ranks", "1", "4", "--launcher", "srun -n {ranks}"])
    plan = cpu_plan(args, cpu.CPUSettings(threads=2))
    assert [launcher for _, launcher in plan] == ["srun -n 1", "srun -n 4"]
    with pytest.raises(ValueError, match="affinity allows 8"):
        cpu_plan(arguments(["cpu", "--cpu-ranks", "1", "9"]), cpu.CPUSettings())
    with pytest.raises(ValueError, match="requires .ranks."):
        cpu_plan(arguments(["cpu", "--cpu-ranks", "1", "2", "--launcher", "mpiexec -n 2"]), cpu.CPUSettings())
