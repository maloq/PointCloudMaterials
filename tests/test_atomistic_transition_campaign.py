from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.nose_hoover_chain import IsotropicMTKNPT

from src.data_utils.synthetic.atomistic.artifacts import label_interface
from src.data_utils.synthetic.atomistic.config import potential_calculator_settings
from src.data_utils.synthetic.atomistic.generator import select_calculator
from src.data_utils.synthetic.atomistic.provenance import (
    TRANSITION_CAMPAIGN_MD_PRODUCER_FILES,
    _producer_code_provenance,
)
from src.data_utils.synthetic.atomistic.simulation import (
    set_maxwell_boltzmann_velocities,
)
from src.data_utils.synthetic.atomistic.transition_campaign import (
    analyze_transition_task,
    finalize_transition_campaign,
    run_analysis_worker,
    run_md_worker,
    run_transition_task,
)
from src.data_utils.synthetic.atomistic.transition_campaign_config import (
    load_content_bound_prepared_interface,
    load_transition_campaign_config,
)
from src.data_utils.synthetic.atomistic.transition_campaign_queue import (
    TransitionCampaignTask,
    campaign_rows,
    claim_analysis_task,
    claim_md_task,
    complete_analysis_task,
    complete_md_task,
    fail_task,
    initialize_transition_queue,
)
from src.data_utils.synthetic.atomistic.transition_generator import PreparedInterface
from src.data_utils.synthetic.atomistic.transition_config import load_transition_config
from src.data_utils.synthetic.atomistic.transition_resumable import (
    TransitionCheckpointStore,
    build_transition_mtk_dynamics,
    capture_mtk_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_CONFIG = (
    REPOSITORY_ROOT / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)


def _write_campaign(tmp_path: Path) -> Path:
    generator_raw = yaml.safe_load(GENERATOR_CONFIG.read_text(encoding="utf-8"))
    generator_raw["dataset_name"] = "small_transition_source"
    generator_raw["random_seeds"] = [1]
    generator_raw["potential"].update(
        {
            "device": "cpu",
            "enable_cueq": False,
            "enable_oeq": False,
            "compile_mode": None,
            "compile_fullgraph": False,
            "pad_num_atoms": 0,
            "pad_num_edges": 0,
            "neighbor_skin_A": 0.1,
        }
    )
    generator_raw["system"].update(
        {"repetitions": [2, 2, 2], "interface_half_width_A": 0.5}
    )
    generator_raw["validation"].update(
        {
            "maximum_force_eV_per_A": 100.0,
            "maximum_pressure_error_GPa": 1000.0,
            "maximum_temperature_error_K": 1000.0,
            "minimum_pair_distance_A": 1.0,
        }
    )
    generator_raw["output"]["root_dir"] = str(tmp_path / "source")
    source_generator = tmp_path / "source_generator.yaml"
    source_generator.write_text(yaml.safe_dump(generator_raw), encoding="utf-8")
    runtime_generator = tmp_path / "runtime_generator.yaml"
    runtime_generator.write_text(yaml.safe_dump(generator_raw), encoding="utf-8")

    transition = {
        "dataset_name": "small_queued_transition",
        "source_generator_config": str(source_generator),
        "source_dataset": str(tmp_path / "source"),
        "source_interface_environment": "replica_000_solid_liquid_interface",
        "source_frame_step": 0,
        "random_seeds": [11, 12],
        "sample_interval": 1,
        "analysis": {
            "profile_bins": 4,
            "profile_smoothing_bins": 1,
            "ptm_rmsd_cutoff": 0.1,
            "minimum_profile_contrast": 0.1,
            "minimum_velocity_fit_r_squared": 0.0,
            "rdf_cutoff_A": 4.0,
            "rdf_bins": 40,
        },
        "temperature_runs": [
            {
                "name": "temperature_0300K",
                "temperature_K": 300.0,
                "expected_direction": "growth",
                "equilibration_steps": 2,
                "production_steps": 3,
                "steady_state_start_step": 1,
                "steady_state_end_step": 3,
                "minimum_crystalline_fraction_change": 0.0,
            },
            {
                "name": "temperature_0600K",
                "temperature_K": 600.0,
                "expected_direction": "melting",
                "equilibration_steps": 2,
                "production_steps": 3,
                "steady_state_start_step": 1,
                "steady_state_end_step": 3,
                "minimum_crystalline_fraction_change": 0.0,
            },
        ],
        "output": {
            "root_dir": str(tmp_path / "campaign_output"),
            "overwrite": False,
            "save_extxyz": False,
            "create_visualizations": False,
        },
    }
    transition_path = tmp_path / "transition.yaml"
    transition_path.write_text(yaml.safe_dump(transition), encoding="utf-8")
    transition_config = load_transition_config(transition_path)
    source_root = transition_config.source_dataset
    interface_dir = source_root / transition_config.source_interface_environment
    interface_dir.mkdir(parents=True)
    source_atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    with (interface_dir / "trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=np.array([0], dtype=np.int64),
            positions_A=np.asarray([source_atoms.positions], dtype=np.float64),
            cell_vectors_A=np.asarray([source_atoms.cell.array], dtype=np.float64),
            volume_A3=np.asarray([source_atoms.get_volume()], dtype=np.float64),
        )
    (interface_dir / "metadata.json").write_text(
        json.dumps(
            {
                "intermediate_regions": [
                    {"definition": {"slab_bounds_fractional": [0.25, 0.75]}}
                ]
            }
        ),
        encoding="utf-8",
    )
    potential = transition_config.generator.potential
    calculator = {
        "source": "configured_mace_model",
        "identity": f"{potential.model_name}:{potential.sha256}:{potential.head}",
        "implementation_class": (
            "src.data_utils.synthetic.atomistic.calculator."
            "VerletSkinMACECalculator"
        ),
        "model_name": potential.model_name,
        "family": potential.family,
        "model_path": str(potential.model_path),
        "model_sha256": potential.sha256,
        "head": potential.head,
        "available_heads": [potential.head],
        "source_url": potential.source_url,
        "license_identifier": potential.license_identifier,
        "usage_mode": potential.usage_mode,
        "validation_report_path": None,
        "validation_report_sha256": None,
        "validation_report_type": None,
        "scientifically_qualified": potential.scientifically_qualified,
        "qualification_scope": None,
        "settings": potential_calculator_settings(potential),
    }
    (source_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "config": transition_config.generator.to_dict(),
                "potential_sha256": potential.sha256,
                "execution_provenance": {
                    "calculator": calculator,
                    "runtime": {
                        "python": "test",
                        "numpy": "test",
                        "ase": "test",
                        "torch": "test",
                        "platform": "test",
                        "machine": "test",
                        "mace_torch": "test",
                        "torch_cuda": "test",
                        "cudnn": "test",
                    },
                    "producer_code": _producer_code_provenance(),
                },
            }
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "transition_config": str(transition_path),
                "runtime_generator_config": str(runtime_generator),
                "execution": {"chunk_steps": 2, "checkpoint_retention": 2},
            }
        ),
        encoding="utf-8",
    )
    return campaign_path


def _prepared_interface() -> PreparedInterface:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    bounds = (0.25, 0.75)
    return PreparedInterface(
        atoms=atoms,
        labels=label_interface(atoms, bounds, interface_half_width_A=0.5),
        slab_bounds_fractional=bounds,
    )


def test_campaign_runtime_config_rejects_scientific_changes(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path)
    config = load_transition_campaign_config(path)
    assert config.execution.chunk_steps == 2
    runtime_path = Path(
        yaml.safe_load(path.read_text(encoding="utf-8"))["runtime_generator_config"]
    )
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    runtime["dynamics"]["timestep_fs"] = 2.0
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientific or model fields"):
        load_transition_campaign_config(path)


def test_campaign_source_evidence_rejects_replaced_prepared_source(
    tmp_path: Path,
) -> None:
    config = load_transition_campaign_config(_write_campaign(tmp_path))
    metadata_path = (
        config.transition.source_dataset
        / config.transition.source_interface_environment
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["intermediate_regions"][0]["definition"][
        "slab_bounds_fractional"
    ] = [0.2, 0.8]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content-bound campaign identity"):
        load_content_bound_prepared_interface(config)


def test_queue_is_disjoint_branch_replica_cartesian_product(tmp_path: Path) -> None:
    config = load_transition_campaign_config(_write_campaign(tmp_path))
    initialize_transition_queue(config, retry_failed=False)
    rows = campaign_rows(config)
    assert [row["run_name"] for row in rows] == [
        "temperature_0300K/replica_000",
        "temperature_0300K/replica_001",
        "temperature_0600K/replica_000",
        "temperature_0600K/replica_001",
    ]
    assert len({row["simulation_seed"] for row in rows}) == 4
    assert all(row["md_status"] == "queued" for row in rows)


def test_queue_initialization_rolls_back_metadata_and_tasks_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_transition_campaign_config(_write_campaign(tmp_path))
    import src.data_utils.synthetic.atomistic.transition_campaign_queue as queue_module

    expected = queue_module._expected_tasks(config)
    duplicate_seed = list(expected[1])
    duplicate_seed[-1] = expected[0][-1]
    monkeypatch.setattr(
        queue_module,
        "_expected_tasks",
        lambda _config: [expected[0], tuple(duplicate_seed)],
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        initialize_transition_queue(config, retry_failed=False)

    connection = sqlite3.connect(queue_module.campaign_database_path(config))
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name IN ('campaign_metadata', 'tasks')"
        ).fetchall()
    finally:
        connection.close()
    assert tables == []


def test_reclaimed_claim_rejects_stale_completion_and_failure(tmp_path: Path) -> None:
    config = load_transition_campaign_config(_write_campaign(tmp_path))
    initialize_transition_queue(config, retry_failed=False)
    stale_md = claim_md_task(config, worker_name="md_before_restart")
    assert stale_md is not None
    assert stale_md.claim_generation == 1
    assert set(stale_md.__dict__) == {
        "task_index",
        "run_name",
        "branch_index",
        "branch_name",
        "replica_index",
        "configured_replica_seed",
        "simulation_seed",
    }

    initialize_transition_queue(config, retry_failed=False)
    active_md = claim_md_task(config, worker_name="md_after_restart")
    assert active_md is not None
    assert active_md.task_index == stale_md.task_index
    assert active_md.claim_generation == stale_md.claim_generation + 1
    assert active_md.claim_token != stale_md.claim_token
    with pytest.raises(RuntimeError, match="cannot transition MD to complete"):
        complete_md_task(
            config,
            task=stale_md,
            raw_directory=tmp_path / "stale_raw",
            raw_commit_sha256="a" * 64,
        )
    with pytest.raises(RuntimeError, match="cannot transition failed MD task"):
        fail_task(config, task=stale_md, error="stale", analysis=False)
    row = campaign_rows(config)[0]
    assert row["md_status"] == "running"
    assert row["md_claim_token"] == active_md.claim_token

    complete_md_task(
        config,
        task=active_md,
        raw_directory=tmp_path / "active_raw",
        raw_commit_sha256="b" * 64,
    )
    stale_analysis = claim_analysis_task(config, worker_name="analysis_before_restart")
    assert stale_analysis is not None
    initialize_transition_queue(config, retry_failed=False)
    active_analysis = claim_analysis_task(config, worker_name="analysis_after_restart")
    assert active_analysis is not None
    assert active_analysis.task_index == stale_analysis.task_index
    assert active_analysis.claim_generation == stale_analysis.claim_generation + 1
    assert active_analysis.claim_token != stale_analysis.claim_token
    with pytest.raises(RuntimeError, match="cannot transition analysis to complete"):
        complete_analysis_task(
            config,
            task=stale_analysis,
            analysis_directory=tmp_path / "stale_analysis",
            analysis_commit_sha256="c" * 64,
        )
    with pytest.raises(RuntimeError, match="cannot transition failed analysis task"):
        fail_task(config, task=stale_analysis, error="stale", analysis=True)
    complete_analysis_task(
        config,
        task=active_analysis,
        analysis_directory=tmp_path / "active_analysis",
        analysis_commit_sha256="d" * 64,
    )


def test_transition_mtk_restore_matches_uninterrupted_emt(tmp_path: Path) -> None:
    config = load_transition_campaign_config(_write_campaign(tmp_path))
    branch = config.transition.temperature_runs[0]
    atoms = _prepared_interface().atoms
    atoms.calc = EMT()
    set_maxwell_boltzmann_velocities(atoms, 300.0, np.random.default_rng(71))
    split_atoms = atoms.copy()
    split_atoms.calc = EMT()
    dynamics = config.transition.generator.dynamics
    uninterrupted = IsotropicMTKNPT(
        atoms,
        timestep=dynamics.timestep_fs * units.fs,
        temperature_K=branch.temperature_K,
        pressure_au=dynamics.pressure_GPa * units.GPa,
        tdamp=dynamics.thermostat_time_fs * units.fs,
        pdamp=dynamics.barostat_time_fs * units.fs,
    )
    split = build_transition_mtk_dynamics(
        split_atoms, config=config, branch=branch, state=None
    )
    uninterrupted.run(4)
    split.run(2)
    state = capture_mtk_state(split)
    restored_atoms = split_atoms.copy()
    restored_atoms.calc = EMT()
    restored = build_transition_mtk_dynamics(
        restored_atoms, config=config, branch=branch, state=state
    )
    restored.run(2)
    np.testing.assert_allclose(restored_atoms.positions, atoms.positions, atol=1e-12)
    np.testing.assert_allclose(restored_atoms.get_momenta(), atoms.get_momenta(), atol=1e-12)
    np.testing.assert_allclose(restored_atoms.cell.array, atoms.cell.array, atol=1e-12)


def test_hashed_checkpoint_snapshot_round_trip(tmp_path: Path) -> None:
    config = load_transition_campaign_config(_write_campaign(tmp_path))
    initialize_transition_queue(config, retry_failed=False)
    task = claim_md_task(config, worker_name="checkpoint_test")
    assert task is not None
    calculator, provenance = select_calculator(
        config.runtime_generator,
        calculator=EMT(),
        injected_calculator_identity="test-only transition checkpoint EMT",
    )
    run_transition_task(
        config,
        task=task,
        prepared=_prepared_interface(),
        calculator=calculator,
        provenance=provenance,
        progress=lambda _message: None,
    )
    checkpoint = TransitionCheckpointStore(config, provenance, task).load()
    assert checkpoint is not None
    assert checkpoint.state.nsteps == 5
    assert checkpoint.trace.step.tolist() == [0, 1, 2, 3, 4, 5]
    assert checkpoint.metadata["completed_global_step"] == 5


def test_persistent_worker_and_deferred_analysis_vertical_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_transition_campaign_config(_write_campaign(tmp_path))
    initialize_transition_queue(config, retry_failed=False)
    prepared = _prepared_interface()
    import src.data_utils.synthetic.atomistic.transition_campaign as campaign_module

    monkeypatch.setattr(
        campaign_module,
        "load_content_bound_prepared_interface",
        lambda _config: prepared,
    )
    real_select = campaign_module.select_calculator
    selection_count = 0

    def counted_select(*args: object, **kwargs: object):
        nonlocal selection_count
        selection_count += 1
        return real_select(*args, **kwargs)

    monkeypatch.setattr(campaign_module, "select_calculator", counted_select)

    def spatial_structure_types(frame_atoms, _rmsd_cutoff):
        z = frame_atoms.get_scaled_positions(wrap=True)[:, 2]
        return ((z < 0.25) | (z >= 0.75)).astype(np.int32)

    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.transition_analysis._ptm_structure_types",
        spatial_structure_types,
    )
    run_md_worker(
        config,
        worker_name="test_persistent_worker",
        calculator=EMT(),
        injected_calculator_identity="test-only persistent transition EMT",
        progress=lambda _message: None,
    )
    assert selection_count == 1
    rows = campaign_rows(config)
    assert all(row["md_status"] == "complete" for row in rows)
    assert all(row["analysis_status"] == "pending" for row in rows)
    assert all(Path(str(row["raw_directory"])).is_dir() for row in rows)

    run_analysis_worker(
        config, worker_name="test_deferred_analysis", progress=lambda _message: None
    )
    rows = campaign_rows(config)
    recovery_row = rows[0]
    recovery_directory = Path(str(recovery_row["analysis_directory"]))
    recovery_commit = json.loads(
        (recovery_directory / "analysis_commit.json").read_text(encoding="utf-8")
    )
    recovery_task = TransitionCampaignTask(
        task_index=int(recovery_row["task_index"]),
        run_name=str(recovery_row["run_name"]),
        branch_index=int(recovery_row["branch_index"]),
        branch_name=str(recovery_row["branch_name"]),
        replica_index=int(recovery_row["replica_index"]),
        configured_replica_seed=int(recovery_row["configured_replica_seed"]),
        simulation_seed=int(recovery_row["simulation_seed"]),
        claim_role="analysis",
        claim_generation=int(recovery_row["analysis_claim_generation"]),
        claim_token=str(recovery_row["analysis_claim_token"]),
    )
    original_raw_digest = str(recovery_row["raw_commit_sha256"])
    connection = sqlite3.connect(config.output_root / "transition_campaign.sqlite3")
    try:
        connection.execute(
            "UPDATE tasks SET raw_commit_sha256=? WHERE task_index=?",
            ("f" * 64, recovery_task.task_index),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(RuntimeError, match="SQLite raw digest"):
        analyze_transition_task(
            config,
            task=recovery_task,
            prepared=prepared,
            prepared_phase_ids=np.zeros(len(prepared.atoms), dtype=np.int64),
            analysis_execution_provenance=recovery_commit[
                "analysis_execution_provenance"
            ],
            progress=lambda _message: None,
        )
    connection = sqlite3.connect(config.output_root / "transition_campaign.sqlite3")
    try:
        connection.execute(
            "UPDATE tasks SET raw_commit_sha256=? WHERE task_index=?",
            (original_raw_digest, recovery_task.task_index),
        )
        connection.commit()
    finally:
        connection.close()
    manifest = finalize_transition_campaign(config)
    assert manifest.is_file()
    rows = campaign_rows(config)
    assert all(row["analysis_status"] == "complete" for row in rows)
    assert all(Path(str(row["analysis_directory"])).is_dir() for row in rows)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    summary = json.loads(
        (config.output_root / "velocity_summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == 2
    assert summary["campaign_source_evidence"] == config.source_evidence
    assert set(summary["analysis_execution_provenance"]) == {
        "runtime",
        "producer_code",
    }
    assert "calculator" not in summary["analysis_execution_provenance"]
    required_artifacts = {
        "trajectory.npz",
        "equilibration_trajectory.npz",
        "transition_progress.npz",
        "metadata.json",
    }
    for temperature in summary["temperatures"]:
        for run in temperature["runs"]:
            assert set(run["artifacts"]) == required_artifacts
            assert all(
                set(record) == {"path", "sha256"}
                for record in run["artifacts"].values()
            )
    assert document["source_evidence"] == config.source_evidence
    assert (
        document["analysis_execution_provenance"]
        == summary["analysis_execution_provenance"]
    )
    raw_commit = json.loads(
        (
            Path(str(rows[0]["raw_directory"])) / "raw_commit.json"
        ).read_text(encoding="utf-8")
    )
    assert raw_commit["execution_provenance"]["calculator"]["source"] == (
        "injected_calculator"
    )
    assert raw_commit["execution_provenance"]["producer_code"]["files"] == list(
        TRANSITION_CAMPAIGN_MD_PRODUCER_FILES
    )
    changed_analysis_provenance = json.loads(
        json.dumps(summary["analysis_execution_provenance"])
    )
    changed_analysis_provenance["runtime"]["machine"] = "different-analysis-host"

    class ChangedAnalysisProvenance:
        def to_dict(self) -> dict[str, object]:
            return changed_analysis_provenance

    monkeypatch.setattr(
        campaign_module,
        "build_transition_deferred_analysis_provenance",
        lambda: ChangedAnalysisProvenance(),
    )
    with pytest.raises(RuntimeError, match="current deferred-analysis provenance"):
        finalize_transition_campaign(config)
    assert document["execution_features"] == {
        "persistent_model_per_gpu_worker": True,
        "dynamic_temperature_replica_queue": True,
        "exact_mtk_state_checkpoint_resume": True,
        "atomic_per_run_raw_and_analysis_commits": True,
        "deferred_cpu_ptm_rdf_analysis": True,
    }
