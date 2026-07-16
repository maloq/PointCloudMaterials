from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.nose_hoover_chain import IsotropicMTKNPT

from src.data_utils.synthetic.atomistic.artifacts import build_atom_table, label_bulk
from src.data_utils.synthetic.atomistic.config import (
    load_config,
    potential_calculator_settings,
)
from src.data_utils.synthetic.atomistic.generator import select_calculator
from src.data_utils.synthetic.atomistic.homogeneous_campaign import (
    RAW_REPLICA_ARTIFACTS,
    analyze_campaign_replica,
    finalize_campaign,
    run_analysis_worker,
    run_md_worker,
)
from src.data_utils.synthetic.atomistic.homogeneous_campaign_config import (
    campaign_config_matches_after_path_relocation,
    load_homogeneous_campaign_config,
)
from src.data_utils.synthetic.atomistic.homogeneous_campaign_queue import (
    CampaignReplicaTask,
    campaign_rows,
    initialize_campaign_queue,
)
from src.data_utils.synthetic.atomistic.homogeneous_liquid_source import (
    generate_homogeneous_liquid_source,
)
from src.data_utils.synthetic.atomistic.homogeneous_online import (
    OnlineCrystallinityObservation,
    OnlineThresholdTracker,
)
from src.data_utils.synthetic.atomistic.homogeneous_resumable import (
    ResumableReplicaCheckpointStore,
    _compatible_checkpoint_identity_migration,
    build_mtk_dynamics,
    capture_mtk_state,
)
from src.data_utils.synthetic.atomistic.potential_selection import (
    POTENTIAL_SELECTION_POLICY_VERSION,
    POTENTIAL_SELECTION_SCHEMA_VERSION,
)
from src.data_utils.synthetic.atomistic.provenance import (
    homogeneous_liquid_source_producer_code_provenance,
)
from src.data_utils.synthetic.atomistic.simulation import (
    set_maxwell_boltzmann_velocities,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_GENERATOR_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)
COMPILED_SOURCE_CONFIGS = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/liquid_source_16384_mpa.yaml",
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/liquid_source_16384_mh1.yaml",
)


def _identity_with_digest(payload: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def test_checkpoint_identity_migration_allows_only_certified_producer_change(
    monkeypatch,
) -> None:
    old_producer = {
        "algorithm": "sha256",
        "scope": "/repository/atomistic",
        "files": ["calculator.py"],
        "sha256": "old",
    }
    new_producer = {**old_producer, "sha256": "new"}
    shared = {
        "schema_version": 1,
        "campaign_config": {"dataset_name": "test"},
        "replica_name": "replica_000",
        "random_seed": 7,
    }
    observed = _identity_with_digest(
        {
            **shared,
            "execution_provenance": {
                "calculator": {"identity": "same"},
                "runtime": {"torch": "same"},
                "producer_code": old_producer,
            },
        }
    )
    expected = _identity_with_digest(
        {
            **shared,
            "execution_provenance": {
                "calculator": {"identity": "same"},
                "runtime": {"torch": "same"},
                "producer_code": new_producer,
            },
        }
    )
    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.homogeneous_resumable."
        "producer_code_is_compatible",
        lambda old, new: old == old_producer and new == new_producer,
    )
    assert _compatible_checkpoint_identity_migration(observed, expected) == expected

    changed_seed_payload = {
        key: value for key, value in expected.items() if key != "identity_sha256"
    }
    changed_seed_payload["random_seed"] = 8
    changed_seed = _identity_with_digest(changed_seed_payload)
    assert _compatible_checkpoint_identity_migration(observed, changed_seed) is None


def test_campaign_config_relocation_changes_only_config_file_locations() -> None:
    observed = {
        "config_path": "/repo/configs/data/campaign.yaml",
        "homogeneous_config": "/repo/configs/data/homogeneous.yaml",
        "homogeneous": {
            "config_path": "/repo/configs/data/homogeneous.yaml",
            "generator": {
                "config_path": "/repo/configs/data/source.yaml",
                "dataset_name": "source",
            },
            "steps": 200000,
        },
        "output_root": "/repo/output/campaign",
    }
    expected = json.loads(json.dumps(observed))
    expected["config_path"] = "/repo/configs/simulation/campaign.yaml"
    expected["homogeneous_config"] = "/repo/configs/simulation/homogeneous.yaml"
    expected["homogeneous"]["config_path"] = (
        "/repo/configs/simulation/homogeneous.yaml"
    )
    expected["homogeneous"]["generator"]["config_path"] = (
        "/repo/configs/simulation/source.yaml"
    )

    assert campaign_config_matches_after_path_relocation(observed, expected)

    changed_physics = json.loads(json.dumps(expected))
    changed_physics["homogeneous"]["steps"] = 199999
    assert not campaign_config_matches_after_path_relocation(
        observed, changed_physics
    )


def _write_test_campaign(
    tmp_path: Path,
    *,
    random_seeds: list[int] | None = None,
) -> tuple[Path, int]:
    source_root = tmp_path / "source"
    source_directory = source_root / "replica_000_bulk_liquid"
    source_directory.mkdir(parents=True)
    generator_raw = yaml.safe_load(BASE_GENERATOR_CONFIG.read_text(encoding="utf-8"))
    generator_raw["dataset_name"] = "test_campaign_source"
    generator_raw["random_seeds"] = [123]
    generator_raw["potential"].update(
        {
            "device": "cpu",
            "enable_cueq": False,
            "enable_oeq": False,
            "compile_mode": None,
            "compile_fullgraph": False,
            "pad_num_atoms": 0,
            "pad_num_edges": 0,
            "md_property_mode": "forces_stress",
        }
    )
    generator_raw["dynamics"].update(
        {
            "target_temperature_K": 300.0,
            "solid_equilibration_steps": 2,
            "melt_steps": 2,
            "quench_steps": 0,
            "target_equilibration_steps": 2,
            "interface_evolution_steps": 2,
            "sample_interval": 1,
        }
    )
    generator_raw["system"]["repetitions"] = [3, 3, 3]
    generator_raw["validation"].update(
        {
            "maximum_force_eV_per_A": 100.0,
            "maximum_pressure_error_GPa": 100.0,
            "maximum_temperature_error_K": 2000.0,
            "minimum_pair_distance_A": 0.5,
            "maximum_liquid_crystalline_fraction": 1.0,
        }
    )
    generator_raw["output"].update(
        {
            "root_dir": str(source_root),
            "overwrite": False,
            "save_extxyz": False,
            "create_visualizations": False,
        }
    )
    generator_raw["data_path"] = str(source_root)
    generator_raw["synthetic"]["root_dir"] = str(source_root)
    generator_path = tmp_path / "generator.yaml"
    generator_path.write_text(yaml.safe_dump(generator_raw), encoding="utf-8")
    generator = load_config(generator_path)

    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    atom_count = len(atoms)
    np.save(
        source_directory / "atoms_full.npy",
        build_atom_table(
            atoms, label_bulk(atom_count, "liquid_bulk", grain_id=1)
        ),
    )
    with (source_directory / "trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=np.array([0], dtype=np.int64),
            positions_A=np.asarray([atoms.positions], dtype=np.float32),
            cell_vectors_A=np.asarray([atoms.cell.array], dtype=np.float64),
            temperature_K=np.array([300.0]),
            pressure_GPa=np.array([0.0]),
            volume_A3=np.array([atoms.get_volume()]),
            potential_energy_eV_per_atom=np.array([0.0]),
        )
    (source_directory / "metadata.json").write_text(
        json.dumps(
            {
                "diagnostics": {
                    "ptm_structure_fractions": {
                        "other": 0.0,
                        "fcc": 1.0,
                        "hcp": 0.0,
                        "bcc": 0.0,
                        "ico": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    potential = generator.potential
    calculator_record = {
        "source": "configured_mace_model",
        "identity": f"{potential.model_name}:{potential.sha256}:{potential.head}",
        "implementation_class": (
            "src.data_utils.synthetic.atomistic.calculator.VerletSkinMACECalculator"
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
        "scientifically_qualified": False,
        "qualification_scope": None,
        "settings": potential_calculator_settings(potential),
    }
    (source_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "source_kind": "immutable_homogeneous_liquid_only",
                "interface_preparation_performed": False,
                "config": generator.to_dict(),
                "potential_sha256": potential.sha256,
                "execution_provenance": {
                    "calculator": calculator_record,
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
                    "producer_code": (
                        homogeneous_liquid_source_producer_code_provenance()
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    homogeneous_path = tmp_path / "homogeneous.yaml"
    homogeneous_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "test_optimized_campaign",
                "source_generator_config": str(generator_path),
                "source_dataset": str(source_root),
                "source_environment": "replica_000_bulk_liquid",
                "source_frame_step": 0,
                "random_seeds": random_seeds or [41, 43],
                "temperature_K": 300.0,
                "equilibration_steps": 2,
                "steps": 2,
                "sample_interval": 1,
                "analysis": {
                    "ptm_rmsd_cutoff": 0.1,
                    "crystalline_cluster_cutoff_A": 3.5,
                    "nucleus_size_threshold_atoms": atom_count + 1,
                    "threshold_persistence_frames": 2,
                    "rdf_cutoff_A": 5.0,
                    "rdf_bins": 20,
                },
                "output": {
                    "root_dir": str(tmp_path / "unused_legacy_output"),
                    "overwrite": False,
                    "save_extxyz": False,
                    "create_visualizations": False,
                },
            }
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "homogeneous_config": str(homogeneous_path),
                "output_root": str(tmp_path / "campaign_output"),
                "potential_selection_report": None,
                "execution": {
                    "chunk_steps": 2,
                    "event_check_interval": 1,
                    "stop_on_event": True,
                    "post_event_steps": 0,
                    "analysis_mode": "deferred",
                    "analysis_workers": 0,
                    "checkpoint_retention": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return campaign_path, atom_count


def _bind_test_selection_report(
    campaign_path: Path,
    *,
    workers: int = 2,
) -> Path:
    campaign_raw = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    homogeneous_path = Path(campaign_raw["homogeneous_config"]).resolve()
    homogeneous_raw = yaml.safe_load(
        homogeneous_path.read_text(encoding="utf-8")
    )
    generator_path = Path(homogeneous_raw["source_generator_config"]).resolve()
    source_root = Path(homogeneous_raw["source_dataset"]).resolve()
    source_directory = source_root / homogeneous_raw["source_environment"]
    source_artifact_paths = {
        "manifest": source_root / "manifest.json",
        "metadata": source_directory / "metadata.json",
        "atom_table": source_directory / "atoms_full.npy",
        "trajectory": source_directory / "trajectory.npz",
    }
    source_artifacts = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in source_artifact_paths.items()
    }
    generator = load_config(generator_path)
    generator_sha256 = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    homogeneous_sha256 = hashlib.sha256(
        homogeneous_path.read_bytes()
    ).hexdigest()
    report = {
        "schema_version": POTENTIAL_SELECTION_SCHEMA_VERSION,
        "report_type": "al_crystallization_mlip_selection",
        "policy_version": POTENTIAL_SELECTION_POLICY_VERSION,
        "baseline_model_name": generator.potential.model_name,
        "candidate_model_name": "unused-test-candidate",
        "selected_generator_config": str(generator_path),
        "selected_model_name": generator.potential.model_name,
        "inputs": {
            "baseline_generator_config": str(generator_path),
            "baseline_generator_config_sha256": generator_sha256,
        },
        "runtime_projection": {
            "is_selection_or_launch_gate": False,
            "workers": workers,
            "makespan_safety_factor": 1.25,
            "baseline": {
                "model_name": generator.potential.model_name,
                "homogeneous_config": str(homogeneous_path),
                "homogeneous_config_sha256": homogeneous_sha256,
                "initial_source_artifacts": source_artifacts,
                "projected_makespan_seconds": 60_000.0,
            },
        },
    }
    report_path = campaign_path.parent / "selection.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    campaign_raw["potential_selection_report"] = str(report_path)
    campaign_path.write_text(yaml.safe_dump(campaign_raw), encoding="utf-8")
    return report_path


@pytest.mark.parametrize("path", COMPILED_SOURCE_CONFIGS)
def test_compiled_liquid_sources_are_explicit_16384_atom_500K_protocols(
    path: Path,
) -> None:
    config = load_config(path)
    assert config.system.repetitions == (16, 16, 16)
    assert len(bulk("Al", "fcc", a=4.05, cubic=True).repeat((16, 16, 16))) == 16384
    assert config.dynamics.target_temperature_K == 500.0
    assert config.dynamics.pressure_GPa == 0.0
    assert config.dynamics.solid_equilibration_steps == 0
    assert config.potential.compile_mode == "reduce-overhead"
    assert config.potential.compile_fullgraph is False
    assert config.potential.enable_cueq is True
    assert config.potential.enable_oeq is False
    assert config.potential.pad_num_atoms == 16384
    assert config.potential.pad_num_edges == 1200000
    assert config.potential.md_property_mode == "forces_stress"
    assert config.output.overwrite is False
    assert config.output.save_extxyz is True


def test_campaign_config_binds_source_and_preserves_persistence_span(
    tmp_path: Path,
) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path)
    config = load_homogeneous_campaign_config(campaign_path)
    assert config.execution.online_persistence_frames == 2
    assert set(config.source_evidence) == {
        "manifest",
        "metadata",
        "atom_table",
        "trajectory",
    }
    assert all(len(item["sha256"]) == 64 for item in config.source_evidence.values())

    initialize_campaign_queue(config, retry_failed=False)
    metadata_path = Path(config.source_evidence["metadata"]["path"])
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    changed_source_config = load_homogeneous_campaign_config(campaign_path)
    with pytest.raises(RuntimeError, match="persisted campaign configuration"):
        initialize_campaign_queue(changed_source_config, retry_failed=False)

    raw = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    raw["potential_selection_report"] = str(tmp_path / "missing_selection.json")
    campaign_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="will not fall back"):
        load_homogeneous_campaign_config(campaign_path)


def test_campaign_binds_schema4_selection_with_advisory_runtime_projection(
    tmp_path: Path,
) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path)
    report_path = _bind_test_selection_report(campaign_path, workers=2)
    config = load_homogeneous_campaign_config(campaign_path)
    assert config.potential_selection_runtime_controls == {
        "workers": 2,
        "makespan_safety_factor": 1.25,
        "projected_makespan_seconds": 60_000.0,
        "is_selection_or_launch_gate": False,
    }
    assert config.potential_selection_runtime_controls["workers"] == 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_projection"]["baseline"][
        "homogeneous_config_sha256"
    ] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="workload changed"):
        load_homogeneous_campaign_config(campaign_path)

    report["schema_version"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected schema_version=4"):
        load_homogeneous_campaign_config(campaign_path)


def test_dense_monitoring_does_not_change_saved_frame_event_definition() -> None:
    tracker = OnlineThresholdTracker(
        threshold_atoms=100,
        persistence_frames=3,
        event_cadence_steps=1000,
    )
    for step, largest_cluster in (
        (0, 110),
        (250, 10),
        (500, 120),
        (750, 20),
        (1000, 115),
        (1250, 5),
        (1500, 125),
        (1750, 15),
        (2000, 130),
    ):
        tracker.append(
            OnlineCrystallinityObservation(
                measurement_step=step,
                crystalline_fraction=0.01,
                crystalline_cluster_count=1,
                largest_crystalline_cluster_atoms=largest_cluster,
            )
        )

    assert tracker.event is not None
    assert tracker.event.onset_step == 0
    assert tracker.event.confirmation_step == 2000


def test_liquid_only_source_is_immutable_and_omits_interface(tmp_path: Path) -> None:
    raw = yaml.safe_load(BASE_GENERATOR_CONFIG.read_text(encoding="utf-8"))
    output_root = tmp_path / "liquid_only"
    raw["dataset_name"] = "test_liquid_only"
    raw["random_seeds"] = [17]
    raw["potential"].update(
        {
            "device": "cpu",
            "enable_cueq": False,
            "enable_oeq": False,
            "compile_mode": None,
            "compile_fullgraph": False,
            "pad_num_atoms": 0,
            "pad_num_edges": 0,
            "md_property_mode": "forces_stress",
        }
    )
    raw["dynamics"].update(
        {
            "target_temperature_K": 300.0,
            "solid_equilibration_steps": 1,
            "melt_steps": 1,
            "quench_steps": 0,
            "target_equilibration_steps": 1,
            "interface_evolution_steps": 1,
            "sample_interval": 1,
        }
    )
    raw["system"]["repetitions"] = [2, 2, 2]
    raw["validation"].update(
        {
            "maximum_force_eV_per_A": 100.0,
            "maximum_pressure_error_GPa": 100.0,
            "maximum_temperature_error_K": 2000.0,
            "minimum_pair_distance_A": 0.5,
            "maximum_liquid_crystalline_fraction": 1.0,
        }
    )
    raw["data_path"] = str(output_root)
    raw["synthetic"]["root_dir"] = str(output_root)
    raw["output"].update(
        {
            "root_dir": str(output_root),
            "overwrite": False,
            "save_extxyz": False,
            "create_visualizations": False,
        }
    )
    path = tmp_path / "liquid_source.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = generate_homogeneous_liquid_source(
        load_config(path),
        calculator=EMT(),
        injected_calculator_identity="test-only liquid-source EMT",
        progress=lambda _message: None,
    )
    manifest = json.loads(
        (result.output_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_kind"] == "immutable_homogeneous_liquid_only"
    assert manifest["interface_preparation_performed"] is False
    assert manifest["environment_dirs"] == ["replica_000_bulk_liquid"]
    assert not any("interface" in path.name for path in result.output_root.iterdir())
    with pytest.raises(FileExistsError, match="Immutable homogeneous liquid source"):
        generate_homogeneous_liquid_source(
            load_config(path),
            calculator=EMT(),
            injected_calculator_identity="test-only liquid-source EMT",
            progress=lambda _message: None,
        )


def test_exact_mtk_state_resume_matches_uninterrupted_emt(tmp_path: Path) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path, random_seeds=[47])
    config = load_homogeneous_campaign_config(campaign_path)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    atoms.calc = EMT()
    set_maxwell_boltzmann_velocities(atoms, 300.0, np.random.default_rng(71))
    resumed_atoms = atoms.copy()
    resumed_atoms.calc = EMT()

    uninterrupted = IsotropicMTKNPT(
        atoms,
        timestep=config.homogeneous.generator.dynamics.timestep_fs * units.fs,
        temperature_K=300.0,
        pressure_au=0.0,
        tdamp=config.homogeneous.generator.dynamics.thermostat_time_fs * units.fs,
        pdamp=config.homogeneous.generator.dynamics.barostat_time_fs * units.fs,
    )
    split = build_mtk_dynamics(resumed_atoms, config=config, state=None)
    uninterrupted.run(4)
    split.run(2)
    state = capture_mtk_state(split)
    restored_atoms = resumed_atoms.copy()
    restored_atoms.calc = EMT()
    restored = build_mtk_dynamics(restored_atoms, config=config, state=state)
    restored.run(2)

    assert np.allclose(restored_atoms.positions, atoms.positions, atol=1.0e-12)
    assert np.allclose(restored_atoms.get_momenta(), atoms.get_momenta(), atol=1.0e-12)
    assert np.allclose(restored_atoms.cell.array, atoms.cell.array, atol=1.0e-12)


def test_dynamic_worker_reuses_calculator_and_offline_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path)
    config = load_homogeneous_campaign_config(campaign_path)
    initialize_campaign_queue(config, retry_failed=False)
    import src.data_utils.synthetic.atomistic.homogeneous_campaign as campaign_module

    real_select = campaign_module.select_calculator
    selection_calls = 0

    def counted_select(*args: object, **kwargs: object):
        nonlocal selection_calls
        selection_calls += 1
        return real_select(*args, **kwargs)

    monkeypatch.setattr(campaign_module, "select_calculator", counted_select)
    run_md_worker(
        config,
        worker_name="test_cpu_worker",
        calculator=EMT(),
        injected_calculator_identity="test-only persistent EMT",
        progress=lambda _message: None,
    )
    assert selection_calls == 1
    rows = campaign_rows(config)
    assert [row["md_status"] for row in rows] == ["complete", "complete"]
    assert [row["outcome"] for row in rows] == [
        "right_censored",
        "right_censored",
    ]
    assert all(len(str(row["run_metadata_sha256"])) == 64 for row in rows)
    assert all(
        json.loads(str(row["online_threshold_event_json"]))["observed"] is False
        for row in rows
    )
    run_analysis_worker(
        config,
        worker_name="test_analysis_worker",
        follow_md=False,
        progress=lambda _message: None,
    )
    manifest = finalize_campaign(config)
    assert manifest.is_file()
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_document["outcome_counts"]["right_censored"] == 2
    assert set(manifest_document["campaign_artifacts_sha256"]) == {
        "survival_summary.json",
        "survival_curve.npz",
    }
    assert (
        manifest_document["replica_initial_state_design"]
        ["independently_sampled_coordinate_configurations"]
        is False
    )
    assert "conditional" in manifest_document["scientific_scope"][
        "supported_claim"
    ]
    completed_rows = campaign_rows(config)
    assert all(len(str(row["full_analysis_sha256"])) == 64 for row in completed_rows)
    for row in completed_rows:
        replica = config.output_root / "replicas" / str(row["replica_name"])
        assert (replica / "online_crystallinity.npz").is_file()
        assert (replica / "total_rdf.npz").is_file()
        metadata = json.loads(
            (replica / "run_metadata.json").read_text(encoding="utf-8")
        )
        analysis = json.loads(
            (replica / "full_analysis.json").read_text(encoding="utf-8")
        )
        assert set(metadata["raw_artifacts_sha256"]) == set(RAW_REPLICA_ARTIFACTS)
        assert analysis["raw_artifacts_sha256"] == metadata[
            "raw_artifacts_sha256"
        ]
        assert analysis["run_metadata_sha256"] == row["run_metadata_sha256"]
        assert analysis["queue_outcome"] == row["outcome"]
        assert analysis["online_offline_shared_frame_audit"]["status"] == "exact_match"


def test_finalize_rejects_run_metadata_outcome_event_tampering(
    tmp_path: Path,
) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path, random_seeds=[61])
    config = load_homogeneous_campaign_config(campaign_path)
    initialize_campaign_queue(config, retry_failed=False)
    run_md_worker(
        config,
        worker_name="test_metadata_anchor_worker",
        calculator=EMT(),
        injected_calculator_identity="test-only metadata-anchor EMT",
        progress=lambda _message: None,
    )
    run_analysis_worker(
        config,
        worker_name="test_metadata_anchor_analysis",
        follow_md=False,
        progress=lambda _message: None,
    )
    metadata_path = (
        config.output_root / "replicas" / "replica_000" / "run_metadata.json"
    )
    original_metadata = metadata_path.read_text(encoding="utf-8")
    metadata = json.loads(original_metadata)
    assert metadata["outcome"] == "right_censored"
    metadata["outcome"] = "event_stopped"
    metadata["online_threshold_event"].update(
        {
            "observed": True,
            "onset_step": 1,
            "confirmation_step": 2,
            "onset_time_ps": 0.001,
            "confirmation_time_ps": 0.002,
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="externally anchored run-metadata SHA-256 mismatch"
    ):
        finalize_campaign(config)

    metadata_path.write_text(original_metadata, encoding="utf-8")
    analysis_path = metadata_path.parent / "full_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["saved_frame_analysis"]["nucleation_observed"] = True
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="externally anchored full-analysis SHA-256 mismatch"
    ):
        finalize_campaign(config)


def test_deferred_analysis_rejects_committed_raw_artifact_corruption(
    tmp_path: Path,
) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path, random_seeds=[59])
    config = load_homogeneous_campaign_config(campaign_path)
    initialize_campaign_queue(config, retry_failed=False)
    run_md_worker(
        config,
        worker_name="test_raw_integrity_worker",
        calculator=EMT(),
        injected_calculator_identity="test-only raw-integrity EMT",
        progress=lambda _message: None,
    )
    trajectory_path = (
        config.output_root / "replicas" / "replica_000" / "trajectory.npz"
    )
    trajectory_path.write_bytes(trajectory_path.read_bytes() + b"corruption")

    with pytest.raises(RuntimeError, match="raw replica artifact integrity failed"):
        analyze_campaign_replica(
            config,
            task=CampaignReplicaTask(
                replica_index=0,
                replica_name="replica_000",
                random_seed=59,
            ),
            progress=lambda _message: None,
        )


def test_checkpoint_rejects_hashed_artifact_corruption(tmp_path: Path) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path, random_seeds=[53])
    config = load_homogeneous_campaign_config(campaign_path)
    initialize_campaign_queue(config, retry_failed=False)
    run_md_worker(
        config,
        worker_name="test_checkpoint_worker",
        calculator=EMT(),
        injected_calculator_identity="test-only checkpoint EMT",
        progress=lambda _message: None,
    )
    _, provenance = select_calculator(
        config.homogeneous.generator,
        calculator=EMT(),
        injected_calculator_identity="test-only checkpoint EMT",
    )
    store = ResumableReplicaCheckpointStore(
        config,
        provenance,
        replica_name="replica_000",
        random_seed=53,
    )
    committed = store.load()
    assert committed is not None
    # A logically identical same-step commit is accepted only after the committed hashes
    # have been verified; a different replay is rejected by the store.
    store.save(
        atoms=committed.atoms,
        trace=committed.trace,
        online_observations=committed.online_observations,
        integrator_state=committed.integrator_state,
        metadata=committed.metadata,
    )
    latest = (store.directory / "LATEST").read_text(encoding="utf-8").strip()
    state_path = store.directory / latest / "mtk_state.npz"
    state_path.write_bytes(state_path.read_bytes() + b"corruption")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        store.load()


def test_checkpoint_recovers_missing_latest_only_from_verified_snapshots(
    tmp_path: Path,
) -> None:
    campaign_path, _ = _write_test_campaign(tmp_path, random_seeds=[67])
    config = load_homogeneous_campaign_config(campaign_path)
    initialize_campaign_queue(config, retry_failed=False)
    run_md_worker(
        config,
        worker_name="test_missing_latest_worker",
        calculator=EMT(),
        injected_calculator_identity="test-only missing-LATEST EMT",
        progress=lambda _message: None,
    )
    _, provenance = select_calculator(
        config.homogeneous.generator,
        calculator=EMT(),
        injected_calculator_identity="test-only missing-LATEST EMT",
    )
    store = ResumableReplicaCheckpointStore(
        config,
        provenance,
        replica_name="replica_000",
        random_seed=67,
    )
    snapshots = sorted(store.directory.glob("step_*"))
    assert len(snapshots) == 2
    newest = snapshots[-1]
    latest_path = store.directory / "LATEST"
    latest_path.unlink()

    recovered = store.load()
    assert recovered is not None
    assert recovered.integrator_state.nsteps == int(newest.name.removeprefix("step_"))
    assert latest_path.read_text(encoding="utf-8").strip() == newest.name

    # Recovery verifies every candidate, not merely the newest one it would select.
    latest_path.unlink()
    older_state_path = snapshots[0] / "mtk_state.npz"
    original_older_state = older_state_path.read_bytes()
    older_state_path.write_bytes(original_older_state + b"corruption")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        store.load()
    older_state_path.write_bytes(original_older_state)

    inconsistent_name = newest.with_name(
        f"step_{recovered.integrator_state.nsteps + 1:012d}"
    )
    newest.rename(inconsistent_name)
    with pytest.raises(RuntimeError, match="directory name is inconsistent"):
        store.load()
