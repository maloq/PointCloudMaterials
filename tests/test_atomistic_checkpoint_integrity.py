from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from src.data_utils.synthetic.atomistic.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointStore,
)
from src.data_utils.synthetic.atomistic.config import load_config
from src.data_utils.synthetic.atomistic.provenance import (
    build_execution_provenance,
    injected_calculator_provenance,
    _producer_code_provenance,
    validate_configured_source_manifest,
)
from src.data_utils.synthetic.atomistic.simulation import (
    ThermodynamicTrace,
    _TraceRecorder,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)


def _trace(positions_A: np.ndarray, cell_A: np.ndarray) -> ThermodynamicTrace:
    atom_count = len(positions_A)
    volume_A3 = float(np.linalg.det(cell_A))
    return ThermodynamicTrace(
        step=np.array([0, 1], dtype=np.int64),
        temperature_K=np.array([300.0, 301.0], dtype=np.float64),
        pressure_GPa=np.array([0.0, 0.1], dtype=np.float64),
        volume_A3=np.array([volume_A3, volume_A3], dtype=np.float64),
        potential_energy_eV_per_atom=np.array([-1.0, -0.9], dtype=np.float64),
        positions_A=np.repeat(
            np.asarray(positions_A, dtype=np.float64)[None, :, :], 2, axis=0
        ),
        cell_vectors_A=np.repeat(cell_A[None, :, :], 2, axis=0),
    )


def _execution_provenance(identity: str, calculator: EMT | None = None):
    return build_execution_provenance(
        injected_calculator_provenance(
            EMT() if calculator is None else calculator,
            identity=identity,
        )
    )


def test_checkpoint_identity_binds_calculator_runtime_and_producer_code(
    tmp_path: Path,
) -> None:
    config = load_config(PRODUCTION_CONFIG)
    config = replace(config, output=replace(config.output, root_dir=tmp_path / "dataset"))
    execution = _execution_provenance("ase-emt-default:test-a")
    calculator_changed = _execution_provenance("ase-emt-default:test-b")
    calculator_parameters_changed = _execution_provenance(
        "ase-emt-default:test-a", EMT(asap_cutoff=True)
    )
    runtime_changed = replace(
        execution,
        runtime={**execution.runtime, "ase": "deliberately-different-test-version"},
    )
    producer_changed = replace(
        execution,
        producer_code={**execution.producer_code, "sha256": "0" * 64},
    )

    stores = [
        CheckpointStore(config, provenance)
        for provenance in (
            execution,
            calculator_changed,
            calculator_parameters_changed,
            runtime_changed,
            producer_changed,
        )
    ]
    assert len({store.directory for store in stores}) == 5

    manifest = json.loads(
        (stores[0].directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert manifest["execution_provenance"] == execution.to_dict()
    assert manifest["checkpoint_identity_sha256"] == stores[0].identity_sha256

    serialized_config = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":")
    )
    legacy_config_hash = hashlib.sha256(serialized_config.encode("utf-8")).hexdigest()
    legacy_directory = (
        config.output.root_dir.parent
        / f".{config.output.root_dir.name}.generation-{legacy_config_hash[:12]}"
    )
    assert stores[0].directory != legacy_directory


def test_checkpoint_rejects_corrupt_cell_volume_on_save_and_load(tmp_path: Path) -> None:
    config = load_config(PRODUCTION_CONFIG)
    config = replace(config, output=replace(config.output, root_dir=tmp_path / "dataset"))
    store = CheckpointStore(config, _execution_provenance("ase-emt-default:test"))
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    trace = _trace(atoms.positions, atoms.cell.array)

    corrupt_trace = replace(trace, cell_vectors_A=trace.cell_vectors_A * 1.01)
    with pytest.raises(ValueError, match=r"det\(cell_vectors_A\).+internally corrupt"):
        store.save("corrupt-save", atoms, corrupt_trace)

    store.save("corrupt-load", atoms, trace)
    trace_path = store.directory / "corrupt-load.trace.npz"
    with np.load(trace_path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["cell_vectors_A"] = arrays["cell_vectors_A"] * 1.01
    with trace_path.open("wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(ValueError, match=r"det\(cell_vectors_A\).+internally corrupt"):
        store.load("corrupt-load")


def test_checkpoint_rejects_endpoint_mismatch_on_save_and_load(tmp_path: Path) -> None:
    config = load_config(PRODUCTION_CONFIG)
    config = replace(config, output=replace(config.output, root_dir=tmp_path / "dataset"))
    store = CheckpointStore(config, _execution_provenance("ase-emt-default:endpoint"))
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    trace = _trace(atoms.positions, atoms.cell.array)

    displaced_atoms = atoms.copy()
    displaced_atoms.positions[0, 0] += 0.05
    with pytest.raises(RuntimeError, match="positions do not match"):
        store.save("mismatch-save", displaced_atoms, trace)

    store.save("mismatch-load", atoms, trace)
    atoms_path = store.directory / "mismatch-load.traj"
    displaced_atoms.write(atoms_path, format="traj")
    with pytest.raises(RuntimeError, match="positions do not match"):
        store.load("mismatch-load")


def test_trace_finish_rejects_aliased_or_inconsistent_cells() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.calc = EMT()
    recorder = _TraceRecorder(atoms)
    recorder.sample(0)
    recorder.cell_vectors_A[0] *= 1.01

    with pytest.raises(ValueError, match=r"det\(cell_vectors_A\).+internally corrupt"):
        recorder.finish("deliberately-corrupted-test")


def test_derived_dataset_rejects_legacy_source_without_provenance(
    tmp_path: Path,
) -> None:
    config = load_config(PRODUCTION_CONFIG)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 4,
        "config": config.to_dict(),
        "potential_sha256": config.potential.sha256,
    }
    with pytest.raises(RuntimeError, match="no execution_provenance.*must be regenerated"):
        validate_configured_source_manifest(
            manifest,
            config=config,
            manifest_path=manifest_path,
        )

    manifest["execution_provenance"] = {
        "calculator": {
            "source": "configured_mace_model",
            "identity": (
                f"{config.potential.model_name}:{config.potential.sha256}:"
                f"{config.potential.head}"
            ),
            "implementation_class": (
                "src.data_utils.synthetic.atomistic.calculator."
                "VerletSkinMACECalculator"
            ),
            "model_name": config.potential.model_name,
            "family": config.potential.family,
            "model_path": str(config.potential.model_path),
            "model_sha256": config.potential.sha256,
            "head": config.potential.head,
            "available_heads": [config.potential.head],
            "source_url": config.potential.source_url,
            "license_identifier": config.potential.license_identifier,
            "usage_mode": config.potential.usage_mode,
            "validation_report_path": None,
            "validation_report_sha256": None,
            "validation_report_type": None,
            "scientifically_qualified": config.potential.scientifically_qualified,
            "qualification_scope": None,
            "settings": {
                "device": config.potential.device,
                "default_dtype": config.potential.default_dtype,
                "enable_cueq": config.potential.enable_cueq,
                "neighbor_skin_A": config.potential.neighbor_skin_A,
            },
        },
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
            "cuequivariance": "test",
            "cuequivariance_torch": "test",
            "cuequivariance_ops_torch_cu12": "test",
            "cuda_device_index": 0,
            "cuda_device_name": "test",
        },
        "producer_code": _producer_code_provenance(),
    }
    calculator = validate_configured_source_manifest(
        manifest,
        config=config,
        manifest_path=manifest_path,
    )
    assert calculator["head"] == "default"
