from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT

from src.data_utils.synthetic.atomistic.config import load_config
from src.data_utils.synthetic.atomistic.jumpy_ffs import (
    DynamicalState,
    JumpyFFSAlgorithmConfig,
    require_branchable_integrator,
    run_jumpy_ffs,
)
from src.data_utils.synthetic.atomistic.jumpy_ffs_config import (
    load_jumpy_ffs_config,
)
from src.data_utils.synthetic.atomistic.jumpy_ffs_engine import (
    LangevinNVTShotEngine,
)
from src.data_utils.synthetic.atomistic.potential_selection import (
    POTENTIAL_SELECTION_POLICY_VERSION,
    POTENTIAL_SELECTION_SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/jumpy_ffs_16384_mpa.yaml"
)
MH1_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/jumpy_ffs_16384_mh1.yaml"
)


class PositionClusterCV:
    def __init__(self) -> None:
        self.evaluated_steps: list[int] = []

    @property
    def contract(self) -> dict[str, Any]:
        return {"name": "deterministic_test_integer_cv", "version": 1}

    def evaluate(self, state: DynamicalState) -> int:
        self.evaluated_steps.append(state.step)
        return int(round(float(state.positions_A[0, 0])))


class ScriptedJumpyEngine:
    def __init__(
        self,
        *,
        fail_on_advance_call: int | None = None,
        branch_outcome: str = "mixed",
    ) -> None:
        if branch_outcome not in {"mixed", "all_failure", "all_success"}:
            raise ValueError(f"unsupported test branch_outcome={branch_outcome!r}")
        self.advance_calls = 0
        self.fail_on_advance_call = fail_on_advance_call
        self.branch_outcome = branch_outcome
        self._contract = {
            "ensemble": "langevin_nvt",
            "fixed_cell": True,
            "complete_restart_state": True,
            "integrator": "deterministic_test_engine",
        }

    @property
    def contract(self) -> dict[str, Any]:
        return copy.deepcopy(self._contract)

    @property
    def timestep_fs(self) -> float:
        return 1.0

    def branch(
        self, state: DynamicalState, *, random_seed: int
    ) -> DynamicalState:
        rng = np.random.default_rng(random_seed)
        rng_state = copy.deepcopy(rng.bit_generator.state)
        rng_state["test_branch_code"] = random_seed % 6
        return DynamicalState(
            positions_A=state.positions_A.copy(),
            momenta=state.momenta.copy(),
            cell_A=state.cell_A.copy(),
            atomic_numbers=state.atomic_numbers.copy(),
            masses=state.masses.copy(),
            pbc=state.pbc.copy(),
            step=state.step,
            rng_state=rng_state,
            integrator_contract=self.contract,
        )

    def advance(self, state: DynamicalState, *, steps: int) -> DynamicalState:
        assert steps == 1
        self.advance_calls += 1
        if self.advance_calls == self.fail_on_advance_call:
            raise RuntimeError("injected interruption")
        positions = state.positions_A.copy()
        branch_code = state.rng_state.get("test_branch_code")
        if branch_code is None:
            basin_cv_by_step = {1: 2, 2: 0, 3: 5}
            positions[0, 0] = basin_cv_by_step[state.step + 1]
        else:
            source_cv = int(round(float(positions[0, 0])))
            if self.branch_outcome == "all_failure":
                positions[0, 0] = 0
            elif self.branch_outcome == "all_success":
                positions[0, 0] = 8
            elif source_cv == 2:
                positions[0, 0] = 4 if branch_code == 0 else 8
            elif source_cv in {4, 5}:
                positions[0, 0] = 0 if branch_code % 2 == 0 else 8
            else:
                raise AssertionError(f"unexpected scripted branch source CV={source_cv}")
        return DynamicalState(
            positions_A=positions,
            momenta=state.momenta.copy(),
            cell_A=state.cell_A.copy(),
            atomic_numbers=state.atomic_numbers.copy(),
            masses=state.masses.copy(),
            pbc=state.pbc.copy(),
            step=state.step + steps,
            rng_state=copy.deepcopy(state.rng_state),
            integrator_contract=self.contract,
        )


class SparsePhaseJumpyEngine(ScriptedJumpyEngine):
    """Long deterministic paths used to audit sparse checkpoint semantics."""

    def __init__(
        self,
        *,
        fail_on_advance_call: int | None = None,
        second_basin_crossing_absolute_step: int = 20,
    ) -> None:
        super().__init__(fail_on_advance_call=fail_on_advance_call)
        self.second_basin_crossing_absolute_step = (
            second_basin_crossing_absolute_step
        )

    def branch(
        self, state: DynamicalState, *, random_seed: int
    ) -> DynamicalState:
        branched = super().branch(state, random_seed=random_seed)
        rng_state = copy.deepcopy(branched.rng_state)
        rng_state["test_branch_start_step"] = state.step
        return DynamicalState(
            positions_A=branched.positions_A,
            momenta=branched.momenta,
            cell_A=branched.cell_A,
            atomic_numbers=branched.atomic_numbers,
            masses=branched.masses,
            pbc=branched.pbc,
            step=branched.step,
            rng_state=rng_state,
            integrator_contract=branched.integrator_contract,
        )

    def advance(self, state: DynamicalState, *, steps: int) -> DynamicalState:
        assert steps == 1
        self.advance_calls += 1
        if self.advance_calls == self.fail_on_advance_call:
            raise RuntimeError("injected interruption")
        positions = state.positions_A.copy()
        next_step = state.step + 1
        branch_start_step = state.rng_state.get("test_branch_start_step")
        if branch_start_step is None:
            # Six equilibration steps, then basin exits at relative steps 7 and 14.
            positions[0, 0] = (
                2
                if next_step
                in {13, self.second_basin_crossing_absolute_step}
                else 0
            )
        else:
            shot_elapsed_steps = next_step - int(branch_start_step)
            positions[0, 0] = 8 if shot_elapsed_steps == 7 else 2
        return DynamicalState(
            positions_A=positions,
            momenta=state.momenta.copy(),
            cell_A=state.cell_A.copy(),
            atomic_numbers=state.atomic_numbers.copy(),
            masses=state.masses.copy(),
            pbc=state.pbc.copy(),
            step=next_step,
            rng_state=copy.deepcopy(state.rng_state),
            integrator_contract=self.contract,
        )


def _initial_scripted_state(engine: ScriptedJumpyEngine) -> DynamicalState:
    atom_count = 10
    positions = np.zeros((atom_count, 3), dtype=np.float64)
    positions[:, 1] = np.arange(atom_count, dtype=np.float64)
    rng = np.random.default_rng(777)
    return DynamicalState(
        positions_A=positions,
        momenta=np.zeros((atom_count, 3), dtype=np.float64),
        cell_A=np.diag([10.0, 10.0, 10.0]),
        atomic_numbers=np.full(atom_count, 13, dtype=np.int32),
        masses=np.full(atom_count, 26.9815385),
        pbc=np.ones(3, dtype=np.bool_),
        step=0,
        rng_state=copy.deepcopy(rng.bit_generator.state),
        integrator_contract=engine.contract,
    )


def _algorithm() -> JumpyFFSAlgorithmConfig:
    return JumpyFFSAlgorithmConfig(
        interfaces_atoms=(2, 4, 8),
        equilibration_steps=0,
        equilibration_checkpoint_interval_steps=3,
        basin_target_crossings=2,
        basin_max_steps=3,
        basin_checkpoint_interval_steps=3,
        cv_interval_steps=1,
        trials_per_state=2,
        shot_max_steps=2,
        shot_checkpoint_interval_steps=2,
        bootstrap_samples=100,
        bootstrap_block_crossings=1,
        random_seed=123,
    )


def _sparse_phase_algorithm(
    *,
    equilibration_checkpoint_steps: int = 4,
    basin_checkpoint_steps: int = 5,
    shot_checkpoint_steps: int = 3,
    basin_max_steps: int = 14,
) -> JumpyFFSAlgorithmConfig:
    return JumpyFFSAlgorithmConfig(
        interfaces_atoms=(2, 4, 8),
        equilibration_steps=6,
        equilibration_checkpoint_interval_steps=equilibration_checkpoint_steps,
        basin_target_crossings=2,
        basin_max_steps=basin_max_steps,
        basin_checkpoint_interval_steps=basin_checkpoint_steps,
        cv_interval_steps=1,
        trials_per_state=1,
        shot_max_steps=7,
        shot_checkpoint_interval_steps=shot_checkpoint_steps,
        bootstrap_samples=100,
        bootstrap_block_crossings=1,
        random_seed=123,
    )


def _early_b_algorithm(*, equilibration_steps: int) -> JumpyFFSAlgorithmConfig:
    return JumpyFFSAlgorithmConfig(
        interfaces_atoms=(1, 2),
        equilibration_steps=equilibration_steps,
        equilibration_checkpoint_interval_steps=3,
        basin_target_crossings=2,
        basin_max_steps=3,
        basin_checkpoint_interval_steps=3,
        cv_interval_steps=1,
        trials_per_state=1,
        shot_max_steps=2,
        shot_checkpoint_interval_steps=2,
        bootstrap_samples=100,
        bootstrap_block_crossings=1,
        random_seed=123,
    )


TEST_SOURCE_EVIDENCE = {"source": "deterministic unit-test state"}
TEST_SCIENTIFIC_SCOPE = {"claim_status": "test_only"}


def _audit_state_ids(output_root: Path) -> list[str]:
    return (output_root / "state_sha256_audit.log").read_text(
        encoding="ascii"
    ).splitlines()


def _generator_path_from_jffs(path: Path) -> Path:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    generator_path = Path(raw["source_generator_config"])
    if not generator_path.is_absolute():
        generator_path = REPOSITORY_ROOT / generator_path
    return generator_path.resolve()


def _write_selected_jffs_config(
    tmp_path: Path,
    base_path: Path,
    *,
    selected_generator_path: Path | None = None,
    selected_model_name: str | None = None,
    selected_config_sha256: str | None = None,
    schema_version: int = POTENTIAL_SELECTION_SCHEMA_VERSION,
    report_type: str = "al_crystallization_mlip_selection",
    policy_version: str = POTENTIAL_SELECTION_POLICY_VERSION,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_generator_path = _generator_path_from_jffs(base_path)
    selected_path = (
        source_generator_path
        if selected_generator_path is None
        else selected_generator_path.resolve()
    )
    selected_generator = load_config(selected_path)
    report_path = tmp_path / f"{base_path.stem}_selection.json"
    report = {
        "schema_version": schema_version,
        "report_type": report_type,
        "policy_version": policy_version,
        "selected_generator_config": str(selected_path),
        "selected_model_name": (
            selected_generator.potential.model_name
            if selected_model_name is None
            else selected_model_name
        ),
        "inputs": {
            "baseline_generator_config": str(selected_path),
            "baseline_generator_config_sha256": (
                hashlib.sha256(selected_path.read_bytes()).hexdigest()
                if selected_config_sha256 is None
                else selected_config_sha256
            ),
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    raw["potential_selection_report"] = str(report_path)
    config_path = tmp_path / base_path.name
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def test_weighted_jumpy_ffs_accounts_for_skipped_interfaces(tmp_path: Path) -> None:
    engine = ScriptedJumpyEngine()
    output_root = tmp_path / "jffs"
    result = run_jumpy_ffs(
        _initial_scripted_state(engine),
        algorithm=_algorithm(),
        engine=engine,
        collective_variable=PositionClusterCV(),
        output_root=output_root,
        resume=False,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )

    assert result.basin_crossing_count == 2
    assert result.basin_elapsed_time_ps == pytest.approx(0.003)
    assert result.basin_flux_per_ps == pytest.approx(2.0 / 0.003)
    assert result.initial_landing_probabilities == {
        "0": 0.5,
        "1": 0.5,
        "2": 0.0,
    }
    # Seed 123 makes one interface-0 trial jump directly to B (weight 1/4),
    # while the other lands on interface 1 and one of its two children reaches
    # B (weight 1/8). The independently arriving interface-1 root returns to A.
    assert result.crossing_probability == pytest.approx(0.375)
    assert result.root_success_probabilities == pytest.approx((0.75, 0.0))
    assert result.rate_per_A3_ps == pytest.approx((2.0 / 0.003) * 0.375 / 1000.0)
    assert result.rate_per_m3_s == pytest.approx(result.rate_per_A3_ps * 1.0e42)
    assert result.trial_count == 6
    assert result.source_evidence == TEST_SOURCE_EVIDENCE
    assert all(
        result.scientific_scope[key] == value
        for key, value in TEST_SCIENTIFIC_SCOPE.items()
    )
    assert result.uncertainty_status == "paired_moving_block_bootstrap_available"
    assert result.uncertainty_reason is None
    assert result.bootstrap_samples_used == _algorithm().bootstrap_samples
    assert result.confidence_level == 0.95
    assert result.rate_confidence_interval_per_A3_ps is not None
    assert result.rate_bootstrap_standard_error_per_A3_ps is not None
    assert (
        result.scientific_scope["rate_uncertainty_status"]
        == result.uncertainty_status
    )

    interface_zero = result.interface_statistics[0]
    assert interface_zero["weighted_trial_mass"] == pytest.approx(0.5)
    assert interface_zero["conditional_landing_probability"] == pytest.approx(
        {"-1": 0.0, "1": 0.5, "2": 0.5}
    )
    interface_one = result.interface_statistics[1]
    assert interface_one["conditional_landing_probability"] == pytest.approx(
        {"-1": 5.0 / 6.0, "2": 1.0 / 6.0}
    )

    journal = json.loads((output_root / "journal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["identity"]["initial_state_sha256"]) == 64
    assert manifest["identity"]["source_evidence"] == TEST_SOURCE_EVIDENCE
    direct_skip = [
        record
        for record in journal["shooting"]["trial_records"]
        if record["source_interface"] == 0 and record["landing_interface"] == 2
    ]
    assert len(direct_skip) == 1
    assert direct_skip[0]["trial_weight"] == pytest.approx(0.25)
    audit_state_ids = (
        output_root / "state_sha256_audit.log"
    ).read_text(encoding="ascii").splitlines()
    assert audit_state_ids
    assert all(
        len(state_id) == 64
        and all(character in "0123456789abcdef" for character in state_id)
        for state_id in audit_state_ids
    )
    recorded_endpoint_ids = {
        record["endpoint_state_id"]
        for record in journal["shooting"]["trial_records"]
    }
    assert recorded_endpoint_ids.issubset(set(audit_state_ids))
    assert list((output_root / "states").glob("*.npz")) == []


def test_sparse_checkpoints_preserve_events_and_reduce_full_state_writes(
    tmp_path: Path,
) -> None:
    sparse_root = tmp_path / "sparse"
    sparse_engine = SparsePhaseJumpyEngine()
    sparse_cv = PositionClusterCV()
    sparse_result = run_jumpy_ffs(
        _initial_scripted_state(sparse_engine),
        algorithm=_sparse_phase_algorithm(),
        engine=sparse_engine,
        collective_variable=sparse_cv,
        output_root=sparse_root,
        resume=False,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )

    eager_root = tmp_path / "every_cv"
    eager_engine = SparsePhaseJumpyEngine()
    eager_cv = PositionClusterCV()
    eager_result = run_jumpy_ffs(
        _initial_scripted_state(eager_engine),
        algorithm=_sparse_phase_algorithm(
            equilibration_checkpoint_steps=1,
            basin_checkpoint_steps=1,
            shot_checkpoint_steps=1,
        ),
        engine=eager_engine,
        collective_variable=eager_cv,
        output_root=eager_root,
        resume=False,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )

    assert sparse_result.to_dict() == eager_result.to_dict()
    assert sparse_cv.evaluated_steps == eager_cv.evaluated_steps
    assert sparse_cv.evaluated_steps == [
        0,
        *range(1, 21),
        13,
        *range(14, 21),
        20,
        *range(21, 28),
    ]
    sparse_journal = json.loads(
        (sparse_root / "journal.json").read_text(encoding="utf-8")
    )
    assert [item["crossing_step"] for item in sparse_journal["basin"]["exits"]] == [
        7,
        14,
    ]
    assert [item["landing_interface"] for item in sparse_journal["basin"]["exits"]] == [
        0,
        0,
    ]
    assert [item["elapsed_steps"] for item in sparse_journal["shooting"]["trial_records"]] == [
        7,
        7,
    ]
    assert [item["landing_interface"] for item in sparse_journal["shooting"]["trial_records"]] == [
        2,
        2,
    ]

    # 34 CV observations occur. Sparse persistence publishes 15 unique full states:
    # initial + 2 equilibration + 4 basin + 2 trial starts + 6 shot endpoints.
    # Committing every CV observation publishes 37. Event states are included once in
    # both counts and remain SHA-bound even after restart-state garbage collection.
    assert len(_audit_state_ids(sparse_root)) == 15
    assert len(_audit_state_ids(eager_root)) == 37
    assert len(_audit_state_ids(sparse_root)) < len(_audit_state_ids(eager_root)) / 2


@pytest.mark.parametrize(
    ("failure_call", "expected_phase"),
    ((6, "equilibration"), (15, "basin"), (25, "shooting")),
)
def test_sparse_phase_resume_matches_uninterrupted_deterministic_events(
    tmp_path: Path,
    failure_call: int,
    expected_phase: str,
) -> None:
    baseline_root = tmp_path / f"baseline_{expected_phase}"
    baseline_engine = SparsePhaseJumpyEngine()
    baseline_result = run_jumpy_ffs(
        _initial_scripted_state(baseline_engine),
        algorithm=_sparse_phase_algorithm(),
        engine=baseline_engine,
        collective_variable=PositionClusterCV(),
        output_root=baseline_root,
        resume=False,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )

    resumed_root = tmp_path / f"resumed_{expected_phase}"
    interrupted_engine = SparsePhaseJumpyEngine(
        fail_on_advance_call=failure_call
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_jumpy_ffs(
            _initial_scripted_state(interrupted_engine),
            algorithm=_sparse_phase_algorithm(),
            engine=interrupted_engine,
            collective_variable=PositionClusterCV(),
            output_root=resumed_root,
            resume=False,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )
    interrupted_journal = json.loads(
        (resumed_root / "journal.json").read_text(encoding="utf-8")
    )
    assert interrupted_journal["phase"] == expected_phase
    if expected_phase == "equilibration":
        assert interrupted_journal["equilibration"]["completed_steps"] == 4
    elif expected_phase == "basin":
        # The uncommitted step 8 rearmed the crossing detector, but the durable event
        # remains step 7/armed=false. Resume recomputes the rearm and counts step 14 once.
        assert interrupted_journal["basin"]["total_steps"] == 7
        assert interrupted_journal["basin"]["armed"] is False
        assert len(interrupted_journal["basin"]["exits"]) == 1
    else:
        assert interrupted_journal["shooting"]["active_trial"]["elapsed_steps"] == 3

    resume_engine = SparsePhaseJumpyEngine()
    resumed_result = run_jumpy_ffs(
        _initial_scripted_state(resume_engine),
        algorithm=_sparse_phase_algorithm(),
        engine=resume_engine,
        collective_variable=PositionClusterCV(),
        output_root=resumed_root,
        resume=True,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )
    resumed_journal = json.loads(
        (resumed_root / "journal.json").read_text(encoding="utf-8")
    )
    baseline_journal = json.loads(
        (baseline_root / "journal.json").read_text(encoding="utf-8")
    )
    assert resumed_result.to_dict() == baseline_result.to_dict()
    assert resumed_journal == baseline_journal


def test_sparse_resume_does_not_require_bitwise_discarded_path_replay(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "non_bitwise_resume"
    algorithm = _sparse_phase_algorithm(basin_max_steps=15)
    interrupted_engine = SparsePhaseJumpyEngine(fail_on_advance_call=15)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_jumpy_ffs(
            _initial_scripted_state(interrupted_engine),
            algorithm=algorithm,
            engine=interrupted_engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=False,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )

    # Model a legal non-bitwise continuation from the same committed phase-space/RNG
    # state: the second threshold observation moves by one CV interval. The engine
    # contract is unchanged, as it would be for CUDA reduction-order variation. No
    # volatile event was journaled, so recovery remains coherent without assuming that
    # the discarded path is reproduced.
    resumed_engine = SparsePhaseJumpyEngine(
        second_basin_crossing_absolute_step=21
    )
    result = run_jumpy_ffs(
        _initial_scripted_state(resumed_engine),
        algorithm=algorithm,
        engine=resumed_engine,
        collective_variable=PositionClusterCV(),
        output_root=output_root,
        resume=True,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )
    journal = json.loads(
        (output_root / "journal.json").read_text(encoding="utf-8")
    )
    assert result.basin_crossing_count == 2
    assert [item["root_id"] for item in journal["basin"]["exits"]] == [
        "exit_000000",
        "exit_000001",
    ]
    assert [item["crossing_step"] for item in journal["basin"]["exits"]] == [
        7,
        15,
    ]
    assert [item["interval_steps"] for item in journal["basin"]["exits"]] == [
        7,
        8,
    ]


def test_resume_re_raises_terminal_invalid_equilibration_early_b(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "equilibration_early_b"
    algorithm = _early_b_algorithm(equilibration_steps=1)
    engine = ScriptedJumpyEngine()
    with pytest.raises(RuntimeError, match="reached the B interface during pre-flux"):
        run_jumpy_ffs(
            _initial_scripted_state(engine),
            algorithm=algorithm,
            engine=engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=False,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )
    assert engine.advance_calls == 1
    terminal_journal = json.loads(
        (output_root / "journal.json").read_text(encoding="utf-8")
    )
    terminal = terminal_journal["terminal_invalid"]
    assert terminal_journal["phase"] == "terminal_invalid"
    assert terminal == {
        "origin_phase": "equilibration",
        "reason_code": "reached_b_during_equilibration",
        "message": (
            "The trajectory reached the B interface during pre-flux equilibration at "
            "step=1, largest_cluster_atoms=2. The starting liquid is not a valid "
            "metastable-basin state for this jFFS run."
        ),
        "state_id": terminal["state_id"],
        "observation_step": 1,
        "dynamical_step": 1,
        "cv": 2,
    }
    assert {path.stem for path in (output_root / "states").glob("*.npz")} == {
        terminal["state_id"]
    }

    resume_engine = ScriptedJumpyEngine()
    with pytest.raises(
        RuntimeError,
        match="Cannot resume terminal-invalid.*reached the B interface",
    ):
        run_jumpy_ffs(
            _initial_scripted_state(resume_engine),
            algorithm=algorithm,
            engine=resume_engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=True,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )
    assert resume_engine.advance_calls == 0
    assert json.loads(
        (output_root / "journal.json").read_text(encoding="utf-8")
    ) == terminal_journal


def test_resume_re_raises_terminal_invalid_basin_early_b(tmp_path: Path) -> None:
    output_root = tmp_path / "basin_early_b"
    algorithm = _early_b_algorithm(equilibration_steps=0)
    engine = ScriptedJumpyEngine()
    with pytest.raises(RuntimeError, match="reached B before collecting"):
        run_jumpy_ffs(
            _initial_scripted_state(engine),
            algorithm=algorithm,
            engine=engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=False,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )
    assert engine.advance_calls == 1
    terminal_journal = json.loads(
        (output_root / "journal.json").read_text(encoding="utf-8")
    )
    terminal = terminal_journal["terminal_invalid"]
    assert terminal_journal["phase"] == "terminal_invalid"
    assert terminal["origin_phase"] == "basin"
    assert terminal["reason_code"] == "reached_b_before_basin_flux_target"
    assert terminal["observation_step"] == 1
    assert terminal["dynamical_step"] == 1
    assert terminal["cv"] == 2
    assert terminal["state_id"] == terminal_journal["basin"]["current_state_id"]
    assert terminal_journal["basin"]["exits"] == [
        {
            "root_id": "exit_000000",
            "state_id": terminal["state_id"],
            "crossing_step": 1,
            "interval_steps": 1,
            "cv": 2,
            "landing_interface": 1,
        }
    ]
    assert {path.stem for path in (output_root / "states").glob("*.npz")} == {
        terminal["state_id"]
    }

    resume_engine = ScriptedJumpyEngine()
    with pytest.raises(
        RuntimeError,
        match="Cannot resume terminal-invalid.*reached B before collecting",
    ):
        run_jumpy_ffs(
            _initial_scripted_state(resume_engine),
            algorithm=algorithm,
            engine=resume_engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=True,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )
    assert resume_engine.advance_calls == 0
    assert json.loads(
        (output_root / "journal.json").read_text(encoding="utf-8")
    ) == terminal_journal


@pytest.mark.parametrize(
    ("branch_outcome", "expected_probability", "reason_fragment"),
    (
        ("all_failure", 0.0, "No A-to-B success"),
        ("all_success", 1.0, "No A-to-B failure"),
    ),
)
def test_boundary_branching_sample_has_undefined_uncertainty(
    tmp_path: Path,
    branch_outcome: str,
    expected_probability: float,
    reason_fragment: str,
) -> None:
    engine = ScriptedJumpyEngine(branch_outcome=branch_outcome)
    output_root = tmp_path / branch_outcome
    result = run_jumpy_ffs(
        _initial_scripted_state(engine),
        algorithm=_algorithm(),
        engine=engine,
        collective_variable=PositionClusterCV(),
        output_root=output_root,
        resume=False,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )

    assert result.crossing_probability == pytest.approx(expected_probability)
    assert result.uncertainty_status == "undefined_inadequate_boundary_sample"
    assert result.uncertainty_reason is not None
    assert reason_fragment in result.uncertainty_reason
    assert result.confidence_level is None
    assert result.rate_confidence_interval_per_A3_ps is None
    assert result.rate_bootstrap_standard_error_per_A3_ps is None
    assert result.bootstrap_samples_used == 0
    assert (
        result.scientific_scope["rate_uncertainty_status"]
        == "undefined_inadequate_boundary_sample"
    )
    assert reason_fragment in result.scientific_scope["rate_uncertainty_limitation"]

    persisted = json.loads(
        (output_root / "result.json").read_text(encoding="utf-8")
    )
    assert persisted["confidence_level"] is None
    assert persisted["rate_confidence_interval_per_A3_ps"] is None
    assert persisted["rate_bootstrap_standard_error_per_A3_ps"] is None
    assert persisted["uncertainty_status"] == result.uncertainty_status

    resumed_engine = ScriptedJumpyEngine(branch_outcome=branch_outcome)
    resumed = run_jumpy_ffs(
        _initial_scripted_state(resumed_engine),
        algorithm=_algorithm(),
        engine=resumed_engine,
        collective_variable=PositionClusterCV(),
        output_root=output_root,
        resume=True,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )
    assert resumed.uncertainty_status == result.uncertainty_status
    assert resumed.confidence_level is None
    assert resumed.rate_confidence_interval_per_A3_ps is None


def test_jumpy_ffs_resumes_exactly_after_cadence_interruption(tmp_path: Path) -> None:
    output_root = tmp_path / "resumed"
    interrupted_engine = ScriptedJumpyEngine(fail_on_advance_call=2)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_jumpy_ffs(
            _initial_scripted_state(interrupted_engine),
            algorithm=_algorithm(),
            engine=interrupted_engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=False,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )
    interrupted_journal = json.loads(
        (output_root / "journal.json").read_text(encoding="utf-8")
    )
    assert interrupted_journal["phase"] == "basin"
    assert interrupted_journal["basin"]["total_steps"] == 1
    assert len(interrupted_journal["basin"]["exits"]) == 1
    required_state_ids = {
        interrupted_journal["basin"]["current_state_id"],
        *(item["state_id"] for item in interrupted_journal["basin"]["exits"]),
    }
    state_directory = output_root / "states"
    assert {path.stem for path in state_directory.glob("*.npz")} == required_state_ids

    # Simulate a process crash after publishing a new content-addressed blob but before
    # committing a journal that references it. Resume must retain the live state, collect
    # this orphan, and preserve its SHA in the append-only audit log.
    orphan_state_id = "f" * 64
    live_state_path = next(state_directory.glob("*.npz"))
    shutil.copyfile(live_state_path, state_directory / f"{orphan_state_id}.npz")
    with (output_root / "state_sha256_audit.log").open(
        "a", encoding="ascii"
    ) as handle:
        handle.write(f"{orphan_state_id}\n")

    resumed_engine = ScriptedJumpyEngine()
    resumed = run_jumpy_ffs(
        _initial_scripted_state(resumed_engine),
        algorithm=_algorithm(),
        engine=resumed_engine,
        collective_variable=PositionClusterCV(),
        output_root=output_root,
        resume=True,
        source_evidence=TEST_SOURCE_EVIDENCE,
        scientific_scope=TEST_SCIENTIFIC_SCOPE,
        progress=lambda _message: None,
    )
    assert resumed.crossing_probability == pytest.approx(0.375)
    assert resumed.trial_count == 6
    assert list(state_directory.glob("*.npz")) == []
    assert orphan_state_id in (
        output_root / "state_sha256_audit.log"
    ).read_text(encoding="ascii").splitlines()


def test_resume_rejects_changed_source_evidence_and_initial_state(tmp_path: Path) -> None:
    output_root = tmp_path / "identity"
    engine = ScriptedJumpyEngine(fail_on_advance_call=2)
    initial_state = _initial_scripted_state(engine)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_jumpy_ffs(
            initial_state,
            algorithm=_algorithm(),
            engine=engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=False,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )

    resume_engine = ScriptedJumpyEngine()
    with pytest.raises(RuntimeError, match="does not match the stored manifest"):
        run_jumpy_ffs(
            _initial_scripted_state(resume_engine),
            algorithm=_algorithm(),
            engine=resume_engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=True,
            source_evidence={"source": "changed source artifact"},
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )

    changed_state = _initial_scripted_state(resume_engine)
    changed_momenta = changed_state.momenta.copy()
    changed_momenta[0, 0] = 1.0
    changed_state = DynamicalState(
        positions_A=changed_state.positions_A,
        momenta=changed_momenta,
        cell_A=changed_state.cell_A,
        atomic_numbers=changed_state.atomic_numbers,
        masses=changed_state.masses,
        pbc=changed_state.pbc,
        step=changed_state.step,
        rng_state=changed_state.rng_state,
        integrator_contract=changed_state.integrator_contract,
    )
    with pytest.raises(RuntimeError, match="does not match the stored manifest"):
        run_jumpy_ffs(
            changed_state,
            algorithm=_algorithm(),
            engine=resume_engine,
            collective_variable=PositionClusterCV(),
            output_root=output_root,
            resume=True,
            source_evidence=TEST_SOURCE_EVIDENCE,
            scientific_scope=TEST_SCIENTIFIC_SCOPE,
            progress=lambda _message: None,
        )


def test_mtk_npt_branching_without_extended_state_is_refused() -> None:
    with pytest.raises(RuntimeError, match="refusing MTK-NPT branching"):
        require_branchable_integrator(
            {
                "ensemble": "mtk_npt",
                "extended_integrator_state_serialized": False,
            },
            context="test",
        )


def test_langevin_nvt_cadence_restart_is_phase_space_exact() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    engine = LangevinNVTShotEngine(
        EMT(),
        execution_provenance={"source": "test", "identity": "ASE EMT"},
        temperature_K=500.0,
        timestep_fs=1.0,
        friction_time_fs=100.0,
    )
    state = engine.initialize_state(atoms, random_seed=19)
    continuous = engine.advance(state, steps=4)
    segmented = engine.advance(engine.advance(state, steps=2), steps=2)

    assert np.array_equal(segmented.positions_A, continuous.positions_A)
    assert np.array_equal(segmented.momenta, continuous.momenta)
    assert np.array_equal(segmented.cell_A, continuous.cell_A)
    assert segmented.rng_state == continuous.rng_state


def test_production_jumpy_ffs_config_is_explicit_and_nvt(tmp_path: Path) -> None:
    config = load_jumpy_ffs_config(
        _write_selected_jffs_config(tmp_path, PRODUCTION_CONFIG)
    )
    assert config.algorithm.interfaces_atoms == (10, 15, 22, 32, 45, 65, 100)
    assert config.algorithm.cv_interval_steps == 20
    assert config.algorithm.equilibration_checkpoint_interval_steps == 20000
    assert config.algorithm.basin_checkpoint_interval_steps == 20000
    assert config.algorithm.basin_target_crossings == 100
    assert config.algorithm.trials_per_state == 4
    assert config.algorithm.shot_checkpoint_interval_steps == 20000
    assert config.temperature_K == 500.0
    assert config.timestep_fs == 1.0
    assert config.friction_time_fs == 100.0
    assert config.shot_md_property_mode == "forces"
    assert config.generator.dynamics.target_temperature_K == 500.0
    assert config.generator.dynamics.pressure_GPa == 0.0
    assert config.generator.potential.md_property_mode == "forces_stress"
    assert config.generator.system.repetitions == (16, 16, 16)
    assert config.source_dataset.name == "al_liquid_source_16384_compiled_mpa_500K"
    assert config.potential_selection_report.is_file()
    assert len(config.potential_selection_report_sha256) == 64
    assert len(config.selected_generator_config_sha256) == 64
    assert (
        config.to_dict()["potential_selection_report_sha256"]
        == config.potential_selection_report_sha256
    )


@pytest.mark.parametrize(
    ("location", "key"),
    (
        ("root", "equilibration_checkpoint_interval_steps"),
        ("basin", "checkpoint_interval_steps"),
        ("shooting", "checkpoint_interval_steps"),
    ),
)
def test_checkpoint_cadences_must_align_with_cv_observations(
    tmp_path: Path,
    location: str,
    key: str,
) -> None:
    selected_path = _write_selected_jffs_config(
        tmp_path / "selection", PRODUCTION_CONFIG
    )
    raw = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    if location == "root":
        raw[key] = 21
    else:
        raw[location][key] = 21
    invalid_path = tmp_path / f"invalid_{location}.yaml"
    invalid_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must be divisible by cv_interval_steps"):
        load_jumpy_ffs_config(invalid_path)


def test_jumpy_ffs_config_refuses_mtk_npt(tmp_path: Path) -> None:
    raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["ensemble"] = "mtk_npt"
    path = tmp_path / "npt.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="thermostat-chain and barostat state"):
        load_jumpy_ffs_config(path)


def test_mh1_jumpy_ffs_candidate_is_model_and_output_isolated(
    tmp_path: Path,
) -> None:
    mpa = load_jumpy_ffs_config(
        _write_selected_jffs_config(tmp_path / "mpa", PRODUCTION_CONFIG)
    )
    mh1 = load_jumpy_ffs_config(
        _write_selected_jffs_config(tmp_path / "mh1", MH1_CONFIG)
    )
    assert mh1.generator.potential.model_name == "mace-mh-1-omat-pbe"
    assert mh1.generator.potential.head == "omat_pbe"
    assert mh1.generator.potential.sha256 != mpa.generator.potential.sha256
    assert mh1.source_dataset != mpa.source_dataset
    assert mh1.output_root != mpa.output_root
    assert mh1.algorithm == mpa.algorithm


def test_jumpy_ffs_selection_report_strictly_gates_model_and_config(
    tmp_path: Path,
) -> None:
    missing_raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    missing_raw["potential_selection_report"] = str(tmp_path / "missing.json")
    missing_path = tmp_path / "missing_report.yaml"
    missing_path.write_text(yaml.safe_dump(missing_raw), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="will not fall back"):
        load_jumpy_ffs_config(missing_path)

    bad_schema_path = _write_selected_jffs_config(
        tmp_path / "schema",
        PRODUCTION_CONFIG,
        schema_version=1,
    )
    with pytest.raises(RuntimeError, match="expected schema_version=4"):
        load_jumpy_ffs_config(bad_schema_path)

    bad_type_path = _write_selected_jffs_config(
        tmp_path / "type",
        PRODUCTION_CONFIG,
        report_type="unrelated_selection_report",
    )
    with pytest.raises(RuntimeError, match="report_type='al_crystallization"):
        load_jumpy_ffs_config(bad_type_path)

    bad_policy_path = _write_selected_jffs_config(
        tmp_path / "policy",
        PRODUCTION_CONFIG,
        policy_version="obsolete_relative_speed_policy",
    )
    with pytest.raises(RuntimeError, match="policy_version"):
        load_jumpy_ffs_config(bad_policy_path)

    bad_sha_path = _write_selected_jffs_config(
        tmp_path / "sha",
        PRODUCTION_CONFIG,
        selected_config_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="changed after selection"):
        load_jumpy_ffs_config(bad_sha_path)

    bad_name_path = _write_selected_jffs_config(
        tmp_path / "name",
        PRODUCTION_CONFIG,
        selected_model_name="not-the-selected-model",
    )
    with pytest.raises(RuntimeError, match="selected_model_name"):
        load_jumpy_ffs_config(bad_name_path)

    mh1_generator = _generator_path_from_jffs(MH1_CONFIG)
    wrong_config_path = _write_selected_jffs_config(
        tmp_path / "wrong_config",
        PRODUCTION_CONFIG,
        selected_generator_path=mh1_generator,
    )
    with pytest.raises(RuntimeError, match="selection report chose generator config"):
        load_jumpy_ffs_config(wrong_config_path)


def test_mh1_config_refuses_report_that_selected_mpa(tmp_path: Path) -> None:
    mpa_generator = _generator_path_from_jffs(PRODUCTION_CONFIG)
    config_path = _write_selected_jffs_config(
        tmp_path,
        MH1_CONFIG,
        selected_generator_path=mpa_generator,
    )
    with pytest.raises(RuntimeError, match="selection report chose generator config"):
        load_jumpy_ffs_config(config_path)
