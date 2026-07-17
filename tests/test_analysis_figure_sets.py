from src.analysis.figure_sets import (
    _build_unique_snapshot_output_names,
    _sanitize_snapshot_output_name,
)


def test_snapshot_output_name_preserves_decimal_time_labels() -> None:
    assert _sanitize_snapshot_output_name("182.8ps") == "182_8ps"


def test_snapshot_output_name_strips_repository_data_suffixes() -> None:
    assert _sanitize_snapshot_output_name("000000step.npy") == "000000step"
    assert _sanitize_snapshot_output_name("trajectory.lammpstrj") == "trajectory"


def test_snapshot_output_names_remain_unique_after_sanitization() -> None:
    assert _build_unique_snapshot_output_names(["182.8ps", "182_8ps"]) == {
        "182.8ps": "182_8ps",
        "182_8ps": "182_8ps_2",
    }
