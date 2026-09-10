"""Convert repository trajectory formats and audit converted data.

Source files are retained by default. In-place conversions publish the existing
reader metadata; only --delete-source removes verified source dumps.
"""
from src.command_line import dispatch

COMMANDS = {
    "relaxation": "src.data_utils.conversion.relaxation",
    "temporal-storage": "src.data_utils.conversion.position_storage",
    "training-cache": "src.data_utils.conversion.training_cache",
    "elemental": "src.data_utils.conversion.elemental",
    "shooting": "src.data_utils.conversion.shooting",
    "temporal": "src.data_utils.conversion.temporal",
    "export-shooting": "src.data_utils.conversion.shooting_export",
    "export-npz-dump": "src.data_utils.conversion.npz_dump",
    "audit-shooting": "src.data_utils.conversion.audit_shooting",
    "audit-temporal": "src.data_utils.conversion.audit_temporal",
}


def main(argv=None):
    return dispatch(COMMANDS, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
