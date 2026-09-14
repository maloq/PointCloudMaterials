"""Convert repository trajectory formats and audit converted data.

Source files are retained by default. In-place conversions publish the existing
reader metadata; only --delete-source removes verified source dumps.
"""
from src.command_line import dispatch

COMMANDS = {
    "relaxation": "src.data.conversion.relaxation",
    "temporal-storage": "src.data.conversion.position_storage",
    "training-cache": "src.data.conversion.training_cache",
    "embedding-cache": "src.data.conversion.embedding_cache",
    "elemental": "src.data.conversion.elemental",
    "shooting": "src.data.conversion.shooting",
    "temporal": "src.data.conversion.temporal",
    "export-shooting": "src.data.conversion.shooting_export",
    "export-npz-dump": "src.data.conversion.npz_dump",
    "audit-shooting": "src.data.conversion.audit_shooting",
    "audit-temporal": "src.data.conversion.audit_temporal",
}


def main(argv=None):
    return dispatch(COMMANDS, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
