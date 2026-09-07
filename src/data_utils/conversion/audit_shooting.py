"""Read-only verification of converted shooting branches and retained sources."""
import argparse
import json
from pathlib import Path

from src.data_utils.conversion.shooting import (
    _expected_timesteps,
    _load_json_object,
    _sha256_file,
    _validate_binary_for_migration,
)
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.shooting_dataset import validate_complete_shooting_branch


def audit_campaign(root: Path, *, require_source_deleted: bool = False) -> dict:
    manifest = _load_json_object(root / "manifest.json")
    complete = 0
    incomplete = 0
    retained = 0
    for branch in manifest["branches"]:
        branch_dir = root / branch["branch_dir"]
        outcome_path = branch_dir / "outcome.json"
        if not outcome_path.is_file():
            incomplete += 1
            continue
        outcome = _load_json_object(outcome_path)
        if outcome["state"] != "complete":
            incomplete += 1
            continue
        validate_complete_shooting_branch(root, manifest, branch, outcome)
        if "trajectory_artifact" not in outcome:
            raise RuntimeError(f"Complete branch has not been converted: {branch_dir}")
        artifact = outcome["trajectory_artifact"]
        source_record = artifact["source_lammpstrj"]
        source = branch_dir / "trajectory.lammpstrj"
        binary = ShootingBinaryTrajectory.load(branch_dir / artifact["path"])
        _validate_binary_for_migration(
            binary,
            source_size_bytes=source_record["size_bytes"],
            source_sha256=source_record["sha256"],
            expected_timesteps=_expected_timesteps(manifest),
            atom_count=manifest["atom_count"],
        )
        if source.exists():
            if source_record["deleted"] or require_source_deleted:
                raise RuntimeError(f"Source dump should have been deleted: {source}")
            if _sha256_file(source) != source_record["sha256"]:
                raise RuntimeError(f"Retained source checksum changed: {source}")
            retained += 1
        elif not source_record["deleted"]:
            raise RuntimeError(f"Retained source is missing: {source}")
        complete += 1
    return dict(campaign_root=str(root), verified_branches=complete,
                incomplete_branches=incomplete, retained_sources=retained)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_roots", nargs="+", type=Path)
    parser.add_argument("--require-source-deleted", action="store_true")
    args = parser.parse_args(argv)
    reports = [audit_campaign(root.expanduser().resolve(), require_source_deleted=args.require_source_deleted)
               for root in args.campaign_roots]
    print(json.dumps(reports, indent=2))
