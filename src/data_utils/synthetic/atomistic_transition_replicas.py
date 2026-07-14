"""Run independent replicas of one direct-coexistence transition recipe."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .atomistic.generator import build_calculator
from .atomistic.transition_config import load_transition_config
from .atomistic.transition_generator import generate_transition_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run transition replicas sequentially while reusing one MACE calculator."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--replica",
        action="append",
        nargs=2,
        required=True,
        metavar=("RANDOM_SEED", "OUTPUT_ROOT"),
    )
    args = parser.parse_args()

    base_config = load_transition_config(args.config)
    calculator = build_calculator(base_config.generator)
    for seed_text, output_text in args.replica:
        random_seed = int(seed_text)
        output_root = Path(output_text).expanduser().resolve()
        replica_config = replace(
            base_config,
            random_seed=random_seed,
            output=replace(
                base_config.output,
                root_dir=output_root,
                overwrite=False,
            ),
        )
        manifest_path = output_root / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest["config"] != replica_config.to_dict():
                raise RuntimeError(
                    f"Completed replica configuration does not match the requested run: "
                    f"{manifest_path}."
                )
            print(
                f"Replica already complete: random_seed={random_seed}, "
                f"output={output_root}"
            )
            continue
        print(f"Starting transition replica: random_seed={random_seed}, output={output_root}")
        generate_transition_dataset(replica_config, calculator=calculator)


if __name__ == "__main__":
    main()
