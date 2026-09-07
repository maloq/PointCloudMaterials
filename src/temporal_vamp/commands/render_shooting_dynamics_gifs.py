#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.predictability_map import load_fitted_probe
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_gifs import (
    embedding_trajectory_coordinates,
    extract_example_clouds_and_embeddings,
    render_embedding_gif,
    render_structure_gif,
    select_transition_gif_examples,
)


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Shooting-GIF configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Render periodic local structures and embedding trajectories for shooting siblings."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = _resolve_path(args.config)
    cfg: DictConfig = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    result_root = _resolve_path(_required(cfg, "result_root"))
    metrics_path = result_root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"The completed predictability result is required before GIF rendering: {metrics_path}"
        )
    output_dir = result_root / "gifs"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing GIF directory: {output_dir}")
    output_dir.mkdir()
    cache = ShootingEmbeddingCache.load(
        _resolve_path(_required(cfg, "embedding_cache"))
    )
    spec = cache.manifest["spec"]
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")
    encoder = load_frozen_encoder(
        str(spec["checkpoint"]),
        device=device,
        repeats=int(spec["encoder_repeats"]),
        seed=int(spec["encoder_seed"]),
        representation_source=str(spec["representation_source"]),
    )
    local_probe = load_fitted_probe(result_root / "models" / "input_local.pt")
    examples = select_transition_gif_examples(
        cache,
        temperatures_K=[float(value) for value in _required(cfg, "temperatures_K")],
    )
    metadata: list[dict[str, Any]] = []
    markdown = [
        "# Shooting dynamics GIFs",
        "",
        "Every displayed frame is loaded from a completed fixed-24-ps binary trajectory. Coordinates are centered with the minimum-image convention; no structural frame is interpolated.",
        "",
        "The embedding view compares frozen GeoFrame PCA with the local-only predictive bottleneck. PCA fits use optimization parents only. The predictive map is evaluated on later frames as a diagnostic; it was trained on parent states, not as a transition model.",
        "",
    ]
    for example in examples:
        print(
            f"[shooting-gifs] extracting T={example.temperature_K:g} parent={example.parent_id} atom={example.atom_id}",
            flush=True,
        )
        extracted = extract_example_clouds_and_embeddings(
            cache,
            example,
            encoder=encoder,
            frame_stride=int(_required(cfg, "frame_stride")),
            point_batch_size=int(_required(cfg, "point_batch_size")),
        )
        coordinates = embedding_trajectory_coordinates(
            cache,
            extracted,
            local_probe,
            device=device,
            batch_size=int(_required(cfg, "probe_batch_size")),
        )
        prefix = f"T{example.temperature_K:g}_parent{example.parent_position:02d}_atom{example.atom_id}"
        structure_path = output_dir / f"{prefix}_structure.gif"
        embedding_path = output_dir / f"{prefix}_embeddings.gif"
        print(f"[shooting-gifs] rendering {structure_path.name}", flush=True)
        render_structure_gif(
            example,
            extracted,
            structure_path,
            fps=int(_required(cfg, "fps")),
        )
        print(f"[shooting-gifs] rendering {embedding_path.name}", flush=True)
        render_embedding_gif(
            example,
            extracted,
            coordinates,
            embedding_path,
            fps=int(_required(cfg, "fps")),
        )
        np.savez_compressed(
            output_dir / f"{prefix}_data.npz",
            local_clouds_A=extracted["local_clouds_A"],
            embeddings=extracted["embeddings"],
            timesteps=extracted["timesteps"],
            relative_times_ps=extracted["relative_times_ps"],
            parent_embedding_2d=coordinates["parent_embedding_2d"],
            trajectory_embedding_2d=coordinates["trajectory_embedding_2d"],
            parent_predictive_2d=coordinates["parent_predictive_2d"],
            trajectory_predictive_2d=coordinates["trajectory_predictive_2d"],
        )
        outcomes = [
            {
                "branch_id": str(branch["branch_id"]),
                "momentum_index": int(branch["momentum_index"]),
                "thermostat_index": int(branch["thermostat_index"]),
                "first_passage_outcome": (
                    "censored"
                    if bool(outcome["censored"])
                    else str(outcome["first_passage_outcome"])
                ),
                "first_passage_time_ps": (
                    None
                    if bool(outcome["censored"])
                    else float(outcome["first_passage_time_ps"])
                ),
                "censored": bool(outcome["censored"]),
            }
            for branch, outcome in zip(
                extracted["branches"], extracted["outcomes"], strict=True
            )
        ]
        record = {
            "parent_position": example.parent_position,
            "parent_id": example.parent_id,
            "temperature_K": example.temperature_K,
            "center_position": example.center_position,
            "atom_id": example.atom_id,
            "terminal_embedding_dispersion": example.terminal_embedding_dispersion,
            "displayed_frames": int(extracted["timesteps"].size),
            "displayed_frame_interval_ps": float(
                extracted["relative_times_ps"][1] - extracted["relative_times_ps"][0]
            ),
            "last_time_ps": float(extracted["relative_times_ps"][-1]),
            "structure_gif": structure_path.name,
            "embedding_gif": embedding_path.name,
            "data": f"{prefix}_data.npz",
            "outcomes": outcomes,
        }
        metadata.append(record)
        markdown.extend(
            [
                f"## {example.temperature_K:g} K — parent {example.parent_position}, atom {example.atom_id}",
                "",
                f"![Periodic local structure]({structure_path.name})",
                "",
                f"![Embedding trajectories]({embedding_path.name})",
                "",
            ]
        )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    print(f"[shooting-gifs] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
