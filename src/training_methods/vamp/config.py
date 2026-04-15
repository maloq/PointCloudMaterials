from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def project_root() -> Path:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(f"Could not resolve project root from {this_file}.")


PROJECT_ROOT = project_root()


def resolve_vamp_config_path(config_arg: str) -> Path:
    candidate = Path(str(config_arg)).expanduser()
    if candidate.exists():
        return candidate.resolve()

    configs_root = (PROJECT_ROOT / "configs" / "vamp").resolve()
    named_candidate = configs_root / str(config_arg)
    if named_candidate.exists():
        return named_candidate.resolve()
    if named_candidate.suffix == "":
        yaml_candidate = named_candidate.with_suffix(".yaml")
        if yaml_candidate.exists():
            return yaml_candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve VAMP config path. "
        f"Tried {config_arg!r} relative to the current working directory and to {configs_root}."
    )


def load_vamp_config(config_arg: str) -> tuple[DictConfig, Path, Path]:
    resolved = resolve_vamp_config_path(config_arg)
    cfg = OmegaConf.load(resolved)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a DictConfig from {resolved}, got {type(cfg)!r}.")
    OmegaConf.resolve(cfg)
    return cfg, resolved, resolved.parent


def resolve_path(path: str | None, *, base_dir: Path) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if text == "":
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    cwd_candidate = (Path.cwd() / candidate).resolve()
    config_candidate = (base_dir / candidate).resolve()
    project_candidate = (PROJECT_ROOT / candidate).resolve()

    for resolved_candidate in (cwd_candidate, config_candidate, project_candidate):
        if resolved_candidate.exists():
            return str(resolved_candidate)

    return str(project_candidate)
