"""Shared setup for configuration-driven temporal prediction experiments."""
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def required(cfg: DictConfig, path: str):
    value = OmegaConf.select(cfg, path, throw_on_missing=True)
    if value is None:
        raise KeyError(f"Temporal prediction configuration requires {path!r}.")
    return value


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def prepare_run(config_path: Path) -> tuple[DictConfig, Path]:
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    output_dir = resolve_path(required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    return cfg, output_dir


def load_context_features(cfg: DictConfig):
    from src.temporal_vamp.shooting_context import ShootingContextTokenCache
    from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
    from src.temporal_vamp.shooting_multiscale import build_multiscale_feature_variants

    base_cache = ShootingEmbeddingCache.load(resolve_path(required(cfg, "base_embedding_cache")))
    context_cache = ShootingContextTokenCache.load(resolve_path(required(cfg, "context_token_cache")))
    feature_variants = build_multiscale_feature_variants(
        base_cache, context_cache,
        radial_scales_angstrom=required(cfg, "context.radial_scales_angstrom"),
    )
    return base_cache, context_cache, feature_variants


def distributional_targets(base_cache, cfg: DictConfig):
    from src.temporal_vamp.shooting_distribution import prepare_distributional_target_data

    return prepare_distributional_target_data(
        base_cache,
        horizons_ps=required(cfg, "target.horizons_ps"),
        change_pca_dim=required(cfg, "target.change_pca_dim"),
        rff_features_per_bandwidth=required(cfg, "target.rff_features_per_bandwidth"),
        bandwidth_multipliers=required(cfg, "target.bandwidth_multipliers"),
        selection_source_velocity_seeds=required(cfg, "split.selection_source_velocity_seeds"),
        seed=required(cfg, "target.seed"),
    )
