import warnings

from omegaconf import OmegaConf


_MISSING = object()


def _get_config_value(cfg, field: str):
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, field, default=_MISSING)
    return getattr(cfg, field, _MISSING)


def warn_ignored_config_fields(cfg, fields: tuple[str, ...], *, reason: str) -> None:
    configured = []
    for field in fields:
        value = _get_config_value(cfg, field)
        if value is _MISSING or value is None:
            continue
        configured.append(field)

    if not configured:
        return

    noun = "field" if len(configured) == 1 else "fields"
    joined = ", ".join(configured)
    warnings.warn(
        f"Ignoring configured {noun} {joined}: {reason}",
        stacklevel=3,
    )


def warn_common_view_sampler_ignored_fields(
    cfg,
    *,
    prefix: str,
    jitter_std: float,
    jitter_mode: str,
    neighbor_view: bool,
    neighbor_view_mode: str,
    drop_ratio: float,
    rotation_mode: str,
    strain_std: float,
    occlusion_mode: str,
) -> None:
    def field(name: str) -> str:
        return f"{prefix}_{name}"

    jitter_mode = str(jitter_mode).lower()
    neighbor_view_mode = str(neighbor_view_mode).lower()
    rotation_mode = str(rotation_mode).lower()
    occlusion_mode = str(occlusion_mode).lower()

    if float(jitter_std) <= 0.0:
        warn_ignored_config_fields(
            cfg,
            (field("jitter_mode"), field("jitter_scale")),
            reason=f"{field('jitter_std')}={float(jitter_std)} disables jitter.",
        )
    elif jitter_mode != "physical":
        warn_ignored_config_fields(
            cfg,
            (field("jitter_scale"),),
            reason=f"{field('jitter_mode')}={jitter_mode!r} uses unit jitter scaling.",
        )

    if not bool(neighbor_view):
        warn_ignored_config_fields(
            cfg,
            (
                field("neighbor_view_mode"),
                field("neighbor_k"),
                field("neighbor_max_relative_distance"),
            ),
            reason=f"{field('neighbor_view')}=false disables neighbor-shifted views.",
        )
    elif neighbor_view_mode == "none":
        warn_ignored_config_fields(
            cfg,
            (
                field("neighbor_k"),
                field("neighbor_max_relative_distance"),
            ),
            reason=f"{field('neighbor_view_mode')}='none' disables neighbor shifts.",
        )

    if float(drop_ratio) <= 0.0:
        warn_ignored_config_fields(
            cfg,
            (field("drop_apply_to_both"),),
            reason=f"{field('drop_ratio')}={float(drop_ratio)} disables point dropping.",
        )
    elif (not bool(neighbor_view)) or neighbor_view_mode == "none":
        warn_ignored_config_fields(
            cfg,
            (field("drop_apply_to_both"),),
            reason=(
                f"{field('drop_apply_to_both')} only affects neighbor-view dropping, "
                "but no neighbor view is sampled."
            ),
        )

    if rotation_mode == "none":
        warn_ignored_config_fields(
            cfg,
            (field("rotation_deg"),),
            reason=f"{field('rotation_mode')}='none' disables rotations.",
        )
    elif rotation_mode == "full":
        warn_ignored_config_fields(
            cfg,
            (field("rotation_deg"),),
            reason=f"{field('rotation_mode')}='full' ignores the max-angle setting.",
        )

    if float(strain_std) <= 0.0:
        warn_ignored_config_fields(
            cfg,
            (field("strain_volume_preserve"),),
            reason=f"{field('strain_std')}={float(strain_std)} disables strain augmentation.",
        )

    if occlusion_mode == "none":
        warn_ignored_config_fields(
            cfg,
            (
                field("occlusion_view"),
                field("occlusion_slab_frac"),
                field("occlusion_cone_deg"),
                field("occlusion_prob"),
            ),
            reason=f"{field('occlusion_mode')}='none' disables occlusion.",
        )
    elif occlusion_mode == "slab":
        warn_ignored_config_fields(
            cfg,
            (field("occlusion_cone_deg"),),
            reason=f"{field('occlusion_mode')}='slab' does not use cone angle.",
        )
    elif occlusion_mode == "cone":
        warn_ignored_config_fields(
            cfg,
            (field("occlusion_slab_frac"),),
            reason=f"{field('occlusion_mode')}='cone' does not use slab thickness.",
        )


def warn_fixed_invariant_fields(cfg, *, prefix: str) -> None:
    warn_ignored_config_fields(
        cfg,
        (
            f"{prefix}_invariant_mode",
            f"{prefix}_invariant_max_factor",
            f"{prefix}_invariant_groups",
            f"{prefix}_invariant_use_third_order",
            f"{prefix}_invariant_eps",
        ),
        reason="contrastive training now always uses norms(eq_z) when available and falls back to inv_z.",
    )


def warn_disabled_radial_fields(cfg, *, prefix: str, radial_enabled: bool) -> None:
    if bool(radial_enabled):
        return

    warn_ignored_config_fields(
        cfg,
        (
            f"{prefix}_radial_beta1",
            f"{prefix}_radial_beta2",
            f"{prefix}_radial_m",
            f"{prefix}_radial_eps",
        ),
        reason=f"{prefix}_radial_enabled=false disables radial regularization.",
    )
