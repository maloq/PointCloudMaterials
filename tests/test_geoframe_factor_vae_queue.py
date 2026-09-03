from pathlib import Path

from hydra.core.override_parser.overrides_parser import OverridesParser

from scripts.run_geoframe_factor_vae_queue import FactorSetting, RunSpec, _run_command


def test_checkpoint_path_with_equals_is_quoted_for_hydra(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "GeoFrameTransformer-epoch=159.ckpt"
    checkpoint.touch()
    monkeypatch.setattr(
        "scripts.run_geoframe_factor_vae_queue.V1_CHECKPOINT", checkpoint
    )
    spec = RunSpec(
        index=1,
        label="v1_control",
        encoder="V1",
        config_name="vicreg_geo_frame_factor_vae_sweep_v1",
        experiment_name="TEST",
        epochs=1,
        factor=FactorSetting("control", False, 0.1, 0.0, 3.0e-6, 2),
        run_dir=str(tmp_path / "run"),
        wandb_run_id="test",
    )

    command = _run_command(spec, device=0, v2_checkpoint=None)
    override = next(arg for arg in command if arg.startswith("init_from_checkpoint="))
    parsed = OverridesParser.create().parse_override(override)

    assert parsed.value() == str(checkpoint)
