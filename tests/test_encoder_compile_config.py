import pytest

from src.training_methods.base_ssl_module import _validate_encoder_compile_mode


def test_geoframe_rejects_reduce_overhead_compile_mode() -> None:
    with pytest.raises(ValueError, match="call-order-dependent"):
        _validate_encoder_compile_mode(
            encoder_name="GeoFrameTransformer",
            compile_enabled=True,
            compile_mode="reduce-overhead",
        )


@pytest.mark.parametrize(
    ("compile_enabled", "compile_mode"),
    [
        (True, "default"),
        (False, "reduce-overhead"),
    ],
)
def test_geoframe_accepts_safe_compile_configuration(
    compile_enabled: bool,
    compile_mode: str,
) -> None:
    _validate_encoder_compile_mode(
        encoder_name="GeoFrameTransformer",
        compile_enabled=compile_enabled,
        compile_mode=compile_mode,
    )
