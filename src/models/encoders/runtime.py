from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import Encoder


@dataclass(frozen=True)
class EncoderOutput:
    invariant: torch.Tensor
    equivariant: torch.Tensor | None
    aux: tuple[object, ...]
    raw: object


def resolve_encoder_output_dim(encoder: Encoder) -> int:
    if encoder.invariant_dim is None:
        raise RuntimeError(
            f"{encoder.__class__.__name__} does not declare its invariant_dim."
        )
    return encoder.invariant_dim


def prepare_encoder_input(encoder: Encoder, points: torch.Tensor) -> torch.Tensor:
    """Convert repository point clouds from dataset layout (B, N, 3)."""
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(
            "Repository encoder inputs must have dataset shape (B, N, 3), "
            f"got {tuple(points.shape)}."
        )
    if encoder.input_layout == "bn3":
        return points.contiguous()
    if encoder.input_layout == "b3n":
        return points.transpose(1, 2).contiguous()
    raise RuntimeError(
        f"{encoder.__class__.__name__} declares unknown input_layout={encoder.input_layout!r}."
    )


def split_encoder_output(encoder: Encoder, raw_output: object) -> EncoderOutput:
    """Apply the concrete forward-output contract declared by an encoder."""
    contract = encoder.output_contract
    if contract == "invariant":
        if not torch.is_tensor(raw_output):
            raise TypeError(
                f"{encoder.__class__.__name__} must return one invariant tensor, "
                f"got {type(raw_output)}."
            )
        return EncoderOutput(raw_output, None, (), raw_output)

    if not isinstance(raw_output, tuple) or not raw_output:
        raise TypeError(
            f"{encoder.__class__.__name__} declares output_contract={contract!r} "
            f"and must return a non-empty tuple, got {type(raw_output)}."
        )
    invariant = raw_output[0]
    if not torch.is_tensor(invariant):
        raise TypeError(
            f"{encoder.__class__.__name__} returned a non-tensor invariant value: "
            f"{type(invariant)}."
        )

    if contract == "invariant_aux":
        return EncoderOutput(invariant, None, tuple(raw_output[1:]), raw_output)
    if contract == "invariant_equivariant":
        equivariant = raw_output[1]
        if equivariant is not None and not torch.is_tensor(equivariant):
            raise TypeError(
                f"{encoder.__class__.__name__} returned a non-tensor equivariant value: "
                f"{type(equivariant)}."
            )
        return EncoderOutput(invariant, equivariant, tuple(raw_output[2:]), raw_output)
    raise RuntimeError(
        f"{encoder.__class__.__name__} declares unknown output_contract={contract!r}."
    )


def encode_point_clouds(encoder: Encoder, points: torch.Tensor) -> EncoderOutput:
    prepared = prepare_encoder_input(encoder, points)
    return split_encoder_output(encoder, encoder(prepared))


class EncoderAdapter:
    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def prepare_input(self, points: torch.Tensor) -> torch.Tensor:
        return prepare_encoder_input(self.encoder, points)

    def split_output(self, raw_output: object) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = split_encoder_output(self.encoder, raw_output)
        return output.invariant, output.equivariant

    def encode(self, points: torch.Tensor) -> EncoderOutput:
        return encode_point_clouds(self.encoder, points)
