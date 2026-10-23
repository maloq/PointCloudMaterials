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
    return encoder.invariant_dim


def prepare_encoder_input(encoder: Encoder, points: torch.Tensor) -> torch.Tensor:
    """Adapt dataset (B, N, 3) clouds or (B, T, N, 3) histories."""
    if encoder.input_layout == "b3n":
        points = points.transpose(1, 2)
    return points.contiguous()


def split_encoder_output(encoder: Encoder, raw_output: object) -> EncoderOutput:
    """Apply the concrete forward-output contract declared by an encoder."""
    contract = encoder.output_contract
    if contract == "invariant":
        return EncoderOutput(raw_output, None, (), raw_output)
    if contract == "invariant_aux":
        return EncoderOutput(raw_output[0], None, raw_output[1:], raw_output)
    if contract == "invariant_equivariant":
        return EncoderOutput(raw_output[0], raw_output[1], raw_output[2:], raw_output)
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
