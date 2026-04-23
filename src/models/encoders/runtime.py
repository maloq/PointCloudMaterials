from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig


@dataclass(frozen=True)
class EncoderOutput:
    invariant: torch.Tensor | None
    equivariant: torch.Tensor | None
    aux: tuple[object, ...] = ()
    raw: object | None = None

    def require_invariant(self, *, context: str) -> torch.Tensor:
        if self.invariant is None:
            raise RuntimeError(f"{context} requires an invariant latent, but the encoder returned none.")
        return self.invariant


def resolve_encoder_output_dim(cfg: DictConfig, *, encoder: nn.Module | None = None) -> int | None:
    if encoder is not None:
        invariant_dim = getattr(encoder, "invariant_dim", None)
        if invariant_dim is not None:
            return int(invariant_dim)

    if hasattr(cfg, "latent_size"):
        return int(cfg.latent_size)

    encoder_cfg = getattr(cfg, "encoder", None)
    encoder_kwargs = getattr(encoder_cfg, "kwargs", None) if encoder_cfg is not None else None
    if encoder_kwargs is not None:
        latent_size = encoder_kwargs.get("latent_size", None)
        if latent_size is not None:
            return int(latent_size)

    if encoder is not None:
        for attr_name in ("output_dim", "latent_size"):
            value = getattr(encoder, attr_name, None)
            if value is not None:
                return int(value)
    return None


def prepare_encoder_input(encoder: nn.Module, points):
    if not torch.is_tensor(points):
        if bool(getattr(encoder, "supports_precomputed_input", False)):
            return points
        raise TypeError(f"Encoder input must be a torch.Tensor, got {type(points)}.")
    if points.dim() != 3:
        raise ValueError(
            "Encoder input must have shape (B, N, 3) or (B, 3, N), "
            f"got {tuple(points.shape)}."
        )

    expects_channel_first = bool(getattr(encoder, "expects_channel_first", False))
    if expects_channel_first:
        if points.shape[1] == 3:
            return points.contiguous()
        if points.shape[-1] == 3:
            return points.transpose(1, 2).contiguous()
    else:
        if points.shape[-1] == 3:
            return points.contiguous()
        if points.shape[1] == 3:
            return points.transpose(1, 2).contiguous()

    raise ValueError(
        "Unable to infer point-cloud layout for the configured encoder. "
        f"expects_channel_first={expects_channel_first}, points.shape={tuple(points.shape)}."
    )


def split_encoder_output(enc_out) -> EncoderOutput:
    if isinstance(enc_out, EncoderOutput):
        return enc_out

    if isinstance(enc_out, (tuple, list)):
        if not enc_out:
            raise ValueError("Encoder returned an empty tuple/list output.")

        invariant = enc_out[0]
        equivariant = None
        for candidate in enc_out[1:]:
            if not (torch.is_tensor(candidate) and candidate.dim() == 3 and candidate.shape[-1] == 3):
                continue
            if torch.is_tensor(invariant) and invariant.dim() == 2:
                if candidate.shape[1] == invariant.shape[1]:
                    equivariant = candidate
                    break
                if candidate.shape[1] == 3:
                    continue
            if candidate.shape[1] != 3:
                equivariant = candidate
                break
        return EncoderOutput(
            invariant=invariant,
            equivariant=equivariant,
            aux=tuple(enc_out[1:]),
            raw=enc_out,
        )

    return EncoderOutput(invariant=enc_out, equivariant=None, raw=enc_out)


def encode_point_clouds(encoder: nn.Module, points: torch.Tensor) -> EncoderOutput:
    prepared = prepare_encoder_input(encoder, points)
    return split_encoder_output(encoder(prepared))


class EncoderAdapter:
    def __init__(self, encoder: nn.Module) -> None:
        self.encoder = encoder

    def prepare_input(self, points: torch.Tensor) -> torch.Tensor:
        return prepare_encoder_input(self.encoder, points)

    def split_output(self, enc_out):
        output = split_encoder_output(enc_out)
        return output.invariant, output.equivariant

    def encode(self, points: torch.Tensor) -> EncoderOutput:
        return encode_point_clouds(self.encoder, points)
