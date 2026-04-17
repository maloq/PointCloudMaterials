from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .utils import build_mlp, lag_to_key, mean_by_offsets


class TemporalContextModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        dropout: float,
        context_frames: int,
        model_type: str,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.context_frames = int(context_frames)
        self.model_type = str(model_type).strip().lower()

        if self.context_frames <= 0:
            raise ValueError(f"context_frames must be > 0, got {self.context_frames}.")
        if self.model_type not in {"transformer", "gru", "mlp"}:
            raise ValueError(
                "tmf.temporal.model_type must be one of {'transformer', 'gru', 'mlp'}, "
                f"got {self.model_type!r}."
            )

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_embed = nn.Parameter(torch.zeros(self.context_frames, self.hidden_dim))
        nn.init.normal_(self.position_embed, mean=0.0, std=self.hidden_dim ** -0.5)

        if self.model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=4 * self.hidden_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.model = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=max(1, self.depth),
                enable_nested_tensor=False,
            )
        elif self.model_type == "gru":
            self.model = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=max(1, self.depth),
                batch_first=True,
                dropout=self.dropout if self.depth > 1 else 0.0,
            )
        else:
            self.model = build_mlp(
                self.context_frames * self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                num_layers=max(2, self.depth),
                dropout=self.dropout,
            )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(self, context_seq: torch.Tensor) -> torch.Tensor:
        if context_seq.ndim != 3 or int(context_seq.shape[1]) != self.context_frames:
            raise ValueError(
                "TemporalContextModel expects context_seq with shape (B, C, D). "
                f"Expected C={self.context_frames}, got {tuple(context_seq.shape)}."
            )

        x = self.input_proj(context_seq)
        x = x + self.position_embed.unsqueeze(0).to(device=x.device, dtype=x.dtype)
        if self.model_type == "transformer":
            mask = torch.triu(
                torch.ones(self.context_frames, self.context_frames, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            encoded = self.model(x, mask=mask)
            return encoded[:, -1, :]
        if self.model_type == "gru":
            _, hidden = self.model(x)
            return hidden[-1]
        return self.model(x.reshape(int(x.shape[0]), -1))


class MultiLagFuturePredictor(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        hidden_dim: int,
        stable_dim: int,
        residual_dim: int,
        lags: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.lags = [int(lag) for lag in lags]
        self.stable_heads = nn.ModuleDict()
        self.residual_heads = nn.ModuleDict()
        self.hazard_heads = nn.ModuleDict()

        for lag in self.lags:
            key = lag_to_key(lag)
            self.stable_heads[key] = build_mlp(
                context_dim,
                hidden_dim,
                stable_dim,
                num_layers=2,
                dropout=dropout,
            )
            self.residual_heads[key] = build_mlp(
                context_dim,
                hidden_dim,
                residual_dim,
                num_layers=2,
                dropout=dropout,
            )
            self.hazard_heads[key] = build_mlp(
                context_dim,
                hidden_dim,
                1,
                num_layers=2,
                dropout=dropout,
            )

    def forward(self, context_repr: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        stable_logits: dict[int, torch.Tensor] = {}
        residual_pred: dict[int, torch.Tensor] = {}
        hazard_logits: dict[int, torch.Tensor] = {}
        for lag in self.lags:
            key = lag_to_key(lag)
            stable_logits[lag] = self.stable_heads[key](context_repr)
            residual_pred[lag] = self.residual_heads[key](context_repr)
            hazard_logits[lag] = self.hazard_heads[key](context_repr)
        return stable_logits, residual_pred, hazard_logits


class MotifFieldHead(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        stable_dim: int,
        residual_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.neighbor_encoder = build_mlp(
            stable_dim + residual_dim,
            hidden_dim,
            hidden_dim,
            num_layers=max(2, depth),
            dropout=0.0,
        )
        self.predictor = build_mlp(
            context_dim + stable_dim + hidden_dim,
            hidden_dim,
            stable_dim,
            num_layers=max(2, depth),
            dropout=0.0,
        )

    def forward(
        self,
        *,
        context_repr: torch.Tensor,
        anchor_stable_probs: torch.Tensor,
        neighbor_stable_probs: torch.Tensor,
        neighbor_residual: torch.Tensor,
        neighbor_offsets: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_features = torch.cat(
            [neighbor_stable_probs, neighbor_residual],
            dim=-1,
        )
        encoded_neighbors = self.neighbor_encoder(neighbor_features)
        pooled_neighbors = mean_by_offsets(encoded_neighbors, neighbor_offsets)
        field_input = torch.cat([context_repr, anchor_stable_probs, pooled_neighbors], dim=-1)
        return self.predictor(field_input)
