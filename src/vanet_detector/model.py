from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Chomp1d(nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, : -self.amount] if self.amount else values


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.GroupNorm(1, output_channels)
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.GroupNorm(1, output_channels)
        self.dropout = nn.Dropout(dropout)
        self.projection = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        output = self.conv1(values)
        output = self.dropout(F.gelu(self.norm1(self.chomp1(output))))
        output = self.conv2(output)
        output = self.dropout(F.gelu(self.norm2(self.chomp2(output))))
        return F.gelu(output + residual)


class AttentionPool(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.Tanh(),
            nn.Linear(dimension // 2, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class BehaviorEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        channels: list[int],
        kernel_size: int,
        gru_hidden: int,
        bidirectional: bool,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_features
        for index, channels_out in enumerate(channels):
            blocks.append(
                ResidualTCNBlock(
                    current,
                    channels_out,
                    kernel_size=kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            current = channels_out
        self.tcn = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=current,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_dimension = gru_hidden * (2 if bidirectional else 1)
        if output_dimension % attention_heads != 0:
            raise ValueError("GRU output dimension must be divisible by attention_heads")
        self.self_attention = nn.MultiheadAttention(
            output_dimension,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(output_dimension)
        self.pool = AttentionPool(output_dimension)
        self.output_dimension = output_dimension

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)
        output, _ = self.gru(output)
        attended, _ = self.self_attention(output, output, output, need_weights=False)
        output = self.attention_norm(output + attended)
        return self.pool(output)


class RelationEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_network = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden,
            hidden,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_dimension = hidden * (2 if bidirectional else 1)
        self.pool = AttentionPool(output_dimension)
        self.output_dimension = output_dimension

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.input_network(sequence)
        output, _ = self.gru(output)
        return self.pool(output)


class VANETDetector(nn.Module):
    """Two-branch TCN-BiGRU-attention classifier with attack-specific auxiliaries."""

    def __init__(
        self,
        behavior_features: int,
        relation_features: int,
        num_classes: int = 3,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        gru_hidden: int = 64,
        relation_hidden: int = 32,
        bidirectional: bool = True,
        attention_heads: int = 4,
        fusion_hidden: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        channels = tcn_channels or [64, 64, 96]
        self.behavior_encoder = BehaviorEncoder(
            behavior_features,
            channels,
            kernel_size,
            gru_hidden,
            bidirectional,
            attention_heads,
            dropout,
        )
        self.relation_encoder = RelationEncoder(
            relation_features,
            relation_hidden,
            bidirectional,
            dropout,
        )
        combined = (
            self.behavior_encoder.output_dimension + self.relation_encoder.output_dimension
        )
        self.fusion = nn.Sequential(
            nn.Linear(combined, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden // 2, num_classes)
        self.illusion_auxiliary = nn.Linear(self.behavior_encoder.output_dimension, 1)
        self.sybil_auxiliary = nn.Linear(self.relation_encoder.output_dimension, 1)

    def forward(self, behavior: torch.Tensor, relation: torch.Tensor) -> dict[str, torch.Tensor]:
        behavior_vector, behavior_attention = self.behavior_encoder(behavior)
        relation_vector, relation_attention = self.relation_encoder(relation)
        fused = self.fusion(torch.cat([behavior_vector, relation_vector], dim=-1))
        return {
            "logits": self.classifier(fused),
            "illusion_logit": self.illusion_auxiliary(behavior_vector).squeeze(-1),
            "sybil_logit": self.sybil_auxiliary(relation_vector).squeeze(-1),
            "behavior_attention": behavior_attention,
            "relation_attention": relation_attention,
        }


def build_model(config: dict, behavior_features: int, relation_features: int) -> VANETDetector:
    model_config = config.get("model", {})
    return VANETDetector(
        behavior_features=behavior_features,
        relation_features=relation_features,
        num_classes=3,
        tcn_channels=list(model_config.get("tcn_channels", [64, 64, 96])),
        kernel_size=int(model_config.get("kernel_size", 3)),
        gru_hidden=int(model_config.get("gru_hidden", 64)),
        relation_hidden=int(model_config.get("relation_hidden", 32)),
        bidirectional=bool(model_config.get("bidirectional", True)),
        attention_heads=int(model_config.get("attention_heads", 4)),
        fusion_hidden=int(model_config.get("fusion_hidden", 128)),
        dropout=float(model_config.get("dropout", 0.25)),
    )

