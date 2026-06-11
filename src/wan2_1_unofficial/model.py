from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    task: str = "video"
    vocab_size: int = 32768
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    image_channels: int = 3
    audio_channels: int = 1
    max_positions: int = 4096
    moe_experts: int = 4


class MultiHeadLatentBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, experts: int) -> None:
        super().__init__()
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)) for _ in range(experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(x), dim=-1)
        stacked = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (stacked * weights.unsqueeze(-1)).sum(dim=-2)


class UnofficialModel(nn.Module):
    """Compact PyTorch interface for the paper-specific reproduction repo."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.hidden_dim
        self.token_embed = nn.Embedding(self.config.vocab_size, d)
        self.position_embed = nn.Embedding(self.config.max_positions, d)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.config.image_channels, d // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d // 2, d, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.video_encoder = nn.Sequential(
            nn.Conv3d(self.config.image_channels, d // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(d // 2, d, 3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
        )
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(self.config.audio_channels, d // 2, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(d // 2, d, 7, stride=2, padding=3),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([MultiHeadLatentBlock(d, self.config.num_heads) for _ in range(self.config.num_layers)])
        self.expert = ExpertMLP(d, self.config.moe_experts)
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, self.config.vocab_size)
        self.regression_head = nn.Linear(d, d)

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        positions = positions.clamp(max=self.config.max_positions - 1)
        return self.token_embed(token_ids) + self.position_embed(positions)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.image_encoder(image).flatten(2).transpose(1, 2)
        return feat

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        feat = self.video_encoder(video).flatten(2).transpose(1, 2)
        return feat

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        feat = self.audio_encoder(audio).transpose(1, 2)
        return feat

    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        streams = []
        if token_ids is not None:
            streams.append(self.encode_tokens(token_ids))
        if image is not None:
            streams.append(self.encode_image(image))
        if video is not None:
            streams.append(self.encode_video(video))
        if audio is not None:
            streams.append(self.encode_audio(audio))
        if not streams:
            token_ids = torch.zeros(1, 16, dtype=torch.long, device=self.token_embed.weight.device)
            streams.append(self.encode_tokens(token_ids))
        x = torch.cat(streams, dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x + self.expert(x))
        pooled = x.mean(dim=1)
        return {
            "sequence": x,
            "pooled": pooled,
            "logits": self.lm_head(x),
            "embedding": self.regression_head(pooled),
        }


def reconstruction_loss(output: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, torch.zeros_like(output))
