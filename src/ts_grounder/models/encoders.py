from __future__ import annotations

import torch
from torch import nn


# 轻量 1D 卷积块，不使用 BatchNorm，避免小 batch / CPU 下的额外不稳定因素。
class ConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TextEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ConvBlock(hidden_dim, kernel_size=5, dropout=dropout) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return self.output_proj(x)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(image))
