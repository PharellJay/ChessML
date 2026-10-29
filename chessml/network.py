"""Policy/value network: a small residual CNN sized for CPU training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_MOVE_PLANES, NUM_PLANES, planes_to_float


@dataclass
class NetConfig:
    blocks: int = 4
    channels: int = 48


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    """Input (B, NUM_PLANES, 8, 8) -> (policy logits (B, 4672), value (B,) in [-1, 1])."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()
        )
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(self.cfg.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, NUM_MOVE_PLANES, 1),
        )
        self.value_conv = nn.Sequential(nn.Conv2d(c, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(8 * 64, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.body(self.stem(x))
        policy = self.policy_head(h).flatten(1)  # index = plane * 64 + square
        value = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(1)
        return policy, value


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Evaluator:
    """Wraps a network for batched inference on uint8 planes."""

    def __init__(self, model: ChessNet):
        self.model = model.eval()

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(planes_to_float(planes))
        logits, value = self.model(x)
        return logits.numpy(), value.numpy()


def save_checkpoint(path: str, model: ChessNet, optimizer=None, **meta) -> None:
    """Atomically write a checkpoint (model weights, optional optimizer, metadata)."""
    data = {"model": model.state_dict(), "net": asdict(model.cfg), **meta}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(data, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def model_from_checkpoint(path_or_data) -> ChessNet:
    data = load_checkpoint(path_or_data) if isinstance(path_or_data, (str, os.PathLike)) else path_or_data
    model = ChessNet(NetConfig(**data["net"]))
    model.load_state_dict(data["model"])
    return model.eval()

