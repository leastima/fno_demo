"""
Mutable AutoResearch training script.

AutoResearch agents may modify this file to improve the frozen benchmark metric.
Do not modify prepare.py during benchmark experiments.

Architecture: original FNO-2D (raj-brown/fourier_neural_operator / Li et al. 2020)
  - SpectralConv2d with torch.fft.rfft2 (PyTorch >= 1.8 compatible)
  - 4 Fourier layers, width=32, modes=12, input has 3 channels (a(x) + 2D grid)
  - UnitGaussianNormalizer applied to both x and y (via prepare.py)
  - y_normalizer.decode() used for loss and evaluation
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare


# ---------------------------------------------------------------------------
# Model (modifiable)
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1,  :self.modes2] = \
            self._mul(x_ft[:, :, :self.modes1,  :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._mul(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    """4-layer FNO with linear bypass and BatchNorm."""
    def __init__(self, modes1: int, modes2: int, width: int, in_channels: int = 3):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)

        self.conv = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(4)])
        self.w    = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.bn   = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(4)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        x = self.fc0(x)              # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)   # (B, width, H, W)

        for conv, w, bn in zip(self.conv, self.w, self.bn):
            x1 = conv(x)
            x2 = w(x.view(B, x.shape[1], -1)).view(B, x.shape[1], H, W)
            x  = bn(x1 + x2)
            if conv is not self.conv[-1]:
                x = F.relu(x)

        x = x.permute(0, 2, 3, 1)   # (B, H, W, width)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)         # (B, H, W)


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    return FNO2d(
        modes1=int(cfg.get("n_modes", [12, 12])[0]),
        modes2=int(cfg.get("n_modes", [12, 12])[1]),
        width=int(cfg.get("hidden_channels", 32)),
        in_channels=3,  # a(x) + 2D grid
    )


def build_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "Adam")).lower()
    lr   = float(cfg.get("lr", 1e-3))
    wd   = float(cfg.get("weight_decay", 1e-4))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {cfg.get('optimizer')!r}")


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    name = str(cfg.get("scheduler", "StepLR")).lower()
    if name in ("none", "null"):
        return None
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 100)),
            gamma=float(cfg.get("gamma", 0.5)),
        )
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.get("epochs", 500)),
        )
    raise ValueError(f"Unsupported scheduler: {cfg.get('scheduler')!r}")


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.flatten(pred - target, start_dim=1)
    denom = torch.flatten(target, start_dim=1)
    return (torch.linalg.vector_norm(diff, ord=2, dim=1) /
            (torch.linalg.vector_norm(denom, ord=2, dim=1) + 1e-12)).mean()


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    prepare.set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, y_normalizer = prepare.get_dataloaders(cfg)
    y_normalizer.to(device)

    model = build_model(cfg).to(device)
    n_params = prepare.count_parameters(model)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    epochs       = int(cfg.get("epochs", 500))
    log_interval = int(cfg.get("log_interval", max(1, epochs // 10)))
    batch_size   = int(cfg["batch_size"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples  = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            out       = model(x)
            out_phys  = y_normalizer.decode(out)
            y_phys    = y_normalizer.decode(y)
            loss = relative_l2_loss(out_phys.view(out_phys.shape[0], -1),
                                    y_phys.view(y_phys.shape[0], -1))
            loss.backward()

            if cfg.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
            optimizer.step()

            train_loss += float(loss.item()) * x.shape[0]
            n_samples  += x.shape[0]

        if scheduler is not None:
            scheduler.step()

        if ep == 1 or ep == epochs or ep % log_interval == 0:
            print(f"epoch {ep:04d}/{epochs}  train_l2re={train_loss / n_samples:.6f}")

    train_seconds = time.perf_counter() - t_start
    metrics = prepare.evaluate(model, test_loader, device, cfg, y_normalizer)
    metrics["train_seconds"] = f"{train_seconds:.2f}"
    metrics["peak_vram_mb"]  = (
        f"{torch.cuda.max_memory_allocated() / 1024**2:.1f}"
        if torch.cuda.is_available() else "NA"
    )
    metrics["num_params"] = n_params
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        cfg     = prepare.load_config(args.config)
        metrics = train(cfg)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    prepare.print_final_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
