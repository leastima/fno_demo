"""
FROZEN EVALUATION HARNESS.

Do not modify this file during AutoResearch experiments.

Based on the original FNO paper code (raj-brown/fourier_neural_operator).
Loads data from official .mat files, applies UnitGaussianNormalizer,
appends a 2D coordinate grid, and provides frozen evaluation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    try:
        from ruamel.yaml import YAML
    except ImportError as exc:
        raise ImportError("ruamel.yaml is required. Run `pip install ruamel.yaml`.") from exc
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    yaml = YAML(typ="safe")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return cfg


# ---------------------------------------------------------------------------
# Data loading helpers (from utilities3.py / original FNO repo)
# ---------------------------------------------------------------------------

class _MatReader:
    """Load MATLAB v7.3 (.mat) files using h5py or scipy.io."""
    def __init__(self, file_path: str | Path):
        import scipy.io
        import h5py

        self.file_path = str(file_path)
        try:
            self._data = scipy.io.loadmat(self.file_path)
            self._old_mat = True
        except Exception:
            self._data = h5py.File(self.file_path, "r")
            self._old_mat = False

    def load_file(self, file_path: str | Path) -> None:
        import scipy.io
        import h5py

        self.file_path = str(file_path)
        try:
            self._data = scipy.io.loadmat(self.file_path)
            self._old_mat = True
        except Exception:
            self._data = h5py.File(self.file_path, "r")
            self._old_mat = False

    def read_field(self, field: str) -> torch.Tensor:
        x = self._data[field]
        if not self._old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        return torch.from_numpy(x.astype(np.float32))


class UnitGaussianNormalizer:
    """Per-point mean/std normalizer (frozen after construction from training data)."""
    def __init__(self, x: torch.Tensor, eps: float = 1e-5):
        self.mean = torch.mean(x, dim=0)
        self.std  = torch.std(x,  dim=0)
        self.eps  = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    def to(self, device) -> "UnitGaussianNormalizer":
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self

    def cuda(self) -> "UnitGaussianNormalizer":
        return self.to("cuda")

    def cpu(self) -> "UnitGaussianNormalizer":
        return self.to("cpu")


# ---------------------------------------------------------------------------
# Main data pipeline
# ---------------------------------------------------------------------------

def _build_grid(s: int) -> torch.Tensor:
    """Return a (1, s, s, 2) grid of [0,1]x[0,1] coordinates."""
    lin = np.linspace(0, 1, s)
    gx, gy = np.meshgrid(lin, lin)
    grid = np.stack([gx, gy], axis=-1).reshape(1, s, s, 2)
    return torch.tensor(grid, dtype=torch.float32)


def get_dataloaders(
    cfg: dict[str, Any],
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, UnitGaussianNormalizer]:
    """
    Returns (train_loader, test_loader, y_normalizer).

    Batch format
    ------------
    Both loaders yield (x, y) tuples where:
      x : (B, s, s, 3)  — normalised a(x)  +  2D grid coords
      y : (B, s, s)     — train: normalised u(x);  test: normalised u(x)
                          (call y_normalizer.decode() to get physical values)
    """
    data_root   = Path(cfg.get("data_root", DEFAULT_DATA_ROOT))
    train_file  = data_root / "piececonst_r421_N1024_smooth1.mat"
    test_file   = data_root / "piececonst_r421_N1024_smooth2.mat"

    for f in (train_file, test_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Official Darcy .mat file not found: {f}\n"
                f"Download from: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-"
            )

    ntrain      = int(cfg["train_samples"])
    ntest       = int(cfg["test_samples"])
    batch_size  = int(cfg["batch_size"])

    # Downsample factor: 421 → 85 (r=5)
    r = 5
    s = int(((421 - 1) / r) + 1)   # = 85

    reader = _MatReader(train_file)
    x_train = reader.read_field("coeff")[:ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field("sol")[:ntrain,   ::r, ::r][:, :s, :s]

    reader.load_file(test_file)
    x_test = reader.read_field("coeff")[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field("sol")[:ntest,   ::r, ::r][:, :s, :s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test  = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_enc = y_normalizer.encode(y_train)
    y_test_enc  = y_normalizer.encode(y_test)

    grid = _build_grid(s)
    x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)
    x_test  = torch.cat([x_test.reshape(ntest,  s, s, 1), grid.repeat(ntest,  1, 1, 1)], dim=3)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train_enc),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test_enc),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, y_normalizer


# ---------------------------------------------------------------------------
# Evaluation  (always in physical / decoded space)
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    cfg: dict[str, Any],
    y_normalizer: UnitGaussianNormalizer,
) -> dict[str, float]:
    """
    Evaluate mean relative L2 error on the physical (decoded) solution.
    Both model output and target are decoded before computing the error.
    """
    del cfg
    model.eval()
    y_normalizer.to(device)
    batch_size = test_loader.batch_size

    total_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred  = y_normalizer.decode(model(x))
            truth = y_normalizer.decode(y)
            diff  = torch.flatten(pred - truth, start_dim=1)
            norms = torch.linalg.vector_norm(diff, ord=2, dim=1)
            denom = torch.linalg.vector_norm(torch.flatten(truth, start_dim=1), ord=2, dim=1)
            total_error   += float((norms / (denom + 1e-12)).sum().item())
            total_samples += int(x.shape[0])

    if total_samples == 0:
        raise ValueError("Test loader produced zero samples.")
    return {"test_l2re": total_error / total_samples}


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_final_metrics(metrics: dict[str, Any]) -> None:
    print("FINAL_METRICS_START")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("FINAL_METRICS_END")
