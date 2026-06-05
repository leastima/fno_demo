"""
prepare.py — FROZEN eval harness for FNO autoresearch on Darcy Flow.

DO NOT MODIFY THIS FILE.
It defines the fixed evaluation protocol that makes all experiments comparable.
The agent must only modify train.py and neuralop/ source files.
"""

import time
import torch
from neuralop.data.datasets import load_darcy_flow_small
from neuralop import LpLoss

# ---------------------------------------------------------------------------
# Fixed constants — changing these breaks comparability between experiments
# ---------------------------------------------------------------------------
N_TRAIN     = 1000    # training samples
N_TEST      = 100     # test samples (resolution 16)
TEST_RES    = 16      # primary eval resolution
BATCH_SIZE  = 64
N_EPOCHS    = 500     # fixed training budget (wall-clock varies; steps are fair)
SEED        = 42      # global seed, applied before model init and data loading


def set_seed(seed: int = SEED):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(device: str = "cuda"):
    """
    Returns train_loader, test_loader (res=16), data_processor.
    Fixed split and resolution — do not expose these to train.py as arguments.
    """
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=N_TRAIN,
        batch_size=BATCH_SIZE,
        n_tests=[N_TEST],
        test_resolutions=[TEST_RES],
        test_batch_sizes=[32],
    )
    data_processor = data_processor.to(device)
    return train_loader, test_loaders[TEST_RES], data_processor


def evaluate(model, test_loader, data_processor, device: str = "cuda") -> dict:
    """
    Run the fixed evaluation protocol.
    Returns {"test_l2re": float, "test_h1": float}.
    This function must not be modified.
    """
    from neuralop import H1Loss

    l2loss = LpLoss(d=2, p=2, reduction="mean")
    h1loss = H1Loss(d=2)

    model.eval()
    total_l2, total_h1, n_samples = 0.0, 0.0, 0
    peak_mem_before = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if torch.cuda.is_available() else 0.0
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = data_processor.preprocess(batch, batched=True)
            x = batch["x"].to(device)
            out = model(x)
            out, batch = data_processor.postprocess(out, batch)
            y = batch["y"].to(device)
            bs = x.shape[0]
            total_l2 += l2loss(out, y).item() * bs
            total_h1 += h1loss(out, y).item() * bs
            n_samples += bs

    peak_mem = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if torch.cuda.is_available() else 0.0
    )
    return {
        "test_l2re":    total_l2 / n_samples,
        "test_h1":      total_h1 / n_samples,
        "peak_vram_mb": peak_mem,
    }
