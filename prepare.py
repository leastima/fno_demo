"""
prepare.py — FROZEN eval harness for FNO paper reproduction on Darcy Flow.

Branch: paper-repro/darcy-flow
Target: reproduce Table 4 in Li et al. 2020 (arXiv:2010.08895)

DO NOT MODIFY THIS FILE.
Data is downloaded from Zenodo record 12784353 on first run (~few hundred MB).
"""

import time
import torch
from pathlib import Path
from neuralop.data.datasets import DarcyDataset
from neuralop import LpLoss, H1Loss

# ---------------------------------------------------------------------------
# Fixed constants — match paper Section 5.2 / Appendix A.3.2
# ---------------------------------------------------------------------------
DATA_ROOT       = Path("~/.cache/neuralop/darcy_paper").expanduser()
N_TRAIN         = 1000
N_TEST          = 200
TRAIN_RES       = 421        # full resolution (paper trains at 421x421)
TEST_RESOLUTIONS = [85, 211, 421]   # paper Table 4: s=85,141,211,421 (skip 141 for speed)
BATCH_SIZE      = 20         # paper uses small batch due to 421x421 size
TEST_BATCH_SIZE = 20
N_EPOCHS        = 500        # fixed training budget
SEED            = 42


def set_seed(seed: int = SEED):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(device: str = "cuda"):
    """
    Downloads (on first run) and returns loaders for the full Darcy Flow dataset.
    Returns train_loader, test_loaders dict {res: loader}, data_processor.
    Fixed split — do not expose parameters to train.py.
    """
    dataset = DarcyDataset(
        root_dir=DATA_ROOT,
        n_train=N_TRAIN,
        n_tests=[N_TEST] * len(TEST_RESOLUTIONS),
        batch_size=BATCH_SIZE,
        test_batch_sizes=[TEST_BATCH_SIZE] * len(TEST_RESOLUTIONS),
        train_resolution=TRAIN_RES,
        test_resolutions=TEST_RESOLUTIONS,
        encode_input=False,
        encode_output=True,
        download=True,
    )
    data_processor = dataset.data_processor.to(device)
    return dataset.train_loader, dataset.test_loaders, data_processor


def evaluate(model, test_loaders: dict, data_processor, device: str = "cuda") -> dict:
    """
    Run fixed evaluation at all test resolutions.
    Returns dict with keys like "test_l2re_85", "test_l2re_421", and "test_l2re" (primary = lowest res).
    This function must not be modified.
    """
    l2loss = LpLoss(d=2, p=2, reduction="mean")
    h1loss = H1Loss(d=2)

    model.eval()
    results = {}
    peak_mem = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for res, loader in test_loaders.items():
            total_l2, total_h1, n_samples = 0.0, 0.0, 0
            for batch in loader:
                batch       = data_processor.preprocess(batch, batched=True)
                x           = batch["x"].to(device)
                out         = model(x)
                out, batch  = data_processor.postprocess(out, batch)
                y           = batch["y"].to(device)
                bs = x.shape[0]
                total_l2 += l2loss(out, y).item() * bs
                total_h1 += h1loss(out, y).item() * bs
                n_samples += bs
            results[f"test_l2re_{res}"] = total_l2 / n_samples
            results[f"test_h1_{res}"]   = total_h1 / n_samples

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    # primary metric = smallest test resolution (85x85), matches paper Table 4 first column
    results["test_l2re"]    = results[f"test_l2re_{TEST_RESOLUTIONS[0]}"]
    results["peak_vram_mb"] = peak_mem
    return results
