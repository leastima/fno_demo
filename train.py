"""
train.py — training script for FNO paper reproduction on Darcy Flow.

Branch: paper-repro/darcy-flow
Target: reproduce Table 4 in Li et al. 2020 (arXiv:2010.08895)

Hyperparameters from paper Appendix A.3.2:
  - FNO: n_modes=12, width=32, n_layers=4
  - Adam lr=1e-3, weight_decay=1e-4
  - StepLR step_size=100, gamma=0.5
  - N_train=1000, train_res=421
  - Expected: test_l2re ≈ 0.0108 at s=85 (Table 4)

The agent may modify this file freely.
DO NOT modify prepare.py.
"""

import time
import torch
from neuralop.models import FNO
from neuralop import LpLoss
from neuralop.utils import count_model_params

from prepare import get_data, evaluate, set_seed, N_EPOCHS, SEED, TEST_RESOLUTIONS

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
set_seed(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading data (downloads on first run)...")
train_loader, test_loaders, data_processor = get_data(device)
print("Data loaded.")

# ---------------------------------------------------------------------------
# Model — paper config: n_modes=12, width=32, n_layers=4
# ---------------------------------------------------------------------------
model = FNO(
    n_modes=(12, 12),
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    n_layers=4,
)
model = model.to(device)
n_params = count_model_params(model)

# ---------------------------------------------------------------------------
# Optimizer — paper: Adam lr=1e-3, weight_decay=1e-4
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Scheduler — paper: StepLR step_size=100, gamma=0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# ---------------------------------------------------------------------------
# Loss — paper trains with relative L2
# ---------------------------------------------------------------------------
train_loss_fn = LpLoss(d=2, p=2, reduction="mean")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)

t0 = time.perf_counter()

for epoch in range(N_EPOCHS):
    model.train()
    for batch in train_loader:
        batch       = data_processor.preprocess(batch, batched=True)
        x           = batch["x"].to(device)
        out         = model(x)
        out, batch  = data_processor.postprocess(out, batch)
        y           = batch["y"].to(device)
        loss        = train_loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

train_seconds = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Evaluation at all test resolutions
# ---------------------------------------------------------------------------
result = evaluate(model, test_loaders, data_processor, device)

# ---------------------------------------------------------------------------
# Summary — do not change format
# ---------------------------------------------------------------------------
print("---")
print(f"test_l2re:       {result['test_l2re']:.6f}   # primary = s={TEST_RESOLUTIONS[0]}")
for res in TEST_RESOLUTIONS:
    print(f"test_l2re_{res}:  {result[f'test_l2re_{res}']:.6f}")
print(f"train_seconds:   {train_seconds:.1f}")
print(f"peak_vram_mb:    {result['peak_vram_mb']:.1f}")
print(f"num_params:      {n_params}")
print(f"n_modes:         12")
print(f"n_layers:        4")
print(f"hidden_channels: 32")
print(f"train_res:       421")
print("---")
