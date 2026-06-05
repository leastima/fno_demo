"""
train.py — MUTABLE training script for FNO autoresearch on Darcy Flow.

The agent may freely modify this file and neuralop/ source files.
The agent must NOT modify prepare.py.

Goal: minimize test_l2re (relative L2 error on Darcy Flow, resolution 16x16).
"""

import time
import torch
from neuralop.models import FNO
from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params

from prepare import get_data, evaluate, set_seed, N_EPOCHS, SEED

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
set_seed(SEED)

# ---------------------------------------------------------------------------
# Data  (do not change the call signature — prepare.py controls the split)
# ---------------------------------------------------------------------------
train_loader, test_loader, data_processor = get_data(device)

# ---------------------------------------------------------------------------
# Model  (agent: feel free to change architecture, n_modes, hidden_channels, etc.)
# ---------------------------------------------------------------------------
model = FNO(
    n_modes=(16, 16),
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    n_layers=4,
)
model = model.to(device)
n_params = count_model_params(model)

# ---------------------------------------------------------------------------
# Optimizer & scheduler  (agent: feel free to change)
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# ---------------------------------------------------------------------------
# Loss  (agent: feel free to change — but eval uses LpLoss from prepare.py)
# ---------------------------------------------------------------------------
train_loss_fn = H1Loss(d=2)

# ---------------------------------------------------------------------------
# Training loop  (agent: feel free to restructure)
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
# Evaluation  (do not change — uses the frozen evaluate() from prepare.py)
# ---------------------------------------------------------------------------
result = evaluate(model, test_loader, data_processor, device)

# ---------------------------------------------------------------------------
# Summary  (do not change the format — program.md greps these lines)
# ---------------------------------------------------------------------------
print("---")
print(f"test_l2re:       {result['test_l2re']:.6f}")
print(f"test_h1:         {result['test_h1']:.6f}")
print(f"train_seconds:   {train_seconds:.1f}")
print(f"peak_vram_mb:    {result['peak_vram_mb']:.1f}")
print(f"num_params:      {n_params}")
print(f"n_modes:         {model.fno_blocks.convs[0].weight.shape[-2]}")
print(f"n_layers:        {len(model.fno_blocks.convs)}")
print(f"hidden_channels: {model.fno_blocks.convs[0].weight.shape[1]}")
print("---")
