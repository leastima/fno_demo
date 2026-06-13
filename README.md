# FNO Darcy Flow — AutoResearch Benchmark

A self-contained Fourier Neural Operator (FNO) benchmark for 2-D Darcy Flow,
adapted from [raj-brown/fourier_neural_operator](https://github.com/raj-brown/fourier_neural_operator)
(Li et al., 2020 — "Fourier Neural Operator for Parametric Partial Differential Equations").

Used as the target task in the `sciml_auto_research` automated research framework.

---

## Task

Learn the solution operator of the 2-D Darcy Flow equation:

```
−∇ · (a(x)∇u(x)) = f(x),   x ∈ (0,1)²
u(x) = 0,                    x ∈ ∂(0,1)²
```

- Input `a(x)`: permeability field sampled from a log-normal push-forward of a Gaussian random field  
  (`ψ#N(0,(−Δ+9I)⁻²)`, values 12 on positive part, 3 on negative)
- Output `u(x)`: PDE solution via 2nd-order finite differences on a 421×421 grid
- Evaluation resolution: 85×85 (downsampled by factor 5 from 421)

---

## Paper Baseline (Table 4, FNO paper)

| Parameter | Value |
|-----------|-------|
| Fourier modes | 12 × 12 |
| Hidden width | 32 |
| Layers | 4 |
| Optimizer | Adam |
| Learning rate | 1e-3, halved every 100 epochs |
| Epochs | 500 |
| Train samples | 1000 |
| Test samples | 100 |
| Reported test L2 rel. error | ~0.0108 |

For AutoResearch runs, `epochs=50` is used for faster iteration.

---

## Dataset

**Official FNO paper dataset** (required):

| File | Size | Content |
|------|------|---------|
| `piececonst_r421_N1024_smooth1.mat` | ~1.6 GB | 1024 training samples, 421×421 |
| `piececonst_r421_N1024_smooth2.mat` | ~1.6 GB | 1024 test samples, 421×421 |

### Download

```bash
pip install gdown

# Training data
gdown --id 1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV -O piececonst_r421_N1024_smooth1.mat

# Test data
gdown --id 1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV -O piececonst_r421_N1024_smooth2.mat
```

Alternatively, download `Darcy_421.zip` from the
[FNO Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)
and extract both `.mat` files.

Set `data_root` in `configs/train_config.yaml` to the directory containing both `.mat` files.

---

## File Structure

```
fno_demo/
├── prepare.py               # Frozen evaluation harness (do not modify)
├── train.py                 # Mutable training script (AutoResearch modifies this)
├── configs/
│   ├── train_config.yaml    # Tunable hyperparameters
│   └── autoresearch_421.yaml
└── program.md               # AutoResearch agent instructions
```

### `prepare.py` (frozen)

- Loads `.mat` files via `MatReader`
- Applies `UnitGaussianNormalizer` to both `a(x)` (input) and `u(x)` (output)
- Appends 2-D grid coordinates to input (3 channels total)
- Exposes `get_dataloaders()` and `evaluate()` with `y_normalizer` for decoding predictions back to physical space

### `train.py` (mutable)

- `SpectralConv2d` + `FNO2d` defined inline (agents may modify architecture)
- Uses `torch.fft.rfft2` / `irfft2` (PyTorch ≥ 1.8)
- Decodes both prediction and target via `y_normalizer.decode()` before computing loss
- Prints `FINAL_METRICS_START / FINAL_METRICS_END` block for framework parsing

---

## Quick Start

```bash
conda activate cfd-gpu

# Edit data_root in configs/train_config.yaml first
python train.py --config configs/train_config.yaml
```

## AutoResearch

```bash
cd ../   # sciml_auto_research root
bash run_claude.sh   # Claude (Anthropic)
bash run_codex.sh    # Codex / GPT-4.1 (OpenAI)
```

Results are saved to `results/`, logs to `logs/`, plots to `images/`.
