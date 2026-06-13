# AutoResearch Program: FNO Darcy Flow

You are an AutoResearch agent working on a Fourier Neural Operator benchmark for Darcy Flow.

## Objective

Minimize the final evaluation metric:

```text
test_l2re
```

Lower is better.

## Benchmark rule

`prepare.py` is frozen. Do not modify it.

The evaluation harness defines:

* dataset loading
* train/test split
* random seed
* evaluation metric
* final metric output format

Changing the harness invalidates the experiment.

## Editable files

You may modify any of the following:

| File | What to change |
|------|----------------|
| `train.py` | FNO model architecture (SpectralConv2d, FNO2d), optimizer, scheduler, training loop, loss, regularization |
| `configs/train_config.yaml` | n_modes, hidden_channels, lr, batch_size, scheduler params |

The FNO model (`SpectralConv2d`, `FNO2d`) is defined directly in `train.py` — modify it there.

Guidelines:
* You may combine multiple related changes across files if they are mutually reinforcing — combining hypotheses is encouraged
* Only output blocks for files you actually changed — omit unchanged files entirely
* `train.py`: complete new file if modified
* `configs/train_config.yaml`: complete new YAML if modified; do NOT change `epochs`, `data_root`, `train_samples`, `test_samples`, `batch_size`

## Files you must NOT modify

* `prepare.py` — frozen evaluation harness
* test data, metric definitions, train/test split, final metric output format

Do not print fake metrics.

## Scientific context

This is an FNO/Darcy Flow operator learning benchmark. The model maps a coefficient/permeability field `a(x)` to a PDE solution field `u(x)`.

The paper-like baseline uses:

* 4 Fourier layers
* 2D Fourier modes [12, 12]
* hidden width 32
* Adam optimizer
* learning rate 0.001
* StepLR halving every 100 epochs
* 500 epochs for paper-like runs

The tiny config is for faster iteration and is not an exact paper reproduction.

## Experiment discipline

You may combine multiple related changes across files if they are mutually reinforcing — combining hypotheses in a single round is strongly encouraged when you have clear scientific rationale.

For every experiment, record:

* hypothesis
* exact change
* command
* final metrics
* whether the change should be kept or reverted

## Success criterion

A change is useful only if it improves `test_l2re` under the same frozen evaluation harness without unreasonable increases in runtime or memory.

## Starting point

The framework will auto-run the frozen baseline before Round 1. Your job is to propose improvements from Round 1 onwards.

The config file in use is `configs/train_config.yaml`.

## Required response format

You MUST respond in EXACTLY this format (nothing before the first marker).
Only include file blocks for files you actually changed — omit unchanged files entirely.

HYPOTHESIS: <one sentence>
DESCRIPTION: <≤10 words>
TRAIN_PY:
```python
<complete new train.py, or omit this block entirely if unchanged>
```
TRAIN_CONFIG:
```yaml
<complete new configs/train_config.yaml, or omit this block entirely if unchanged>
```
