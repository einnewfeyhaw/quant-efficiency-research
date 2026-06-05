# Efficient Training: Coreset Selection × Model Quantization

Exploring two complementary axes of neural network efficiency — **data-side compression** via gradient-based coreset selection and **model-side compression** via post-training quantization — and their interaction.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/einnewfeyhaw/quant-efficiency-research/blob/main/quantization_analysis.ipynb)

---

## Motivation

Our prior work ([Noise-Free Gradient, TMLR 2025](https://github.com/ai23resch04001/Noise_free_gradient)) reduces training cost by selecting a **gradient-diverse coreset** — 30% of the dataset that achieves >95% of full-data performance. This raises a natural follow-on question:

> Does training on a gradient-similar coreset affect how easily the resulting model can be quantized?

**Hypothesis:** Gradient-diverse training avoids redundant optimization steps, producing more uniform weight distributions. Lower weight kurtosis → fewer outliers → quantization grids fit better → less accuracy degradation at low bit-widths.

This notebook builds the measurement framework and provides experimental groundwork to test this hypothesis.

---

## What's Implemented (from scratch)

| Method | Core Idea | Key Insight |
|---|---|---|
| **Naive PTQ** | Round weights to nearest grid point | Baseline; ignores weight distributions |
| **GPTQ-lite** | Weight rounding minimizes *output* error, not weight error | Uses diagonal Hessian H ≈ E[x²] to allocate bits by activation magnitude |
| **AWQ-lite** | Scale salient weight channels before quantizing | Protects weights multiplied by large activations; s = (mean\|x\|)^α |

All three are implemented in pure PyTorch — no quantization libraries — so the math is transparent.

---

## Analysis

**Layer-wise sensitivity** — quantize one layer at a time (INT4) and measure accuracy drop, identifying candidates for mixed-precision.

**Weight distribution analysis** — per-layer histograms showing how PTQ vs AWQ reshape weight distributions.

**Kurtosis–sensitivity correlation** — empirical test of the hypothesis: does high kurtosis predict high quantization sensitivity? (Spoiler: yes.)

**Accuracy–efficiency frontier** — Pareto plots of accuracy vs. size reduction and speedup for all methods.

---

## Setup

```bash
pip install torch torchvision matplotlib seaborn tabulate scipy
```

Or run directly in Google Colab (T4 GPU recommended for layer sensitivity sweep).

---

## Connection to Coreset Work

| Efficiency axis | Method | Reduction |
|---|---|---|
| Data-side | Noise-Free Gradient (TMLR 2025) | 30% of dataset, <5% accuracy drop |
| Model-side | AWQ-lite INT4 (this repo) | 8× size reduction, ~X% accuracy drop |
| Combined | *open experiment* | ? |

The coreset paper's gradient computation uses `torch.func.vmap` for per-sample Hessian-vector products — the same Hessian structure that GPTQ exploits for layer-wise optimal quantization. This is not a coincidence: both methods are minimizing a second-order approximation of the loss.
