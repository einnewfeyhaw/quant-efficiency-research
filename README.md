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

## Results (ResNet-18, CIFAR-100, A100)

| Method | Accuracy | Acc Drop | Size | Size Reduction |
|---|---|---|---|---|
| FP32 (baseline) | 56.36% | — | 44.9 MB | 1× |
| Naive PTQ INT8 | 56.67% | −0.31% | 11.2 MB | 4× |
| GPTQ-lite INT8 | 56.06% | +0.30% | 11.2 MB | 4× |
| AWQ-lite INT8 | 56.17% | +0.19% | 11.2 MB | 4× |
| Naive PTQ INT4 | 7.95% | +48.4% | 5.6 MB | 8× |
| GPTQ-lite INT4 | 3.34% | +53.0% | 5.6 MB | 8× |
| AWQ-lite INT4 | 3.00% | +53.4% | 5.6 MB | 8× |

**INT8:** All three methods are nearly lossless (<0.3% accuracy drop) at 4× compression — exactly the expected result for a well-trained model.

**INT4:** Catastrophic degradation across all methods. This is expected with per-tensor quantization: the quantization grid is too coarse to capture weight distributions at 4 bits. Real GPTQ and AWQ use per-column/per-group quantization with outlier suppression — the gap between our simplified implementations and the full methods is itself an illustration of why that extra sophistication matters.

**Kurtosis–sensitivity correlation: r = 0.691, p = 0.0005.** Weight kurtosis is a strong, statistically significant predictor of per-layer quantization sensitivity. This is the empirical backbone for the coreset hypothesis below.

**Most sensitive layers (INT4):** `layer1.0.conv1` (15.6% drop), `conv1` (7.1%), `layer4.1.conv2` (6.4%) — early and late layers dominate, consistent with prior work on mixed-precision quantization.

---

## Analysis

**Layer-wise sensitivity** — quantize one layer at a time (INT4) and measure accuracy drop, identifying candidates for mixed-precision.

**Weight distribution analysis** — per-layer histograms showing how PTQ vs AWQ reshape weight distributions.

**Kurtosis–sensitivity correlation** — empirical test of the hypothesis: does high kurtosis predict high quantization sensitivity? r = 0.691, p = 0.0005.

**Accuracy–efficiency frontier** — Pareto plots of accuracy vs. size reduction and speedup for all methods.

---

## Setup

```bash
pip install torch torchvision matplotlib seaborn tabulate scipy
```

Or run directly in Google Colab (A100 recommended; fine-tuning takes ~4 min).

---

## Connection to Coreset Work

| Efficiency axis | Method | Result |
|---|---|---|
| Data-side | Noise-Free Gradient (TMLR 2025) | 30% of dataset, <5% accuracy drop |
| Model-side | AWQ-lite INT8 (this repo) | 4× size reduction, 0.19% accuracy drop |
| Combined | *open experiment* | Does coreset training lower weight kurtosis → better INT4 tolerance? |

**Key finding:** Weight kurtosis strongly predicts quantization sensitivity (r = 0.691, p = 0.0005). If gradient-diverse coreset training produces more uniform weight distributions — as our hypothesis predicts — coreset-trained models should tolerate lower bit-widths with less accuracy degradation. This is the next experiment.

The coreset paper's gradient computation uses `torch.func.vmap` for per-sample Hessian-vector products — the same Hessian structure that GPTQ exploits for layer-wise optimal quantization. Both methods are minimizing a second-order approximation of the loss; the connection is structural, not coincidental.
