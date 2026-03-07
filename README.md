# 🔥 VPINN — Inverse Heat Source Identification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaximeAuger/pinn-inverse-heat/blob/main/notebooks/vpinn_inverse_article_benchmark.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-pink)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> Recovering an **unknown heat source** \(f(x)\) from sparse noisy temperature measurements using **Variational Physics-Informed Neural Networks (VPINNs)**.

This project implements a **robust inverse PDE solver** combining weak-form physics constraints, spectral source representation, and multi-stage optimization.

---

# Problem Statement

We consider the **1D steady-state heat equation**

$$
-\frac{d^2 T}{dx^2} = f(x), \quad x \in (0,1)
$$

with boundary conditions

$$
T(0) = T(1) = 0
$$

Only sparse noisy measurements of temperature are available:

$$
T^{obs}_i = T(x_i) + \epsilon_i
$$

where $$\( \epsilon_i \)$$ represents measurement noise.

### Objective

Recover the **unknown source function**

$$
f(x)
$$

using only the observations $$\(T^{obs}\)$$.

---

# Method

The solver combines **three stages** for stability.

## Phase 1 — Temperature reconstruction

A neural network approximates the temperature field

$$
T_\theta(x)
$$

with **hard boundary conditions**

$$
T_\theta(x) = x(1-x) N_\theta(x)
$$

Loss:

$$
\mathcal{L}_1 =
w_{data}\|T_\theta(x_i) - T^{obs}_i\|^2
+
w_{H1}\|T'\|^2
+
w_{H2}\|T''\|^2
$$

This produces a **smooth temperature field**.

---

## Phase 2 — Weak-form source identification

The source is represented in a **spectral sine basis**

$$
f(x) = \sum_{k=1}^{K} a_k \sin(k\pi x)
$$

Using the weak formulation:

$$
\int_0^1 T'(x)\phi_k'(x) dx =
\int_0^1 f(x)\phi_k(x) dx
$$

which yields a linear system for the coefficients $$\(a_k\)$$.

A **ridge regularization** stabilizes the inversion.

---

## Phase 3 — VPINN refinement

Temperature and source are jointly optimized using the **weak residual**

$$
\int_0^1 T'(x)\phi_k'(x) dx -
\int_0^1 f(x)\phi_k(x) dx
$$

Total loss:

$$
\mathcal{L} =
w_{data}L_{data}
+
w_{weak}L_{weak}
+
w_{anchor}L_{anchor}
+
w_{reg}L_{reg}
+
w_{H2}L_{H2}
$$

Optimization schedule

```

Adam → L-BFGS

```

---

# Architecture

### Temperature Network

```

Fourier Features
→ MLP (64,64,64)
→ Hard boundary conditions

```

### Source Network

```

Spectral expansion
f(x) = Σ a_k sin(kπx)

```

This representation provides **strong regularization** for the inverse problem.

---

# Results

Typical reconstruction accuracy:

| Metric | Value |
|------|------|
| Temperature L² error | ~3e-4 |
| Source L² error | ~1e-2 |
| Observations | 25 points |
| Noise level | 1% |

Temperature reconstruction is nearly indistinguishable from the ground truth.

---

# Key Insight — Spectral Complexity Control

The most important hyperparameter is

```

k_source

```

If too large:

```

→ source overfits noise
→ oscillatory reconstruction

```

Optimal range:

```

k_source ≈ 3

```

matching the true spectral content of the source.

---

# Experiments

The notebook performs several large-scale studies.

## Noise robustness

Noise levels tested:

```

0%
0.5%
1%
2%
5%

```

Findings:

- temperature reconstruction remains extremely stable
- source accuracy degrades smoothly with noise
- inversion remains reliable up to ~5% noise

---

## k_source sweep

Tested values:

```

k_source ∈ {2,3,4,5,6}

```

This demonstrates the **bias–variance tradeoff**:

| k_source | Behavior |
|---------|---------|
| small | stable but biased |
| optimal (~3) | best reconstruction |
| large | noise overfitting |

---

## Multi-seed evaluation

Experiments repeated over

```

5 random seeds

```

We report:

- mean error
- standard deviation
- median
- min / max

Results are exported as **CSV tables**.

---

# Generated Figures

The notebook automatically produces publication-quality plots:

- temperature reconstruction
- source reconstruction
- training loss curves
- weak residual spectrum
- spectral coefficients
- noise robustness curves
- k_source sensitivity plots
- error heatmaps
- representative case reconstructions

All outputs are saved in

```

results_vpinn_inverse_1d_article/

```

---

# Project Structure

```

pinn-inverse-heat/
├── README.md
├── notebooks/
│   └── vpinn_inverse_article_benchmark.ipynb
├── results_vpinn_inverse_1d_article/
│   ├── figures/
│   ├── tables/
│   └── raw/
└── environment.yml

````

---

# Quickstart

Clone the repository

```bash
git clone https://github.com/MaximeAuger/pinn-inverse-heat
cd pinn-inverse-heat
````

Install environment

```bash
conda env create -f environment.yml
conda activate pinn-heat
```

Run notebook

```bash
jupyter notebook notebooks/vpinn_inverse_article_benchmark.ipynb
```

---

# Key Features

* Variational PINN formulation
* Weak-form PDE enforcement
* Spectral source representation
* Hard boundary conditions
* Adam → L-BFGS training schedule
* Noise robustness experiments
* Multi-seed statistical evaluation
* Publication-quality plots

---

# References

* Raissi et al., *Physics-informed neural networks*, JCP (2019)
* Karniadakis et al., *Physics-informed machine learning*, Nature Reviews Physics (2021)
* Kharazmi et al., *VPINNs: Variational Physics-Informed Neural Networks* (2019)
