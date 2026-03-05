# 🔥 PINN — Inverse Heat Source Identification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaximeAuger/pinn-inverse-heat/blob/main/notebooks/01_inverse_1D_colab.ipynb)
[![Colab V8](https://img.shields.io/badge/Colab%20V8-orange?logo=googlecolab)](https://colab.research.google.com/github/MaximeAuger/pinn-inverse-heat/blob/main/notebooks/01_inverse_1D_colab_v8.ipynb)
[![Colab V9](https://img.shields.io/badge/Colab%20V9-purple?logo=googlecolab)](https://colab.research.google.com/github/MaximeAuger/pinn-inverse-heat/blob/main/notebooks/01_inverse_1D_colab_v9.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> Recovering an **unknown heat source** f(x) from sparse noisy temperature measurements using Physics-Informed Neural Networks.

---

## Problem Statement

Given the steady-state heat equation:

$$-\frac{d^2 T}{dx^2} = f(x), \quad x \in (0,1), \quad T(0) = T(1) = 0$$

**Goal**: given only N noisy measurements, **identify f(x)** without ever knowing it.

---

## Method

Two neural networks trained simultaneously:

| Network | Role |
|---------|------|
| N_T(x) | Approximate temperature field T(x) |
| N_f(x) | Identify the unknown source f(x) |

### Loss function

$$\mathcal{L} = w_{pde}\|N_T'' + N_f\|^2 + w_{bc}\,\mathcal{L}_{bc} + w_{data}\|N_T(x^{obs}) - T^{obs}\|^2 + w_{reg}\left\|\frac{dN_f}{dx}\right\|^2$$

### Training: Adam (5000 ep.) → L-BFGS (300 steps)

---

## Results

| Metric | Value |
|--------|-------|
| Temperature L² error | ~1e-4 |
| Source L² error | ~5e-3 |
| Observations | 15 pts, 2% noise |

---

## Project Structure

```
pinn-inverse-heat/
├── README.md
├── notebooks/
│   └── 01_inverse_1D.ipynb
├── src/
│   ├── model.py       ← PINN + SourceNetwork
│   ├── losses.py      ← PDE residual, BCs, data, Tikhonov
│   └── train.py       ← Adam + L-BFGS schedule
├── results/
└── environment.yml
```

---

## Quickstart

```bash
git clone https://github.com/MaximeAuger/pinn-inverse-heat
cd pinn-inverse-heat
conda env create -f environment.yml
conda activate pinn-heat
jupyter notebook notebooks/01_inverse_1D.ipynb
```

---

## Key Features

- Dual-network architecture (temperature + source)
- Tikhonov regularization (order 0 and 1)
- Adam → L-BFGS training schedule
- Comparison with analytical solution
- Noise sensitivity analysis (0% to 10%)
- Training animation (GIF)

---

## References

- Raissi et al., *Physics-informed neural networks* (JCP, 2019)
- Karniadakis et al., *Physics-informed machine learning* (Nature Reviews Physics, 2021)
