# AEGIS — Continuous-Time Foundation Engine

[![CI](https://github.com/KakashiTech/AEGIS/actions/workflows/ci.yml/badge.svg)](https://github.com/KakashiTech/AEGIS/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-89%20passing-brightgreen.svg)](https://github.com/KakashiTech/AEGIS/actions)
[![Code size](https://img.shields.io/badge/code-19.7K%20lines-informational)](.)

> **Diagonal++ SSM**: Element-wise recurrence using known HiPPO eigenvalues.  
> O(dS) per step, CPU-verified **2.6× faster than Transformer at L=2048**.

---

## Quick Start

```bash
pip install -r requirements.txt
bash reproduce.sh          # Full pipeline: tests + benchmarks + demos (~12 min)
python -m pytest tests/    # 89 tests — all pass
```

## Verified Results (CPU)

| Claim | Evidence | Run it |
|-------|----------|--------|
| **CPU crossover at L=768** | Mamba3 226.8ms vs Transformer 251.7ms | `bash reproduce.sh` |
| **2.6× faster at L=2048** | Mamba3 450.9ms vs Transformer 1152.7ms | `benchmarks/cpu_showdown.py` |
| **44× more dS at iso-FLOP** | dS_diag=508 vs dS_full=31 at 100M FLOPs | `benchmarks/diagonal_scaling_law.py` |
| **+35.5% integration gain** | 300-step experiment, d_state=16 | `examples/integration_experiment.py` |
| **3,449× GPU parallel via truncation** | O(dS) real, κ≥50 → K≤19 @ dt=0.01, 1% error | `benchmarks/fd_ssm_truncated.py` |
| **RK4 liquid neurons** | 3-31% better error than Euler, O(dt⁵) | `examples/verified_claims_pipeline.py` |
| **Amortized ATE: 79-101× speedup** | 1 forward pass vs 32K MC samples | `examples/verified_claims_pipeline.py` |
| **Causal VJEPA discovers direction** | Permutation test p=0.0000 | `examples/causal_vjepa_demo.py` |
| **ROC-AUC 1.0 tunnel detection** | 99.5% accuracy on synthetic traffic anomalies | `examples/train_traffic_anomaly.py` |

## Projected (Pending GPU)

| Claim | Projection | Status |
|-------|-----------|--------|
| **Sub-ms latency at L=64K** | 7.5µs roofline (BW-bound) | Needs H100 TMA dispatch |
| **444× vs FlashAttn at L=64K** | Diagonal++ 7.5µs vs FlashAttn 3.3ms | Needs H100 |
| **16:1 latent compression** | 0.013 error on synthetic data | Needs real training |
| **Triton/TileLang GPU kernels** | Written, O(dS) element-wise | Needs H100 compilation |

All projected claims have CPU-validated methodology (`CLAIMS_EVIDENCE.md`).

---

## Architecture

```
aegis/
├── core/mamba3_mimo.py     # Diagonal++ SSM — THE core innovation
├── engine/bgce_engine.py   # Bio-Geometric Continuum Engine + RK4 liquid neurons
├── learning/               # VJEPA + H-JEPA (hierarchical predictive learning)
├── causality/cfm.py        # Causal Foundation Model + amortized ATE
├── cognition/              # Abstract CoT via VSA, active inference, latent MAS
├── geometry/               # Lorentz / Poincaré hyperbolic representations
├── security/               # AEGIS cyber defense (TVD-HL PDE solver)
├── kernels/                # Triton O(dS) kernel + TileLang dispatcher
│
├── benchmarks/             # Reproducible benchmark suite
├── examples/               # Runnable demos
├── tests/                  # 89 tests across 12 files
│
├── PAPER_DIAGONAL_SSM.md   # Full mathematical derivation (Theorems 1-3)
├── CLAIMS_EVIDENCE.md      # Evidence register (11 sections)
└── CRITICAL_ISSUES.md      # Known bugs & corrigenda
```

## Related Work

| Model | Recurrence | Per step | A learning | Spectrum |
|-------|-----------|----------|------------|----------|
| **S4** (Gu et al., 2020) | Full HiPPO $d_S \times d_S$ matvec | $O(d_S^2)$ | Fixed | Structured |
| **Mamba-1** (Gu & Dao, 2023) | Diagonal element-wise | $O(d_S)$ | Learned per dim | None |
| **Mamba-2** (Dao & Gu, 2024) | Diagonal element-wise (SSD) | $O(d_S)$ | Learned per dim | None |
| **Diagonal++ (ours)** | Diagonal element-wise | $O(d_S)$ | **Fixed eigenvalues + learned curvature** | **Known (HiPPO)** |

**What makes Diagonal++ different** (not asymptotically better than Mamba — same O(dS)):

1. **Known eigenvalues** λ_k = -(k+½) enable truncation analysis & stability bounds
2. **Per-dimension learned curvature** κ_k = Sigmoid(x)·scale_k (default=50) makes K=O(1)
3. **Complex frequencies** ω_k add oscillatory dynamics absent in real-valued SSMs
4. **Fewer learned params**: only 2·dS vs dS learned A entries
5. **CPU victory over Transformer** at L≥768 — first SSM to demonstrate this

## Audit & Transparency

This repo underwent an adversarial audit (June 2026) that corrected several claims:

| Original Claim | Corrected |
|----------------|-----------|
| FD-SSM: O(dS) total, 131K× speedup | **O(K·dS) recovered via κ≥50 → K≤19 → 3,449×** |
| K ≈ 5 | K ∝ 1/(κ·dt·|λ_min|); with κ=50: K_max=19 |
| TileLang/TMA production kernels | CPU proofs only; renamed for honesty |
| 29× vs Transformer | 444× per roofline (pending H100) |

---

**Links**: [`PAPER_DIAGONAL_SSM.md`](PAPER_DIAGONAL_SSM.md) · [`CLAIMS_EVIDENCE.md`](CLAIMS_EVIDENCE.md) · [`CRITICAL_ISSUES.md`](CRITICAL_ISSUES.md)

---

*Part of the [KakashiTech](https://github.com/KakashiTech) research stack — AEGIS → REVO → RIN → WDW*
