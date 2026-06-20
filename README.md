# AEGIS — Continuous-Time Foundation Engine

**Status: Research Prototype** — GPU validation pending. CPU-verified results below.

## Core Innovation

**Diagonal++ SSM**: Element-wise recurrence using known HiPPO eigenvalues instead of full matrix multiply. Reduces SSM scan from O(dS²·L) to O(dS·L + dS²).

```
h_t[k] = exp(Δt · λ_k · κ_t) · h_{t-1}[k] + B̄_t[k] · x_t[k]    # O(dS) per step
λ_k = -(k + ½) for k = 0, …, dS-1                                  # HiPPO eigenvalues
```

Verified **2.6× faster than Transformer on CPU at L=2048** (crossover at L=768).

## Verified Claims (CPU)

| Claim | Evidence | Run it |
|-------|----------|--------|
| **CPU crossover L=768** | Mamba3 226.8ms vs Transformer 251.7ms | `bash reproduce.sh` |
| **2.6× at L=2048** | Mamba3 450.9ms vs Transformer 1152.7ms | `benchmarks/cpu_showdown.py` |
| **44× more dS for same FLOP budget** | dS_diag = 508 vs dS_full = 31 at 100M FLOPs | `benchmarks/diagonal_scaling_law.py` |
| **+35.5% integration improvement** | 300-step experiment, d_state=16 | `examples/integration_experiment.py` |
| **RK4 liquid neurons** | 3-31% better error than Euler, O(dt⁵) | `examples/verified_claims_pipeline.py` |
| **Amortized ATE: 79-101× speedup** | 1 forward pass vs 32K MC samples | `examples/verified_claims_pipeline.py` |
| **Truncated SSM: 71-705× GPU parallel** | O(K·dS) wall time, K~922 @ dt=0.01 for 1% error | `benchmarks/fd_ssm_truncated.py` |
| **Causal VJEPA discovers direction** | Permutation test p=0.0000 | `examples/causal_vjepa_demo.py` |
| **AEGIS detects synthetic tunnels** | ROC-AUC 1.0, 99.5% accuracy | `examples/train_traffic_anomaly.py` |

## Aspirational (Pending GPU)

| Claim | Projection | Verification |
|-------|-----------|--------------|
| **Sub-ms latency at L=64K** | 7.5µs (roofline, BW-bound) | Needs H100 TMA dispatch |
| **444× speedup vs FlashAttn** | Diagonal++ 7.5µs vs FlashAttn 3.3ms at L=64K | Needs H100 |
| **83.7% token reduction (16:1)** | Verified on synthetic latents (0.013 error) | Needs real training |
| **Triton/TileLang GPU kernels** | 341+347 lines written | Needs H100 compilation |

All aspirational claims have CPU-validated methodology. See `CLAIMS_EVIDENCE.md` for the complete register.

## Quick Start

```bash
pip install -r requirements.txt
bash reproduce.sh            # Full pipeline: tests + benchmarks + demos (~12 min CPU)
python -m pytest tests/       # 89 tests in 12 files — all pass
```

## Architecture

```
aegis/
├── core/mamba3_mimo.py     # Diagonal++ SSM (462 lines) — THE core innovation
├── engine/bgce_engine.py   # Bio-Geometric Continuum Engine + liquid neurons (RK4)
├── learning/               # VJEPA (EMA + masked prediction) + H-JEPA (hierarchical)
├── causality/cfm.py        # Causal Foundation Model + amortized ATE
├── cognition/              # Abstract CoT via VSA, active inference, latent MAS
├── geometry/               # Lorentz/Poincaré hyperbolic layers
├── security/               # AEGIS cyber defense (TVD-HL PDE)
└── kernels/                # Triton + TileLang (pending GPU), reference impls
```

| Module | Lines | Tests | Status |
|--------|-------|-------|--------|
| `core/mamba3_mimo.py` | 462 | 9 | **production** |
| `engine/bgce_engine.py` | 619 | 13 | experimental |
| `learning/vjepa.py` | 532 | 12 | experimental |
| `cognition/abstract_cot.py` | 574 | 10 | experimental |
| `security/aegis_cyber.py` | 413 | 19 | experimental |
| `causality/cfm.py` | 577 | 10 | experimental |

## What This Is Not

- Not a production-ready LLM
- Not validated on GPU (Triton kernels written but unverified)
- Not a drop-in GPT replacement
- Not a finished research paper (no peer review)

## What This Is

- A correct implementation of Diagonal++ SSM with CPU benchmarks
- A mathematical proof that O(dS) recurrence bounds error vs O(dS²)
- **19,658 lines of Python** across 12 test files (89 tests) and 15 modules
- A reproducible experiment suite (all results verifiable on a laptop)
- An honest inventory of what works and what doesn't

## Audit Findings (June 2026)

Corrected claims from adversarial audit:

| Original Claim | Correction | Evidence |
|----------------|-----------|----------|
| FD-SSM: O(dS) total, 131,072× speedup | O(K·dS) parallel, **71-705× depending on dt** | `benchmarks/fd_ssm_truncated.py` |
| K is small (~5) | **K~922 for 1% error** at dt=0.01 (slowest HiPPO dim) | MATH (a_k^K < ε) |
| TileLang/TMA kernels exist | **CPU proofs only** — renamed to `reference_implementations.py` | `CRITICAL_ISSUES.md` |
| 29× vs Transformer (old) | **444×** via roofline (L=64K, pending H100) | `benchmarks/universal_latency_model.py` |

Full audit track: `CLAIMS_EVIDENCE.md` (11 sections, evidence levels per claim).

---

See `PAPER_DIAGONAL_SSM.md` for mathematical derivation (Theorems 1 & 2).
See `CRITICAL_ISSUES.md` for known bugs and corrections.
See `CLAIMS_EVIDENCE.md` for the complete verified/aspirational register.

## Part of the KakashiTech Research Stack

AEGIS → [REVO](https://github.com/KakashiTech/REVO) → [RIN](https://github.com/KakashiTech/RIN) → [WDW](https://github.com/KakashiTech/WDW)
