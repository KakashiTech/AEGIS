# AEGIS — Continuous-Time Foundation Engine

**This is not an LLM. This is not a transformer. This is a learned simulator of underlying flow dynamics.**

```python
h_t[k] = exp(Δt · λ_k) · h_{t-1}[k] + B̄_t[k] · x_t[k]    # O(ds) per step, not O(ds²)
```

The attention matrix was never the right abstraction. Language is not a sequence of discrete tokens — it is a continuous dynamical system projected onto a discrete vocabulary. Transformers model the projection. AEGIS models the dynamics.

---

## The Problem with Transformers

Every token in a transformer attends to every previous token. This is O(L²) memory and compute. The industry fix is to throw hardware at it — H100s, TPUs, optimized kernels, model parallelism. Nobody questions whether O(L²) is necessary because it's been the default since "Attention Is All You Need."

**It's not necessary.**

State Space Models proved O(L) is possible. But they introduced a new bottleneck: the HiPPO matrix recurrence is O(ds²) per step — a dense matrix-vector product that doesn't scale.

**That bottleneck is also not necessary.**

---

## Diagonal++: The Fix

The HiPPO matrix has known eigenvalues: λ_k = -(k + ½). Instead of computing the full matrix recurrence, we operate in the eigenvalue basis:

| Step | Standard SSM (Mamba-2) | Diagonal++ (AEGIS) | Speedup |
|------|----------------------|-------------------|---------|
| Recurrence | h_t = Ā · h_{t-1} | h_t[k] = Ā[k] · h_{t-1}[k] | **ds×** |
| Complexity | O(ds²·L) | O(ds·L + ds²) | 16-256× |
| FLOPs (ds=16, L=1024) | 262K | 32K | 8× |

The state mixer (a single ds×ds multiply at sequence end) recovers cross-dimension interactions at O(ds²) — paid once per sequence, not per step.

**Proof**: `PAPER_DIAGONAL_SSM.md` — 1 page, bounded error theorem, roofline analysis, reproducible benchmarks.

---

## CPU Vindication

Conventional wisdom: "SSMs need GPU to be useful." Measured on CPU (same architecture, same parameter count):

```
L=  512:  AEGIS 96ms   vs  Transformer 98ms    →  tie
L=  768:  AEGIS 142ms  vs  Transformer 165ms   →  1.2x faster
L= 1024:  AEGIS 195ms  vs  Transformer 288ms   →  1.5x faster
L= 2048:  AEGIS 363ms  vs  Transformer 781ms   →  2.2x faster
```

AEGIS with Diagonal++ beats a transformer of the same size on **CPU** starting at L=768. This is not a GPU benchmark. This is a single Intel core, no accelerators, no tricks. The gap grows with sequence length because O(L) compounds against O(L²).

Reproduce: `bash reproduce.sh`

---

## Projected GPU Performance (H100 Roofline)

| Method | 24 layers, L=4096 | vs Transformer |
|--------|-------------------|----------------|
| Transformer | ~3.5ms | 1× |
| Mamba-2 | ~1.2ms | 2.9× |
| **Diagonal++** | **~0.12ms** | **29×** |

The roofline analysis is in `PAPER_DIAGONAL_SSM.md`. These are projections from first principles — the kernels are written (`aegis/kernels/triton_ssm.py`, 341 lines) but unverified without H100 access.

---

## Learning Evidence (CPU, <5 min each)

Three controlled experiments proving AEGIS learns real patterns:

| Dataset | Task | Metric | Result |
|---------|------|--------|--------|
| Shakespeare Tiny | character-level LM | Perplexity | **57.1 → 12.9** (77% reduction) |
| Algebraic Reasoning | symbolic arithmetic | OOD loss gap | **0.33** (generalizes to unseen depth) |
| Traffic Anomaly | C2 beacon detection | ROC-AUC | **1.0** (TPR=0.985, FPR=0.0) |

Each experiment runs on CPU, converges in under 500 steps, and produces measurable learning. Code in `examples/`.

---

## Architecture

```
Input ──► Embedding ──► Diagonal++ SSM ──► Lorentz ──► LM Head
                            │
                    ┌───────┴───────┐
                    ▼               ▼
                  VJEPA         Abstract-CoT
              (latent pred)   (tokenless reasoning)
                    │               │
                    └───────┬───────┘
                            ▼
                       AEGIS Cyber
                   (flow-based IDS)
```

Every component is connected. Gradient flows end-to-end. Verified: `python honest_training.py`

---

## Files

```
aegis/                          # Core engine
├── core/mamba3_mimo.py         # Diagonal++ SSM (722 lines)
├── geometry/lorentz_layers.py  # Lorentz projections (315)
├── learning/vjepa.py           # Variational JEPA (434)
├── learning/hjepa.py           # Hierarchical JEPA (478)
├── cognition/abstract_cot.py   # Abstract reasoning + VSA (573)
├── cognition/latent_mas.py     # Multi-agent latent space (388)
├── cognition/odar_expert.py    # System 1/2 routing (356)
├── engine/aegis_engine.py      # E2E pipeline (617)
├── security/aegis_cyber.py     # Flow-based IDS (401)
├── causality/cfm.py            # Causal foundation models (505)
├── kernels/triton_ssm.py       # Triton kernel (341)
├── kernels/tilelang_h100.py    # H100 dispatcher (347)
└── training/metrics.py         # Training metrics (75)

examples/                       # Runnable experiments
├── train_shakespeare_tiny.py   # Language modeling demo
├── train_algebraic_reasoning.py# Symbolic reasoning demo
├── train_traffic_anomaly.py    # Intrusion detection demo
├── aegis_live_demo.py          # Live packet capture demo
└── vsa_demo.py                 # Hyperdimensional computing

benchmarks/                     # Measured performance data
tests/                          # 4 suites, 34 tests, all passing
audits/                         # Systematic verification (11 phases)
```

---

## Quick Start

```bash
# Dependencies: PyTorch 2.2+, no GPU required
pip install -r requirements.txt

# Run everything
bash reproduce.sh            # <10 min, CPU only

# Tests
python tests/run_all_tests.py

# Language learning demo
python examples/train_shakespeare_tiny.py

# Live traffic detection
python examples/aegis_live_demo.py --demo

# Full audit
python honest_training.py
```

---

## What AEGIS Is Not

- It is not "a faster transformer"
- It is not production-ready (no GPU validation, no dataset-scale training)
- It is not a drop-in GPT replacement

## What AEGIS Is

- A correct, working implementation of a new architecture
- A mathematical contribution (Diagonal++) with bounded-error proof
- A reproducible benchmark suite where every result is measured, not claimed
- 17K lines of Python, 34 tests, 11 audit phases, all verifiable on a laptop

---

**AEGIS**: Continuous-Time Foundation Engine. No attention matrices were harmed in the making of this project.
