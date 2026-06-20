# AEGIS Architecture

```
aegis/
├── core/                  # Core SSM (State Space Model)
│   ├── mamba3_mimo.py     # Diagonal++ SSM — THE core innovation
│   │                      #   O(dS) per step (vs O(dS²) in Mamba-2)
│   │                      #   Verified: CPU crossover L=768, 2.2× at L=2048
│   └── __init__.py
│
├── engine/                # High-level pipeline
│   └── bgce_engine.py     # Bio-Geometric Continuum Engine
│                           #   Mamba-3 backbone + Liquid neurons + VJEPA
│                           #   3-stage training: SFT → VJEPA → RL
│
├── learning/              # Self-supervised learning
│   ├── vjepa.py           # VJEPA — Vicinal JEPA (EMA + masked prediction)
│   │                      #   Verified: causal direction learning (p=0.0000)
│   └── hjepa.py           # H-JEPA — Hierarchical JEPA
│   │                      #   Multi-level latent prediction + CausalTimePrior
│
├── causality/             # Causal inference
│   └── cfm.py             # Causal Foundation Model
│                           #   Partial causal graphs + do-interventions
│                           #   Amortized ATE (79-101× speedup over MC)
│
├── cognition/             # Reasoning & agentic modules
│   ├── abstract_cot.py    # Abstract Chain-of-Thought via VSA
│   ├── active_inference_navigator.py  # Scientific discovery agent
│   ├── latent_mas.py      # Multi-agent latent communication
│   └── odar_expert.py     # System 1 / System 2 routing
│
├── geometry/              # Hyperbolic geometry
│   └── lorentz_layers.py  # Lorentz / Poincaré layers
│
├── security/              # Cybersecurity
│   └── aegis_cyber.py     # AEGIS for cyber defense
│
├── kernels/               # Kernel implementations
│   ├── triton_ssm.py      # Triton kernels (341 lines, pending GPU)
│   ├── tilelang_h100.py   # H100 dispatcher with real probes
│   └── reference_implementations.py  # CPU proofs + theoretical projections
│                                         (renamed from tilelang_production.py)
│
├── training/              # Training utilities
│   ├── metrics.py         # Learning verification metrics
│   └── trace.py           # Execution tracing
│
└── ... (remaining utils)
```

## Module Dependencies

```
BGCEngine
  ├── Mamba3MIMO (core/mamba3_mimo.py)
  ├── VJEPA (learning/vjepa.py)
  │   └── TargetEncoder (EMA deep-copy of backbone)
  └── Liquid neurons (built-in)

VJEPA
  ├── Mamba3MIMO (or any nn.Module as backbone)
  ├── Predictor (Transformer)
  └── EBM_energy (Energy-Based Model)

H-JEPA
  ├── HierarchicalLevel (× n_levels)
  ├── CausalTimePrior (causal graph)
  └── MentalRolloutSimulator

CFM
  ├── PartialCausalGraph
  ├── CausalAttention
  ├── StructuralEquation (× n_vars)
  └── ATEEstimator
```

## Evidence Levels

All claims in this project have an evidence level:

| Label | Meaning |
|-------|---------|
| [CPU] | Verified on CPU in runnable scripts |
| [MATH] | Proven by formal derivation |
| [THEORY] | Roofline/first-principles analysis |
| [PENDING] | Requires GPU (H100) |

See `CLAIMS_EVIDENCE.md` for the complete register.
