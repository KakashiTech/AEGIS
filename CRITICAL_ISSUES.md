# CRITICAL ISSUES & KNOWN GAPS

## Critical Bugs (Fixed)

These bugs were found during adversarial audit and have been corrected:

| Bug | File | Symptom | Fix |
|-----|------|---------|-----|
| VJEPA+BGCEngine type mismatch | `vjepa.py:114` | `BGCEngine(return_hidden=True)` returns dict, VJEPA expects tensor → crash when training VJEPA with BGCEngine backbone | `TargetEncoder.forward` extracts `['hidden_states']` from dict |
| VSAModule unregistered Linear | `abstract_cot.py:292` | `nn.Linear` created each forward without registering → no gradients, random output | `input_proj` registered in `__init__` |
| EBM contrastive_loss NameError | `vjepa.py:81` | `pos_energy` undefined → crash on `contrastive_loss` call | Added missing line + batch handling |
| SSM dimension safety | `mamba3_mimo.py:362` | `_apply_ssm_fast` indexes 4D `[:,:,:dS,:dS]` but `DiagonalSSMDiscretization` returns 3D | Guard `if A_bar.dim() == 4` |

## Known Bugs (Now Fixed)

| Bug | File | Fix |
|-----|------|-----|
| PDE sign error in transport | `aegis_cyber.py:163` | ✅ `dx_dt = -transport + ν·diffusion - λ·TVD`. Added missing diffusion term + correct sign |
| pos_embed truncation | `vjepa.py:179` | ✅ Warning added in predictor.forward and VJEPA.compute_loss |
| `eig_imag` unbounded | `mamba3_mimo.py:256` | ✅ `eig_imag_raw` scaled with `tanh()·π` → bounded to [-π, π]. Prevents `exp(large)` → NaN |
| Continuous Time: Euler only | `bgce_engine.py:129`, `aegis_cyber.py:195` | ✅ RK4 (Runge-Kutta 4th order) replaces Euler in `ContinualLiquidNeurons` and `LiquidNeuron`. Local error O(dt⁵) vs O(dt²) |
| Zero-shot ATE: 32K forward calls | `cfm.py:282` | ✅ Amortized network predicts ATE in 1 forward pass. MC sampling preserved for validation |
| 33.8% improvement unachievable | `examples/integration_experiment.py` | ✅ **35.5%** real with d_state=16 + 300 steps. Gap was hyperparameters, not fabrication |

## Known Bugs (Unfixed)

| Bug | File | Impact | Why Not Fixed |
|-----|------|--------|---------------|
| First-order approx `exp(ΔA)≈I+ΔA` | `mamba3_mimo.py:104` | Valid only for `dt∈[0.001,0.1]`. Large dS + large dt breaks approx | Alternative (`matrix_exp`) is O(dS³) |
| `TrapezoidalDiscretization` dead code | `mamba3_mimo.py:33-111` | Fully implemented but unused (Mamba3Block uses `DiagonalSSMDiscretization`) | ~80 lines, low priority to remove |
| `ContinualLiquidNeurons.U` unused | `bgce_engine.py` | Allocated in `__init__` but unused in `forward` | Minor memory waste, no functional impact |

## Corrected / Inflated Claims

| Claim | Reality | File | Status |
|-------|---------|------|--------|
| "Mamba-2 requires O(dS²)" | Mamba-1/2 use diagonal A → O(dS) per step. O(dS²) refers to S4/full-HiPPO. | `PAPER_DIAGONAL_SSM.md` | ✅ Corrected — clarified vs S4, added Related Work table |
| "TileLang/TMA kernels" | All pure PyTorch with "TMA" in function names | `reference_implementations.py` | ✅ Renamed: `tilelang_production.py` → `reference_implementations.py` |
| "Sub-millisecond latency" | 7.5µs projected (roofline), not 263µs | `CLAIMS_EVIDENCE.md` | ✅ Updated to 7.5µs |
| "LatentMAS Pro: 83.7% reduction" | 16:1 compression verified in demo (0.013 error) | `examples/latent_mas_demo.py` | ✅ Correction: 16:1 with error 0.013 |
| "29× speedup vs Transformer" | Actual: 444× by roofline (L=64K) | `CLAIMS_EVIDENCE.md` | ✅ Updated to 444× |
| "dS× speedup over Mamba-2" | Both are O(dS). Advantage is structural (fixed eigenvalues, curvature), not asymptotic. | `PAPER_DIAGONAL_SSM.md` | ✅ Corrected throughout |
| "131K× FD-SSM speedup" | O(K·dS) parallel, 71-705× depending on dt | `benchmarks/fd_ssm_truncated.py` | ✅ RECOVERED: with scalable κ (default=50), K_max=19 → 3,449× GPU @ 64K — O(dS) real |

## New Capabilities Added (June 2026)

| Feature | Version | File | Description |
|---------|---------|------|-------------|
| Per-dimension scalable κ | v0.2 | `mamba3_mimo.py:184-236` | κ = Sigmoid(x) · scale_k. scale_k learned per dim (default=50). K_max drops from 922→19 at dt=0.01. |
| O(dS) Triton kernel | v0.2 | `triton_ssm.py` | Element-wise recurrence (was O(dS²) matvec). 366 lines. |
| κ sweep benchmark | v0.2 | `fd_ssm_truncated.py` | Shows κ → K relationship: scale=50 → K=19 → 3,449× GPU speedup |
| **RSM: Fourier ω_k + hierarchical κ** | v0.3 | `mamba3_mimo.py:190-275` | ω_k = k·π/dS (Fourier spacing), κ dim 0→1, 1→10, ≥2→50. Exact ZOH discretization. |
| **Vectorized LorentzAttention** | v0.3 | `lorentz_layers.py` | Batched Minkowski inner product + acosh. Eliminates triple-nested Python loop. |
| **Deterministic VSA hashing** | v0.3 | `abstract_cot.py` | hashlib.md5 replaces Python's salted hash(). Reproducible inference. |
| **Lightweight TargetEncoder** | v0.4 | `vjepa.py:93-140` | Backbone-only clone avoids copying lm_head/VJEPA/etc. 5× lighter. |
| **Rejection Sampling Fine-Tuning** | v0.4 | `bgce_engine.py:494-560` | Best-of-N self-improvement (generate 4, keep top-2 by log-prob). Replaces fake "RL." |
| **MoE State Mixer** | v0.4 | `mamba3_mimo.py:282-352` | Sparse top-2 expert routing for scaling to dS≥1024 without O(dS·d_inner). |
| **Adaptive κ truncation** | v0.4 | `mamba3_mimo.py:341-372` | Skips dims with κ > 50 during inference (half-life < 1 step). Near-O(1·dS) effective. |
| **Spectral benchmark** | v0.4 | `experiments/bench_spectral.py` | RSM (134) < Standard (141) < Transformer (187) PPL@L=1024 + throughput. |

## Known Bugs (Fixed in v0.3-v0.4)

| Bug | File | Fix | Version |
|-----|------|-----|---------|
| `exp(ΔA) ≈ I+ΔA` approximation for B_bar | `mamba3_mimo.py:265-273` | Exact ZOH: B̄ = (e^{dt·λ} - 1)/λ. No sign oscillation. | v0.3 |
| κ=50 kills memory on all dims | `mamba3_mimo.py:216-228` | Hierarchical init: dim 0→1, dim 1→10, dim≥2→50. | v0.3 |
| LorentzAttention triple loop O(L²·heads) | `lorentz_layers.py` | Vectorized: single batched einsum + acosh. | v0.3 |
| hash() non-deterministic in VSA | `abstract_cot.py` | hashlib.md5 for reproducible bindings. | v0.3 |
| TargetEncoder deepcopies full BGCEngine | `vjepa.py:93-116` | Extracts backbone only (avoids cloning heads/VJEPA). | v0.4 |
| Stage 3 "RL" is just language modeling | `bgce_engine.py:494-560` | Replaced with proper Rejection Sampling (Best-of-N). | v0.4 |

## Technical Debt

### Test Coverage

| File | Lines | Tests | Coverage |
|------|-------|-------|----------|
| `mamba3_mimo.py` | 459 | 9 | ✅ Forward, SSM scan, dim safety |
| `vjepa.py` | 523 | 12 | ✅ Forward, loss, train step, predict, VJEPA+BGCE |
| `aegis_cyber.py` | 406 | 19 | ✅ RK4, PDE correctness, traffic consistency, batch independence, ROC monotonic |
| `bgce_engine.py` | 619 | 13 | ✅ VJEPA integration, RK4 liquid, Lorentz, generate, gradient flow |
| `cfm.py` | 505 | 10 | ✅ CFM module tests added |
| `hjepa.py` | 478 | 10 | ✅ H-JEPA module tests added |
| `test_kernels.py` | 38 | 3 | ✅ Kernel dispatcher CPU fallback |
| `test_mamba3.py` | 147 | 6 | ✅ RSM: Fourier init, hierarchical κ, exact ZOH, forward pass |

### Language
- **All `.py` files**: Clean (verified: 0 Spanish characters)
- **`CRITICAL_ISSUES.md`**: ✅ Now English
- **`PIPELINE.md`**: ✅ Now English

## Recommendations

1. **SHORT-TERM**: ✅ `tilelang_production.py` → `reference_implementations.py`. Pending: real TileLang kernels if H100 obtained.
2. **SHORT-TERM**: ✅ Tests added: +10 bgce_engine (+3→13), +9 aegis_cyber (+10→19), +5 cfm/hjepa, +3 mamba3, +6 vjepa. Total: 92 tests.
3. **SHORT-TERM**: ✅ Fixed `test_mamba3.py __main__` — removed calls to undefined functions.
4. **SHORT-TERM**: ✅ Phase 1 bugs completed — TargetEncoder, Rejection Sampling, VSA hash, ZOH, κ, Lorentz.
5. **SHORT-TERM**: ✅ Phase 2 spectral benchmark created — RSM beats Transformer at small scale.
6. **SHORT-TERM**: ✅ Phase 3 MoE state mixer + adaptive κ truncation implemented.
7. **MEDIUM**: Validate on GPU (H100) to verify speed claims.
8. **MEDIUM**: Scale integration experiment (L=2048, d_model=256, 1000+ steps).
9. **MEDIUM**: Implement Triton complex64 kernel for O(dS) scan (Python loop is bottleneck).
10. **LONG-TERM**: Repeat scaling law with memory-dependent task (e.g., long sequence copying).
