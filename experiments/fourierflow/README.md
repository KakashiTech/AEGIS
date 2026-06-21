# FourierFlow — Compiling Attention to SSMs

## Problem

Trained Transformer attention heads compute expensive pairwise dot-products across sequence length. State Space Models (SSMs) like Mamba achieve linear-time inference by replacing this with a recurrent formulation. **Can a trained attention head be *compiled* into an equivalent Diagonal++ SSM with minimal fidelity loss?**

## Approach

A causal attention head computes:

$$y_t = \sum_{s \leq t} \text{softmax}(q_t \cdot k_s) \, v_s$$

The attention matrix $A_{t,s} = \text{softmax}(q_t \cdot k_s)$ is lower-triangular. A Diagonal++ SSM computes:

$$h_t = \lambda \odot h_{t-1} + B x_t, \quad y_t = C h_t + D x_t$$

which expands to the matrix form:

$$A^{\text{SSM}}_{t,s} = C_t \cdot \text{diag}(\lambda^{t-s}) \cdot B_s$$

We approximate $A$ by a sum of $d$ rank-1 terms modulated by per-dimension exponential decay $\lambda_k^{t-s}$. This is equivalent to finding $\lambda_k, B_k, C_k$ minimizing $\|A - \hat{A}\|_F$.

The key hypothesis: **attention softmax patterns can be captured by a low-dimensional SSM with input-dependent decay, because softmax attention already has a smooth, structured (often near-low-rank) pattern.**

## Current Status

- [x] Synthetic demo: approximate random attention matrices with learnable SSM parameters
- [x] MSE and cosine similarity metrics
- [ ] Real trained Transformer weights (e.g., Pythia, GPT2)
- [ ] End-to-end distillation (replace attention head with SSM and finetune)
- [ ] Theoretical analysis of approximation error bounds

## Key Questions

1. **Expressivity**: How many SSM dimensions $d$ are needed to match a $d_{\text{head}}$ attention head? Is $d \ll d_{\text{head}}$ sufficient?
2. **Input dependence**: The SSM $\lambda$ here is input-independent. Can we learn an input-dependent $\lambda(x_t)$ (like Mamba's selective scan) to improve fidelity?
3. **Training vs Post-hoc**: Is it better to distill (train the SSM to match attention outputs) or post-hoc project (factorize the attention matrix directly)?
4. **Scaling**: Does approximation quality degrade for longer sequences or deeper layers?
5. **Theoretical**: What is the VC-dimension / Rademacher complexity of the Diagonal++ class relative to softmax attention?

## Usage

```bash
python attention_ssm_mapping.py
```
