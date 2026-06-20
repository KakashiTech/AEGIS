#!/usr/bin/env python3
"""
Causal VJEPA — aprendiendo estructura causal via masking estructurado.

Experiment: generate data with known causality X → Y, Z independent.
Entrenamos VJEPA con masking causal (ocultar Y, predecir desde X).
Measure whether model learns correct causal direction.

Hypothesis: causal prediction error (Y|X) < anti-causal error (X|Y).
Esto demuestra que VJEPA captura la flecha del tiempo causal.
"""
import sys, json, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.learning.vjepa import VJEPA, VJEPAConfig, MaskingStrategy
from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig


# ─── Synthetic causal data ────────────────────────────────────────────────
def generate_causal_data(n_samples=500, seq_len=64, d_model=64, seed=42):
    """
    Generate data with known causal structure:
    X (cause) → Y (effect), Z (independent)
    
    X: sinusoidal pattern with phase noise
    Y = 0.7 * X_shifted + 0.3 * X_reverse + Gaussian noise
    Z: independent random walk
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.sin(torch.linspace(0, 8*np.pi, seq_len)).unsqueeze(0).expand(n_samples, -1)
    X = X + torch.randn(n_samples, seq_len) * 0.1
    
    # Y is causally determined by X (with delay and reverse components)
    X_shifted = torch.roll(X, shifts=2, dims=1)
    X_reverse = torch.flip(X, dims=[1])
    Y = 0.7 * X_shifted + 0.3 * X_reverse + torch.randn(n_samples, seq_len) * 0.1
    
    # Z is independent
    Z = torch.cumsum(torch.randn(n_samples, seq_len) * 0.05, dim=1)
    
    # Stack into (n_samples, seq_len, 3) — 3 features
    data = torch.stack([X, Y, Z], dim=-1)
    
    # Expand to d_model by repetition
    n_repeat = d_model // 3
    remainder = d_model % 3
    data = data.repeat(1, 1, n_repeat)
    if remainder > 0:
        data = torch.cat([data, data[:, :, :remainder]], dim=-1)
    
    # Causal graph: Y depends on X (index 0 causes index 1)
    causal_graph = {
        'causes': {1: [0]},    # Y (idx 1) is caused by X (idx 0)
        'effects': {0: [1]},   # X (idx 0) causes Y (idx 1)
        'independent': {2: []}, # Z (idx 2) is independent
    }
    
    return data, causal_graph


class CausalMaskingStrategy(MaskingStrategy):
    """
    Enmascaramiento basado en grafo causal conocido.
    
    - causal_mask: oculta EFFECT, deja CAUSES visibles.
      Model learns to predict effect from its causes.
    - anti_causal_mask: oculta CAUSE, deja EFFECT visible.
      Model should NOT be able to predict cause from effect.
    """
    
    @staticmethod
    def causal_graph_mask(data, causal_graph, feature_dim=3, mask_ratio=0.5):
        """
        Create mask following causal graph.
        
        Para cada feature en [0, feature_dim):
          - Si es un EFFECT (tiene causas): se enmascaran sus dimensiones
          - Si es una CAUSE (causa algo): se dejan visibles
          - Si es INDEPENDIENTE: se enmascaran aleatoriamente
        """
        B, L, D = data.shape
        d_per_feature = D // feature_dim
        mask = torch.zeros(B, L, dtype=torch.bool)
        
        for feat_idx in range(feature_dim):
            start_dim = feat_idx * d_per_feature
            end_dim = (feat_idx + 1) * d_per_feature if feat_idx < feature_dim - 1 else D
            
            # Check if this feature is an effect (has causes)
            has_causes = feat_idx in causal_graph.get('causes', {}) and len(causal_graph['causes'][feat_idx]) > 0
            is_independent = feat_idx in causal_graph.get('independent', {})
            
            if has_causes:
                # EFFECT → ENMASCARAR (ocultar, debe predecirse desde causas)
                n_masked = max(1, int(L * mask_ratio))
                for b in range(B):
                    idx = torch.randperm(L)[:n_masked]
                    mask[b, idx] = True
            elif is_independent:
                # INDEPENDIENTE → enmascarar aleatoriamente
                n_masked = max(1, int(L * mask_ratio))
                for b in range(B):
                    idx = torch.randperm(L)[:n_masked]
                    mask[b, idx] = True
            # CAUSE → siempre visible
        
        return mask


# ─── Model setup ──────────────────────────────────────────────────────────
device = "cpu"
d_model = 48
seq_len = 64
n_samples = 500
feature_dim = 3

print("Generating causal data (X→Y, Z independent)...")
data, causal_graph = generate_causal_data(n_samples, seq_len, d_model)
print(f"  Data shape: {data.shape}")
print(f"  Causal graph: Y depends on X, Z independent")

# Create backbone
ssm_config = SSMConfig(
    d_model=d_model, d_state=8, d_inner=96, dt_rank=4, n_layers=2,
    use_diagonal_ssm=True, device=device,
)
backbone = Mamba3MIMO(ssm_config).to(device)

# VJEPA config with causal masking
vjepa_config = VJEPAConfig(
    d_model=d_model,
    d_pred=d_model // 2,
    predictor_depth=2,
    mask_ratio=0.5,
    mask_strategy="causal",
    loss_type="l1",
)
vjepa = VJEPA(backbone, vjepa_config).to(device)
optimizer = torch.optim.AdamW(
    list(backbone.parameters()) + list(vjepa.predictor.parameters()),
    lr=3e-4, weight_decay=0.01
)

causal_masker = CausalMaskingStrategy()

# ─── Training ──────────────────────────────────────────────────────────────
print(f"\nTraining Causal VJEPA ({n_samples} samples, 100 steps)...")
for step in range(100):
    idx = torch.randperm(n_samples)[:16]
    batch = data[idx]
    
    # Get hidden states from backbone
    with torch.no_grad():
        z_full = backbone(batch, return_hidden=True)
    
    # Create causal mask
    B, L, D = z_full.shape
    target_mask = causal_masker.causal_graph_mask(
        batch, causal_graph, feature_dim, vjepa_config.mask_ratio
    )
    context_mask = ~target_mask
    
    # Build context (causes visible)
    context_list, target_list = [], []
    for i in range(B):
        context_list.append(z_full[i][context_mask[i]])
        target_list.append(z_full[i][target_mask[i]])
    
    max_ctx = max(len(c) for c in context_list)
    max_tgt = min(max(len(t) for t in target_list), vjepa_config.target_length)
    
    ctx_pad = torch.zeros(B, max_ctx, D, device=device)
    tgt_pad = torch.zeros(B, max_tgt, D, device=device)
    tgt_idx = torch.zeros(B, max_tgt, dtype=torch.long, device=device)
    
    for i in range(B):
        ctx_pad[i, :len(context_list[i])] = context_list[i]
        n = min(len(target_list[i]), max_tgt)
        tgt_pad[i, :n] = target_list[i][:n]
        tgt_idx[i, :n] = torch.where(target_mask[i])[0][:n]
    
    # Predict
    z_pred, variance = vjepa.predictor(ctx_pad, tgt_idx, return_variance=True)
    
    # Loss (only on valid targets)
    valid = (tgt_pad.abs().sum(dim=-1) > 0).float()
    diff = z_pred - tgt_pad
    loss = (diff.abs() * valid.unsqueeze(-1)).sum() / (valid.sum() * D + 1e-8)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(vjepa.predictor.parameters()), 1.0)
    optimizer.step()
    vjepa.update_target_encoder()
    
    if (step + 1) % 25 == 0:
        print(f"  Step {step+1:>3d}: loss={loss.item():.4f}")

# ─── Evaluation: Causal vs Anti-causal ────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluation: Causal vs Anti-causal Error")
print("=" * 60)

with torch.no_grad():
    z_full = backbone(data, return_hidden=True)
    B, L, D = z_full.shape
    
    # Causal mask
    causal_mask = causal_masker.causal_graph_mask(data, causal_graph, feature_dim, 0.5)
    
    # Build batch for predictor
    def prepare_batch(mask, z):
        ctx_list, tgt_list = [], []
        for i in range(B):
            ctx_list.append(z[i][~mask[i]])
            tgt_list.append(z[i][mask[i]])
        max_ctx = max(len(c) for c in ctx_list)
        max_tgt = min(max(len(t) for t in tgt_list), 32)
        ctx_pad = torch.zeros(B, max_ctx, D)
        tgt_pad = torch.zeros(B, max_tgt, D)
        tgt_idx = torch.zeros(B, max_tgt, dtype=torch.long)
        for i in range(B):
            ctx_pad[i, :len(ctx_list[i])] = ctx_list[i]
            n = min(len(tgt_list[i]), max_tgt)
            tgt_pad[i, :n] = tgt_list[i][:n]
            tgt_idx[i, :n] = torch.where(mask[i])[0][:n]
        return ctx_pad, tgt_pad, tgt_idx

    # Causal prediction: Y|X
    ctx_c, tgt_c, idx_c = prepare_batch(causal_mask, z_full)
    z_pred_c, _ = vjepa.predictor(ctx_c, idx_c)
    valid_c = (tgt_c.abs().sum(dim=-1) > 0).float()
    diff_c = (z_pred_c - tgt_c).abs() * valid_c.unsqueeze(-1)
    causal_error = diff_c.sum() / (valid_c.sum() * D + 1e-8)
    # Per-sample errors
    causal_per_sample = diff_c.sum(dim=(1, 2)) / (valid_c.sum(dim=1) * D + 1e-8)
    
    # Anti-causal prediction: X|Y (invert mask)
    anti_mask = ~causal_mask
    ctx_a, tgt_a, idx_a = prepare_batch(anti_mask, z_full)
    z_pred_a, _ = vjepa.predictor(ctx_a, idx_a)
    valid_a = (tgt_a.abs().sum(dim=-1) > 0).float()
    diff_a = (z_pred_a - tgt_a).abs() * valid_a.unsqueeze(-1)
    anti_causal_error = diff_a.sum() / (valid_a.sum() * D + 1e-8)
    anti_per_sample = diff_a.sum(dim=(1, 2)) / (valid_a.sum(dim=1) * D + 1e-8)

    observed_gap = (anti_per_sample - causal_per_sample).mean().item()
    print(f"  Error CAUSAL (Y|X):     {causal_error.item():.4f}")
    print(f"  Error ANTI-CAUSAL (X|Y): {anti_causal_error.item():.4f}")
    print(f"  Gap (anti - causal):    {observed_gap:.4f}")

    learned_causality = observed_gap > 0
    print(f"\n  {'✅' if learned_causality else '❌'} VJEPA "
          f"{'learned' if learned_causality else 'did NOT learn'} "
          f"causal direction")

    # ─── Permutation test ────────────────────────────────────────────
    n_perm = 500
    print(f"\n  Permutation test (n={n_perm}, paired sign-flip)...")
    per_sample_gap = anti_per_sample - causal_per_sample  # (B,)
    perm_diffs = []
    for _ in range(n_perm):
        swap = (torch.rand(B) > 0.5).float() * 2 - 1  # ±1 per sample
        perm_gap = (swap * per_sample_gap).mean().item()
        perm_diffs.append(perm_gap)
    perm_diffs = np.array(perm_diffs)
    p_value = (np.abs(perm_diffs) >= np.abs(observed_gap)).mean()
    print(f"  p-value: {p_value:.4f}  "
          f"{'✅ p<0.05 (significativo)' if p_value < 0.05 else '⚠️ Not significant'}")

# Save
results = {
    "causal_error": round(causal_error.item(), 4),
    "anti_causal_error": round(anti_causal_error.item(), 4),
    "gap": round(observed_gap, 4),
    "learned_causality": learned_causality,
    "permutation_p_value": round(float(p_value), 4),
    "significant": bool(p_value < 0.05),
}
with open(Path(__file__).parent.parent / "benchmarks" / "causal_vjepa_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to benchmarks/causal_vjepa_results.json")
