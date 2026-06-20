#!/usr/bin/env python3
"""
Integration experiment: Technique 1+2+3 combined.
Mide el efecto compuesto de Causal VJEPA + Diagonal++ + Thermo Regularizer.
"""
import sys, json, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.learning.vjepa import VJEPA, VJEPAConfig
from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig

device = "cpu"
d_model = 48
seq_len = 64
n_samples = 500
steps = 300
feature_dim = 4  # more features = more opportunity for causal structure
B = 32  # batch size

torch.manual_seed(42)
np.random.seed(42)

# Generate data with known causal structure:
#   dim 0 → dim 1 (direct cause)
#   dim 2 ← dim 1, dim 0 (joint effect)
#   dim 3 independent
L = seq_len
t = torch.linspace(0, 8*np.pi, L)
f0 = torch.sin(t) + 0.1 * torch.randn(L)
f1 = 0.6 * torch.roll(f0, 2) + 0.4 * f0 + 0.1 * torch.randn(L)
f2 = 0.5 * torch.roll(f0, 1) + 0.5 * torch.roll(f1, -1) + 0.1 * torch.randn(L)
f3 = 0.1 * torch.cumsum(torch.randn(L), dim=0)
base = torch.stack([f0, f1, f2, f3], dim=-1)
data = base.unsqueeze(0).expand(n_samples, -1, -1) + 0.05 * torch.randn(n_samples, L, feature_dim)
data = data.to(device)

causal_graph = {
    'causes': {0: [], 1: [0], 2: [0, 1], 3: []},
    'effects': {0: [1, 2], 1: [2], 2: [], 3: []},
    'independent': {3: []}
}

print("=" * 60)
print("Integration Experiment: Tecnica 1 + 2 + 3")
print("=" * 60)

results = {}

for label, use_diag, mask_strat, thermo_beta, d_state in [
    ("Baseline (standard SSM + random mask)",     False, "random", 0.0, 8),
    ("T2: Diagonal++",                            True,  "random", 0.0, 16),
    ("T2+T1: Diagonal++ + Causal mask",           True,  "causal_graph", 0.0, 16),
    ("T2+T1+T3: Diag++ + Causal + Thermo",        True,  "causal_graph", 0.01, 32),
]:
    config = SSMConfig(d_model=d_model, d_state=d_state, d_inner=96, dt_rank=4,
                       n_layers=2, use_diagonal_ssm=use_diag, device=device)
    backbone = Mamba3MIMO(config).to(device)
    
    v = VJEPAConfig(d_model=d_model, d_pred=d_model//2, predictor_depth=2,
                    mask_ratio=0.5, mask_strategy=mask_strat, loss_type='l1',
                    thermo_beta=thermo_beta, causal_graph=causal_graph, input_dim=feature_dim,
                    n_causal_features=feature_dim)
    model = VJEPA(backbone, v).to(device)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(model.predictor.parameters()), lr=3e-4)
    
    losses = []
    for step in range(steps):
        idx = torch.randperm(n_samples)[:B]
        batch = data[idx]
        out = model.forward(batch)
        loss = model.compute_loss(out)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target_encoder()
        losses.append(loss.item())
    # Compute
    final_loss = np.median(losses[-20:])
    results[label] = round(final_loss, 4)
    print(f"  {label}:")
    print(f"    Loss final: {final_loss:.4f}")

# Find winner
best_label = min(results, key=results.get)
baseline_loss = results.get("Baseline (standard SSM + random mask)", 1.0)
improvement = (baseline_loss - results[best_label]) / baseline_loss * 100
print(f"\n  Best configuration: {best_label}")
print(f"  Improvement vs baseline: {improvement:.1f}%")

assert len(results) == 4
print(f"\n✅ Integration experiment OK ({len(results)} configs)")
