#!/usr/bin/env python3
"""
train_traffic_anomaly.py — Train AEGIS on simulated network traffic.
Model: benign traffic (web browsing) vs malicious patterns (C2 beacon, port scan).
"""
import sys, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.security.aegis_cyber import AEGISCyberDefense, AEGISCyberConfig

# ─── Synthetic traffic generator ──────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

def gen_web_traffic(n, seq_len=64):
    """Simulate benign web browsing traffic. Features: [iat, pkt_len, ttl, sin(flags), cos(flags)]"""
    data = []
    for _ in range(n):
        iat = np.abs(np.random.exponential(0.05, seq_len)) * (1 + np.random.normal(0, 0.1, seq_len))
        pkt_size = np.where(np.random.rand(seq_len) < 0.7,
                            np.random.randint(40, 100, seq_len),
                            np.random.randint(500, 1500, seq_len))
        ttl = np.random.randint(48, 64, seq_len)
        flags = np.random.choice([0, 1], seq_len, p=[0.8, 0.2])
        features = np.column_stack([iat, pkt_size, ttl, np.sin(flags), np.cos(flags)])
        data.append(features)
    return np.array(data, dtype=np.float32)

def gen_c2_beacon(n, seq_len=64):
    """Simula C2 beacon. Features: [iat, pkt_len, ttl, sin(flags), cos(flags)]"""
    data = []
    for _ in range(n):
        t = np.linspace(0, 4 * np.pi, seq_len)
        iat = np.clip(0.05 + 0.008 * np.sin(t) + np.random.normal(0, 0.003, seq_len), 0.001, 1.0)
        pkt_size = np.random.randint(100, 300, seq_len)
        ttl = np.random.randint(64, 128, seq_len)
        flags = np.random.choice([2, 4, 16], seq_len, p=[0.6, 0.3, 0.1])
        features = np.column_stack([iat, pkt_size, ttl, np.sin(flags), np.cos(flags)])
        data.append(features)
    return np.array(data, dtype=np.float32)

def gen_port_scan(n, seq_len=64):
    """Simula port scan. Features: [iat, pkt_len, ttl, sin(flags), cos(flags)]"""
    data = []
    for _ in range(n):
        iat = np.abs(np.random.exponential(0.005, seq_len))
        pkt_size = np.random.randint(40, 80, seq_len)
        ttl = np.random.randint(32, 128, seq_len)
        flags = np.random.choice([0, 2, 4], seq_len, p=[0.3, 0.5, 0.2])
        features = np.column_stack([iat, pkt_size, ttl, np.sin(flags), np.cos(flags)])
        data.append(features)
    return np.array(data, dtype=np.float32)

# ─── Config ────────────────────────────────────────────────────────────────
device = "cpu"
seq_len = 64
d_model = 64
n_train = 200
n_eval = 100
steps = 80

config = AEGISCyberConfig(d_model=d_model, sequence_length=seq_len)
model = AEGISCyberDefense(config).to(device).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Generate data
X_benign = gen_web_traffic(n_train + n_eval // 2)
X_c2 = gen_c2_beacon(n_train + n_eval // 4)
X_scan = gen_port_scan(n_train + n_eval // 4)

X_all = np.vstack([X_benign, X_c2, X_scan])
y_all = np.array([0] * len(X_benign) + [1] * (len(X_c2) + len(X_scan)))

# Expand to d_model (AEGIS expects (B, L, d_model))
X_tensor = torch.FloatTensor(X_all)
y_tensor = torch.FloatTensor(y_all).unsqueeze(1)

# Train/eval split
n_total = len(X_all)
perm = np.random.permutation(n_total)
X_tensor, y_tensor = X_tensor[perm], y_tensor[perm]
split = n_total - n_eval
X_train, y_train = X_tensor[:split], y_tensor[:split]
X_eval, y_eval = X_tensor[split:], y_tensor[split:]

print(f"Traffic Anomaly Dataset:")
print(f"  Total: {n_total} samples")
print(f"  Benign: {len(X_benign)} | C2: {len(X_c2)} | Port scan: {len(X_scan)}")
print(f"  Train: {len(X_train)} | Eval: {len(X_eval)}")
print(f"  Features per packet: {X_all.shape[-1]}")
print()

# ─── Training ───────────────────────────────────────────────────────────────
for step in range(steps):
    idx = np.random.choice(len(X_train), 16, replace=False)
    batch = X_train[idx]  # (16, 64, 5) — raw flow features
    lbl = y_train[idx]

    # encode_flow handles any n_features dynamically
    flow_repr = model.flow_encoder.encode_flow(batch)
    processed = model.tvd_hl_ssm(flow_repr)
    raw_score, _, _ = model.tunnel_detector.detect(processed)

    loss = F.binary_cross_entropy(raw_score, lbl)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (step + 1) % 20 == 0:
        with torch.no_grad():
            acc = ((raw_score > 0.5).float() == lbl).float().mean().item()
            print(f"  Step {step+1}: loss={loss.item():.4f}  acc={acc:.3f}")

# ─── Evaluation ────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    fr = model.flow_encoder.encode_flow(X_eval)
    pr = model.tvd_hl_ssm(fr)
    scores, _, _ = model.tunnel_detector.detect(pr)
    scores_np = scores.squeeze().numpy()
    y_np = y_eval.squeeze().numpy()

    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    fpr, tpr, th = roc_curve(y_np, scores_np)
    auc = roc_auc_score(y_np, scores_np)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    opt_thresh = th[best_idx]
    preds = (scores_np > opt_thresh).astype(int)
    cm = confusion_matrix(y_np, preds)
    acc = (preds == y_np).mean()

print()
print("=" * 60)
print("TRAFFIC ANOMALY DETECTION RESULTS")
print("=" * 60)
print(f"ROC-AUC: {auc:.4f}")
print(f"Optimal threshold: {opt_thresh:.4f}")
print(f"Accuracy: {acc:.3f}")
print(f"Confusion matrix:")
print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
tpr_val = cm[1,1] / max(cm[1,1] + cm[1,0], 1)
fpr_val = cm[0,1] / max(cm[0,1] + cm[0,0], 1)
print(f"  TPR={tpr_val:.3f}  FPR={fpr_val:.3f}")

results = {
    "roc_auc": round(auc, 4),
    "optimal_threshold": round(opt_thresh, 4),
    "accuracy": round(acc, 4),
    "tpr": round(tpr_val, 4),
    "fpr": round(fpr_val, 4),
}
with open(Path(__file__).parent.parent / "benchmarks" / "traffic_anomaly_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to benchmarks/traffic_anomaly_results.json")
print(f"\n{'✅' if acc > 0.8 else '❌'} AEGIS detects anomalous traffic on simulated data")
