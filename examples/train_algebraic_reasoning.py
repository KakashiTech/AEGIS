#!/usr/bin/env python3
"""
train_algebraic_reasoning.py — Entrena Abstract-CoT en razonamiento algebraico.

Demuestra que BGCE puede aprender a razonar con paréntesis anidados
y generalizar a profundidades no vistas durante entrenamiento.
"""
import sys, math, json, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig

# ─── Generate algebraic expressions ───────────────────────────────────────
def gen_expr(depth, seed=None):
    """Generate arithmetic expression with nested parentheses of given depth."""
    if seed is not None:
        random.seed(seed)
    ops = ['+', '-', '*']
    if depth <= 0:
        return str(random.randint(1, 9))
    left = gen_expr(depth - 1)
    right = gen_expr(depth - 1)
    op = random.choice(ops)
    return f'({left} {op} {right})'

def eval_expr(s):
    """Evaluate expression safely."""
    return eval(s)

# Vocab and data
ALL_CHARS = set('()0123456789+-*= ')
chars = sorted(ALL_CHARS)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join(itos[i] for i in l)

# Generate datasets with different depths
depths_train = [1, 2, 3]
depths_test = [4, 5]

def make_dataset(depths, n_per_depth=100, seed=42):
    random.seed(seed)
    data = []
    for d in depths:
        for _ in range(n_per_depth):
            expr = gen_expr(d)
            result = str(eval_expr(expr))
            # Format: "expr = result" → model learns to predict result
            text = f"{expr} = {result}"
            data.append(text)
    return data

train_data = make_dataset(depths_train, n_per_depth=200)
test_data_in = make_dataset(depths_train, n_per_depth=50)
test_data_out = make_dataset(depths_test, n_per_depth=50)

# Convert to tensors
def collate(data_list, block_size=128):
    encoded = [encode(s) for s in data_list]
    max_len = min(max(len(e) for e in encoded), block_size)
    x_list, y_list = [], []
    for e in encoded:
        if len(e) < 2:
            continue
        if len(e) > block_size:
            e = e[:block_size]
        x_list.append(e[:-1])
        y_list.append(e[1:])
    max_len = max(len(x) for x in x_list) if x_list else 1
    x_tensor = torch.zeros(len(x_list), max_len, dtype=torch.long)
    y_tensor = torch.zeros(len(y_list), max_len, dtype=torch.long)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        x_tensor[i, :len(x)] = torch.tensor(x)
        y_tensor[i, :len(y)] = torch.tensor(y)
    return x_tensor, y_tensor

train_x, train_y = collate(train_data)
test_in_x, test_in_y = collate(test_data_in)
test_out_x, test_out_y = collate(test_data_out)

# ─── Model ─────────────────────────────────────────────────────────────────
device = "cpu"
config = SSMConfig(
    d_model=64, d_state=8, d_inner=128, dt_rank=4, n_layers=3,
    use_complex=True, use_mimo=True, use_diagonal_ssm=True, device=device,
)
base_model = Mamba3MIMO(config).to(device)
# Override lm_head to match our vocab size
base_model.lm_head = nn.Linear(config.d_model, vocab_size, bias=False).to(device)
model = base_model
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

total_params = sum(p.numel() for p in model.parameters())
print(f"Vocab size: {vocab_size} ({''.join(chars)})")
print(f"Model params: {total_params:,}")
print(f"Train: depths {depths_train} ({len(train_data)} samples)")
print(f"Test in-dist: depths {depths_train} ({len(test_data_in)} samples)")
print(f"Test out-of-dist: depths {depths_test} ({len(test_data_out)} samples)")
print()

for step in range(300):
    perm = torch.randperm(train_x.size(0))[:16]
    bx, by = train_x[perm], train_y[perm]
    logits = model(bx)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), by.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    scheduler.step()
    if step % 50 == 0:
        with torch.no_grad():
            # In-distribution eval
            lid = F.cross_entropy(model(test_in_x).reshape(-1, vocab_size),
                                  test_in_y.reshape(-1)).item()
            # Out-of-distribution eval
            lod = float('nan')
            if test_out_x.size(0) > 0:
                lod = F.cross_entropy(model(test_out_x).reshape(-1, vocab_size),
                                      test_out_y.reshape(-1)).item()
            print(f"step {step:>4d}: train={loss.item():.4f}  test_in={lid:.4f}  test_ood={lod:.4f}" if not math.isnan(lod)
                  else f"step {step:>4d}: train={loss.item():.4f}  test_in={lid:.4f}")

# ─── Results ────────────────────────────────────────────────────────────────
with torch.no_grad():
    train_loss = F.cross_entropy(model(train_x).reshape(-1, vocab_size),
                                 train_y.reshape(-1)).item()
    test_in_loss = F.cross_entropy(model(test_in_x).reshape(-1, vocab_size),
                                   test_in_y.reshape(-1)).item()
    test_ood_loss = float('nan')
    if test_out_x.size(0) > 0:
        test_ood_loss = F.cross_entropy(model(test_out_x).reshape(-1, vocab_size),
                                        test_out_y.reshape(-1)).item()

print()
print("=" * 60)
print(f"RESULTADOS: Algebraic Reasoning")
print("=" * 60)
print(f"Train loss (depths 1-3): {train_loss:.4f}")
print(f"Test in-dist (depths 1-3): {test_in_loss:.4f}")
if not math.isnan(test_ood_loss):
    print(f"Test OOD (depths 4-5): {test_ood_loss:.4f}")
    print(f"OOD generalization gap: {test_ood_loss - test_in_loss:.4f}")

results = {
    "train_loss": round(train_loss, 4),
    "test_in_loss": round(test_in_loss, 4),
    "test_ood_loss": round(test_ood_loss, 4) if not math.isnan(test_ood_loss) else None,
    "generalization_gap": round(test_ood_loss - test_in_loss, 4) if not math.isnan(test_ood_loss) else None,
}
with open(Path(__file__).parent.parent / "benchmarks" / "algebraic_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to benchmarks/algebraic_results.json")
