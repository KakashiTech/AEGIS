#!/usr/bin/env python3
"""
Phase 2: Spectral SSM Benchmark.
Compares RSM vs Standard SSM vs Transformer perplexity & throughput.

Hypothesis: RSM (Fourier ω_k + hierarchical κ + exact ZOH) matches
Transformer perplexity at L=4096 with 4×+ throughput at equal parameter count.

Prediction falsifiable: RSM with dS=256 achieves ≤5% higher perplexity than
Transformer with d_model=768 while achieving ≥4× higher throughput at L=8192.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig


# ──────────────────────────────────────────────
# 1. Synthetic data with long-range structure
# ──────────────────────────────────────────────

class LongRangeDataset(torch.utils.data.Dataset):
    """
    Synthetic data with known long-range dependencies.
    Each sequence has a "control token" at position p that determines
    the output token at position p + gap. Models must learn to
    propagate information across gaps of up to 4096 tokens.
    """
    def __init__(self, vocab_size: int = 5000, seq_len: int = 4096,
                 gap: int = 1024, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.gap = min(gap, seq_len // 4)
        self.data = []
        for _ in range(num_samples):
            tokens = torch.randint(4, vocab_size, (seq_len,))
            ctrl_pos = seq_len // 4
            ctrl_val = torch.randint(0, 10, (1,)).item()
            tokens[ctrl_pos] = ctrl_val
            target_pos = min(ctrl_pos + self.gap, seq_len - 1)
            tokens[target_pos] = ctrl_val
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {'input_ids': x, 'labels': x.clone()}


# ──────────────────────────────────────────────
# 2. Transformer baseline
# ──────────────────────────────────────────────

class CausalTransformer(nn.Module):
    """Standard causal Transformer for baseline comparison"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, max_seq_len: int = 4096):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.drop = nn.Dropout(0.1)
        self.max_seq_len = max_seq_len

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer(
            'causal_mask',
            torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)

    def forward(self, input_ids, return_hidden=False):
        S = input_ids.size(1)
        x = self.embed(input_ids) + self.pos_embed[:, :S]
        x = self.drop(x)
        mask = self.causal_mask[:S, :S]
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        if return_hidden:
            return x
        return self.lm_head(x)

    def get_hidden_states(self, input_ids):
        return self.forward(input_ids, return_hidden=True)


# ──────────────────────────────────────────────
# 3. Benchmark runner
# ──────────────────────────────────────────────

@dataclass
class BenchConfig:
    lr: float = 3e-4
    warmup: int = 100
    steps: int = 1000
    eval_steps: int = 50
    batch_size: int = 2
    seq_len: int = 4096
    vocab_size: int = 5000
    d_model: int = 256
    d_state: int = 64
    d_inner: int = 512
    n_layers: int = 4
    n_heads: int = 4
    num_samples: int = 500


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def throughput_bench(model: nn.Module, seq_len: int, batch_size: int,
                     device: str, vocab_size: int = 5000,
                     num_warmup: int = 10, num_iter: int = 50) -> float:
    """Measure throughput in tokens/second"""
    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iter):
            model(dummy)
    elapsed = time.perf_counter() - start
    tokens = batch_size * seq_len * num_iter
    return tokens / elapsed


def train_and_eval(model: nn.Module, config: BenchConfig, device: str,
                   label: str) -> dict:
    """Train model on synthetic long-range data and return metrics"""
    dataset = LongRangeDataset(
        vocab_size=config.vocab_size,
        seq_len=min(config.seq_len, 2048),  # shorter for training speed
        gap=512,
        num_samples=config.num_samples
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / config.warmup, 1.0))

    model.train()
    losses = []
    step = 0
    while step < config.steps:
        for batch in loader:
            if step >= config.steps:
                break
            x = batch['input_ids'].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            losses.append(loss.item())
            if step % max(1, config.eval_steps) == 0:
                print(f"  [{label}] step {step}: loss {loss.item():.4f}")
            step += 1

    # Final perplexity at various lengths
    model.eval()
    perplexities = {}
    for length in [512, 1024, 2048, 4096]:
        if length > config.seq_len * 2:
            continue
        eval_data = LongRangeDataset(
            vocab_size=config.vocab_size,
            seq_len=length,
            gap=min(256, length // 4),
            num_samples=min(50, config.num_samples // 2)
        )
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=1)
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in eval_loader:
                x = batch['input_ids'].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), reduction='sum')
                total_loss += loss.item()
                total_tokens += x.numel()
        ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        perplexities[length] = ppl

    # Throughput
    tput = throughput_bench(model, config.seq_len, config.batch_size, device,
                            vocab_size=config.vocab_size)

    return {
        'params': count_params(model),
        'perplexities': perplexities,
        'throughput': tput,
        'final_loss': losses[-1] if losses else 0,
    }


# ──────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    config = BenchConfig()
    # Small config for CPU testing
    is_fast = device == 'cpu'
    if is_fast:
        config.d_model = 64
        config.d_state = 8
        config.d_inner = 128
        config.n_layers = 2
        config.n_heads = 2
        config.steps = 100
        config.eval_steps = 20
        config.batch_size = 2
        config.seq_len = 512
        config.vocab_size = 500
        config.num_samples = 50

    # --- Model A: RSM (Spectral SSM) ---
    ssm_rsm = SSMConfig(
        d_model=config.d_model, d_state=config.d_state,
        d_inner=config.d_inner, n_layers=config.n_layers,
        use_diagonal_ssm=True, use_spectral_ssm=True,
        vocab_size=config.vocab_size, device=device
    )
    model_rsm = Mamba3MIMO(ssm_rsm).to(device)
    print(f"RSM params: {count_params(model_rsm):,}")

    # --- Model B: Standard SSM (Diagonal++, no spectral) ---
    ssm_std = SSMConfig(
        d_model=config.d_model, d_state=config.d_state,
        d_inner=config.d_inner, n_layers=config.n_layers,
        use_diagonal_ssm=True, use_spectral_ssm=False,
        vocab_size=config.vocab_size, device=device
    )
    model_std = Mamba3MIMO(ssm_std).to(device)
    print(f"Standard SSM params: {count_params(model_std):,}")

    # --- Model C: Transformer ---
    model_tfm = CausalTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_inner,
        max_seq_len=max(config.seq_len, 4096)
    ).to(device)
    print(f"Transformer params: {count_params(model_tfm):,}")

    # --- Train & evaluate each ---
    results = {}

    print("\n─── Training RSM ───")
    results['rsm'] = train_and_eval(model_rsm, config, device, 'RSM')

    print("\n─── Training Standard SSM ───")
    results['std'] = train_and_eval(model_std, config, device, 'Standard')

    print("\n─── Training Transformer ───")
    results['tfm'] = train_and_eval(model_tfm, config, device, 'Transformer')

    # ── Results ──
    eval_lengths = [256, 512, 1024] if is_fast else [512, 1024, 2048, 4096]
    header = f"{'Model':<20} {'Params':<10} {'Throughput':<15} "
    for L in eval_lengths:
        header += f'PPL@{L:<10} '
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for label in ['rsm', 'std', 'tfm']:
        r = results[label]
        ppl_strs = ' '.join(f'{r["perplexities"].get(L, 0):<14.2f}' for L in eval_lengths)
        print(f"{label:<20} {r['params']:<10,} {r['throughput']:<15,.0f} {ppl_strs}")
    print("=" * len(header))

    # ── Verify hypothesis ──
    max_L = eval_lengths[-1]
    rsm_ppl = results['rsm']['perplexities'].get(max_L, float('inf'))
    std_ppl = results['std']['perplexities'].get(max_L, float('inf'))
    tfm_ppl = results['tfm']['perplexities'].get(max_L, float('inf'))
    rsm_tput = results['rsm']['throughput']
    std_tput = results['std']['throughput']
    tfm_tput = results['tfm']['throughput']

    print(f"\n─── Hypothesis Verification @ L={max_L} ───")
    print(f"RSM PPL:       {rsm_ppl:.2f}")
    print(f"Standard PPL:  {std_ppl:.2f}")
    print(f"Transformer PPL: {tfm_ppl:.2f}")
    print(f"RSM throughput:   {rsm_tput:,.0f} tok/s")
    print(f"Standard throughput: {std_tput:,.0f} tok/s")
    print(f"Transformer throughput: {tfm_tput:,.0f} tok/s")
    if tfm_ppl > 0 and tfm_ppl != float('inf'):
        print(f"RSM/TFM PPL ratio: {rsm_ppl / tfm_ppl:.3f}")
    print(f"RSM/TFM throughput ratio: {rsm_tput / tfm_tput:.1f}x")


if __name__ == '__main__':
    main()
