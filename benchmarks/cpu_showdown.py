#!/usr/bin/env python3
"""
cpu_showdown.py — Diagonal++ SSM vs Transformer en CPU pura.

Benchmarks Mamba3MIMO (con Diagonal++) contra TransformerLM
con los mismos d_model, n_layers para L = 128..8192.

Resultado: muestra en qué L cruza la línea donde BGCE vence a Transformer.
"""
import json
import time
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig
from benchmarks.transformer_baseline import TransformerLM

DEVICE = torch.device("cpu")
RESULTS_FILE = Path(__file__).parent / "cpu_showdown_results.json"

CONFIG = dict(
    d_model=256,
    d_state=16,
    n_layers=6,
    dt_rank=8,
    d_inner=512,
    use_complex=True,
    use_mimo=True,
    use_diagonal_ssm=True,
    device="cpu",
)

L_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192]
N_WARMUP = 5
N_RUNS = 20
BATCH_SIZE = 1
VOCAB_SIZE = 50000


def benchmark_model(model, L: int, n_warmup: int, n_runs: int) -> tuple:
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, L), device=DEVICE)
    for _ in range(n_warmup):
        _ = model(input_ids)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(input_ids)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return min(times), sum(times) / len(times), max(times)


def main():
    print("=" * 70)
    print("CPU SHOWDOWN: Diagonal++ SSM vs Transformer")
    print("=" * 70)
    print(f"Config: d_model={CONFIG['d_model']}, d_state={CONFIG['d_state']}, "
          f"n_layers={CONFIG['n_layers']}")
    print(f"Device: {DEVICE}")
    print(f"L values: {L_VALUES}")
    print(f"Runs: {N_WARMUP} warmup + {N_RUNS} measured")
    print()

    # --- Build models ---
    print("Building Mamba3MIMO (Diagonal++)...")
    ssm_config = SSMConfig(**CONFIG)
    mamba_model = Mamba3MIMO(ssm_config).to(DEVICE).eval()

    print("Building TransformerLM...")
    transformer_model = TransformerLM(
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        n_heads=CONFIG['d_model'] // 32,  # head_dim=32
        vocab_size=VOCAB_SIZE,
        max_seq_len=max(L_VALUES),
    ).to(DEVICE).eval()

    print()
    results = {"config": CONFIG, "device": str(DEVICE), "results": {}}
    header = f"{'L':>6} | {'Mamba3 (ms)':>14} | {'Transformer (ms)':>16} | {'Ratio':>8} | {'Winner':>10}"
    print(header)
    print("-" * len(header))

    for L in L_VALUES:
        m_min, m_mean, m_max = benchmark_model(mamba_model, L, N_WARMUP, N_RUNS)
        t_min, t_mean, t_max = benchmark_model(transformer_model, L, N_WARMUP, N_RUNS)
        ratio = m_mean / t_mean if t_mean > 0 else float('inf')
        winner = "BGCE 🏆" if ratio < 1.0 else "Transformer"

        results["results"][str(L)] = {
            "mamba3_min_ms": round(m_min, 4),
            "mamba3_mean_ms": round(m_mean, 4),
            "mamba3_max_ms": round(m_max, 4),
            "transformer_min_ms": round(t_min, 4),
            "transformer_mean_ms": round(t_mean, 4),
            "transformer_max_ms": round(t_max, 4),
            "ratio_mamba_over_transformer": round(ratio, 4),
            "winner": winner,
        }
        print(f"{L:>6} | {m_mean:>8.4f} ms    | {t_mean:>8.4f} ms       | {ratio:>7.3f}x | {winner}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSIÓN:")
    for L_str, data in results["results"].items():
        L = int(L_str)
        if data["winner"] == "BGCE 🏆":
            print(f"  L={L:>5}: BGCE GANA ({data['ratio']:.3f}x más rápido)")
        else:
            print(f"  L={L:>5}: Transformer gana (BGCE a {data['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
