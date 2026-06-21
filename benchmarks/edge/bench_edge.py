#!/usr/bin/env python3
"""
Edge Benchmark Suite for Diagonal++ SSM.

Measures inference performance on CPU across:
  - Model sizes (nano/tiny/small)
  - Sequence lengths (128 to 8192)
  - Batch sizes (1, 2, 4)

Reports: tok/s, latency P50/P95/P99.
"""

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig


def detect_hardware():
    info = {
        'platform': sys.platform,
        'processor': os.uname().machine if hasattr(os, 'uname') else 'unknown',
        'cpu_count': os.cpu_count() or 1,
    }
    try:
        if sys.platform == 'linux':
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        info['ram_gb'] = round(int(line.split()[1]) / 1_000_000, 1)
                        break
    except:
        pass
    info['cuda_available'] = torch.cuda.is_available()
    return info


def create_model(d_model=128, d_state=16, d_inner=256, n_layers=2):
    config = SSMConfig(d_model=d_model, d_state=d_state, d_inner=d_inner, n_layers=n_layers, use_kappa_truncation=False)
    model = Mamba3MIMO(config)
    model.eval()
    return model


def benchmark_inference(model, x, n_warmup=5, n_iters=30):
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        latencies = []
        for _ in range(n_iters):
            start = time.perf_counter()
            model(x)
            latencies.append((time.perf_counter() - start) * 1000)
    latencies = sorted(latencies)
    seq_len, batch_size = x.shape[1], x.shape[0]
    tokens = seq_len * batch_size
    return {
        'latency_ms_mean': sum(latencies) / len(latencies),
        'latency_ms_p50': latencies[len(latencies) // 2],
        'latency_ms_p95': latencies[int(len(latencies) * 0.95)],
        'latency_ms_p99': latencies[int(len(latencies) * 0.99)],
        'latency_ms_min': latencies[0],
        'latency_ms_max': latencies[-1],
        'tokens_per_sec': tokens / (sum(latencies) / len(latencies) / 1000),
    }


def run_benchmark_suite():
    print(f"\n{'='*70}")
    print(f"  EDGE BENCHMARK SUITE — Diagonal++ SSM")
    print(f"{'='*70}")
    hw = detect_hardware()
    print(f"\nHardware:")
    for k, v in hw.items():
        print(f"  {k}: {v}")
    results = {'hardware': hw, 'models': []}
    configs = [(64, 8, 128, 2, 'nano'), (128, 16, 256, 4, 'tiny'), (256, 32, 512, 6, 'small')]
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    for d_model, d_state, d_inner, n_layers, name in configs:
        print(f"\n{'─'*70}\n  Model: {name} (d={d_model}, dS={d_state}, L={n_layers})\n{'─'*70}")
        model = create_model(d_model, d_state, d_inner, n_layers)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_params:,}")
        print(f"\n  {'Seq_len':>8} {'Batch':>6} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'tok/s':>12}")
        print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

        for seq_len in seq_lengths:
            for batch_size in [1, 2, 4]:
                x = torch.randn(batch_size, seq_len, d_model)
                try:
                    stats = benchmark_inference(model, x, n_warmup=3, n_iters=15)
                    print(f"  {seq_len:>8} {batch_size:>6} {stats['latency_ms_mean']:>10.2f} {stats['latency_ms_p50']:>10.2f} {stats['latency_ms_p95']:>10.2f} {stats['tokens_per_sec']:>12.0f}")
                    results['models'].append({'name': name, 'params': total_params, 'seq_len': seq_len, 'batch_size': batch_size, **stats})
                except Exception as e:
                    print(f"  {seq_len:>8} {batch_size:>6} {'ERROR':>30} {str(e)[:30]}")

    os.makedirs('benchmarks/edge/results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = f'benchmarks/edge/results/bench_{ts}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {path}")
    best_tok = max(r['tokens_per_sec'] for r in results['models']) if results['models'] else 0
    print(f"Peak throughput: {best_tok:,.0f} tok/s")
    return results


if __name__ == "__main__":
    run_benchmark_suite()
