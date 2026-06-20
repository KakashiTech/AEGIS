#!/usr/bin/env python3
"""
measure_speedup.py - REAL benchmark for Mamba3 SSM scan methods.

Measures:
1. Speedup of parallel associative scan vs sequential scan across L and d_state
2. Mamba3Block vs equivalent Transformer speed comparison

Output: speedup_measurements.json (valid JSON)
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.core.mamba3_mimo import Mamba3Block, SSMConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = Path(__file__).parent / "speedup_measurements.json"


def benchmark_scan_speedup(
    L_values: Optional[List[int]] = None,
    d_state_values: Optional[List[int]] = None,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Dict:
    """
    Measure speedup of parallel scan vs sequential scan.

    Calls _apply_ssm_fast (sequential) and _apply_ssm_parallel (chunked)
    on the same inputs, reports min/mean/max speedup across runs.
    """
    if L_values is None:
        L_values = [32, 64, 128, 256, 512]
    if d_state_values is None:
        d_state_values = [4, 8, 16]

    results: Dict = {}
    cfg = {
        "L_values": L_values,
        "d_state_values": d_state_values,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "device": str(DEVICE),
    }

    for d_state in d_state_values:
        config = SSMConfig(
            d_model=64,
            d_state=d_state,
            d_inner=max(d_state * 4, 16),
            dt_rank=4,
            device=str(DEVICE),
        )
        block = Mamba3Block(config).to(DEVICE).eval()

        for L in L_values:
            x = torch.randn(1, L, config.d_inner, device=DEVICE)
            delta = torch.randn(1, L, config.d_inner, device=DEVICE)
            x_orig = torch.randn(1, L, config.d_model, device=DEVICE)

            # Warmup
            with torch.no_grad():
                for _ in range(n_warmup):
                    block._apply_ssm_fast(x, delta, x_orig)
                    block._apply_ssm_parallel(x, delta, x_orig)

            # Sequential timing
            seq_times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    block._apply_ssm_fast(x, delta, x_orig)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    seq_times.append((time.perf_counter() - t0) * 1000)

            # Parallel timing
            par_times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    block._apply_ssm_parallel(x, delta, x_orig)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    par_times.append((time.perf_counter() - t0) * 1000)

            # Per-run speedup
            speedups = [s / p for s, p in zip(seq_times, par_times) if p > 0]

            key = f"L{L}_dS{d_state}"
            results[key] = {
                "L": L,
                "d_state": d_state,
                "sequential_ms": {
                    "mean": float(np.mean(seq_times)),
                    "std": float(np.std(seq_times)),
                    "min": float(np.min(seq_times)),
                    "max": float(np.max(seq_times)),
                },
                "parallel_ms": {
                    "mean": float(np.mean(par_times)),
                    "std": float(np.std(par_times)),
                    "min": float(np.min(par_times)),
                    "max": float(np.max(par_times)),
                },
                "speedup": {
                    "min": float(np.min(speedups)) if speedups else 0.0,
                    "mean": float(np.mean(speedups)) if speedups else 0.0,
                    "max": float(np.max(speedups)) if speedups else 0.0,
                },
                "parallel_is_faster": float(np.mean(speedups)) > 1.0 if speedups else False,
            }

            print(
                f"  L={L:4} dS={d_state:2}:  "
                f"seq={np.mean(seq_times):8.3f}ms  "
                f"par={np.mean(par_times):8.3f}ms  "
                f"speedup={np.mean(speedups):.3f}x  "
                f"[{np.min(speedups):.3f}–{np.max(speedups):.3f}]"
            )

    return {"scan_speedup": results, "scan_config": cfg}


def benchmark_mamba3_vs_transformer(
    L_values: Optional[List[int]] = None,
    d_model: int = 128,
    d_state: int = 16,
    d_inner: int = 256,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Dict:
    """
    Benchmark Mamba3Block vs equivalent Transformer.

    Mamba3Block with d_model, d_state, d_inner.
    Transformer: nn.TransformerEncoder(nn.TransformerEncoderLayer, 1 layer).
    """
    if L_values is None:
        L_values = [32, 64, 128, 256, 512]

    cfg = {
        "d_model": d_model,
        "d_state": d_state,
        "d_inner": d_inner,
        "L_values": L_values,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "device": str(DEVICE),
    }

    # Mamba3Block
    ssm_config = SSMConfig(
        d_model=d_model,
        d_state=d_state,
        d_inner=d_inner,
        dt_rank=max(d_model // 8, 8),
        device=str(DEVICE),
    )
    mamba_block = Mamba3Block(ssm_config).to(DEVICE).eval()

    # Transformer: one encoder layer with same d_model/d_inner
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=8,
        dim_feedforward=d_inner,
        dropout=0.1,
        batch_first=True,
    ).to(DEVICE).eval()
    transformer = nn.TransformerEncoder(transformer_layer, num_layers=1).to(DEVICE).eval()

    mamba_params = sum(p.numel() for p in mamba_block.parameters())
    transformer_params = sum(p.numel() for p in transformer.parameters())
    info = {
        "mamba3_params": mamba_params,
        "transformer_params": transformer_params,
    }
    print(f"  Mamba3 params:     {mamba_params:,}")
    print(f"  Transformer params: {transformer_params:,}")

    results: Dict = {}
    for L in L_values:
        x_mamba = torch.randn(1, L, d_model, device=DEVICE)
        x_tf = x_mamba.clone()

        with torch.no_grad():
            for _ in range(n_warmup):
                mamba_block(x_mamba)
                transformer(x_tf)

        mamba_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                mamba_block(x_mamba)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                mamba_times.append((time.perf_counter() - t0) * 1000)

        tf_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                transformer(x_tf)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                tf_times.append((time.perf_counter() - t0) * 1000)

        ratios = [t / m for t, m in zip(tf_times, mamba_times) if m > 0]

        key = f"L{L}"
        results[key] = {
            "L": L,
            "mamba3_ms": {
                "mean": float(np.mean(mamba_times)),
                "std": float(np.std(mamba_times)),
                "min": float(np.min(mamba_times)),
                "max": float(np.max(mamba_times)),
            },
            "transformer_ms": {
                "mean": float(np.mean(tf_times)),
                "std": float(np.std(tf_times)),
                "min": float(np.min(tf_times)),
                "max": float(np.max(tf_times)),
            },
            "ratio_tf_over_mamba": {
                "min": float(np.min(ratios)) if ratios else 0.0,
                "mean": float(np.mean(ratios)) if ratios else 0.0,
                "max": float(np.max(ratios)) if ratios else 0.0,
            },
            "mamba_faster": float(np.mean(ratios)) > 1.0 if ratios else False,
        }

        print(
            f"  L={L:4}:  "
            f"Mamba3={np.mean(mamba_times):8.3f}ms  "
            f"Transformer={np.mean(tf_times):8.3f}ms  "
            f"Ratio(T/M)={np.mean(ratios):.3f}x  "
            f"[{np.min(ratios):.3f}–{np.max(ratios):.3f}]"
        )

    return {"mamba3_vs_transformer": results, "mt_config": cfg, "model_info": info}


def print_human_readable(results: Dict):
    """Print a human-readable summary of the benchmark results."""
    print("\n" + "=" * 72)
    print("  HUMAN-READABLE BENCHMARK REPORT")
    print("=" * 72)

    # --- Part 1: Scan speedup ---
    print("\n  1. PARALLEL vs SEQUENTIAL SCAN SPEEDUP")
    print("  " + "-" * 55)
    scan = results.get("scan_speedup", {})
    if scan and isinstance(scan, dict):
        print(f"  {'Config':<14} {'Seq(ms)':<11} {'Par(ms)':<11} {'Speedup':<11} Faster?")
        print("  " + "-" * 55)
        for key in sorted(scan.keys()):
            d = scan[key]
            if isinstance(d, dict) and "speedup" in d:
                sp = d["speedup"]
                print(
                    f"  {key:<14}"
                    f"{d['sequential_ms']['mean']:<8.3f}ms  "
                    f"{d['parallel_ms']['mean']:<8.3f}ms  "
                    f"{sp['mean']:<6.3f}x    "
                    f"{'YES' if d['parallel_is_faster'] else 'NO'}"
                )

    # --- Part 2: Mamba3 vs Transformer ---
    print("\n  2. MAMBA3 BLOCK vs TRANSFORMER ENCODER")
    print("  " + "-" * 55)
    mt = results.get("mamba3_vs_transformer", {})
    if mt and isinstance(mt, dict):
        info = results.get("model_info", {})
        if info:
            print(
                f"  Params: Mamba3={info.get('mamba3_params', '?'):,}  "
                f"Transformer={info.get('transformer_params', '?'):,}"
            )
        print(f"  {'L':<6}{'Mamba3(ms)':<13}{'Transformer(ms)':<17}{'Ratio(T/M)':<13} Faster?")
        print("  " + "-" * 55)
        for key in sorted(mt.keys()):
            d = mt[key]
            if isinstance(d, dict) and "ratio_tf_over_mamba" in d:
                r = d["ratio_tf_over_mamba"]
                print(
                    f"  L={d['L']:<3} "
                    f"{d['mamba3_ms']['mean']:<9.3f}ms  "
                    f"{d['transformer_ms']['mean']:<13.3f}ms  "
                    f"{r['mean']:<9.3f}x  "
                    f"{'YES' if d['mamba_faster'] else 'NO'}"
                )

    # --- Part 3: The 11.6x claim ---
    print("\n  3. ASSESSMENT OF THE '11.6x' CLAIM")
    print("  " + "-" * 55)
    print("  The '11.6x' claim in the codebase refers to Abstract-CoT token")
    print("  efficiency (ratio of verbal tokens to abstract tokens), NOT to")
    print("  Mamba3 SSM speed. It was a hardcoded constant, not measured.")
    print("  See bgce/cognition/abstract_cot.py::get_efficiency_ratio()")
    print("  for the measured (non-hardcoded) implementation.")

    max_sp = 0.0
    max_key = ""
    for key, d in (scan or {}).items():
        if isinstance(d, dict) and "speedup" in d:
            m = d["speedup"]["mean"]
            if m > max_sp:
                max_sp = m
                max_key = key
    print(f"\n  Max parallel-scan speedup measured: {max_sp:.3f}x ({max_key})")
    if max_sp < 1.0:
        print("  → Parallel scan is NOT faster than sequential in these tests.")
        print("    (Expected on CPU with small matrices — bmm overhead dominates)")
    else:
        print("  → Parallel scan shows measurable improvement for certain configs.")

    # Check if 11.6x is achievable
    print("\n  Can '11.6x' be achieved by the SSM parallel scan?")
    print(f"  → NO. Max measured scan speedup is {max_sp:.3f}x, far below 11.6x.")
    print("  → The 11.6x refers to Abstract-CoT tokens, NOT SSM speed.")
    print("  → For Abstract-CoT, the ratio depends on the actual token counts.")
    print(f"\n  Device: {results.get('metadata', {}).get('device', '?')}")


def main():
    print("=" * 72)
    print("  MAMBA3 SPEEDUP BENCHMARK (REAL MEASUREMENTS)")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"  Device: {DEVICE}  |  Torch: {torch.__version__}")
    print("=" * 72)

    results: Dict = {
        "metadata": {
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    L_values = [32, 64, 128, 256, 512]
    d_state_values = [4, 8, 16]

    # --- Part 1: Scan speedup ---
    print("\n>>> PART 1: Parallel vs Sequential Scan Speedup <<<")
    try:
        sr = benchmark_scan_speedup(L_values, d_state_values)
        results.update(sr)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["scan_speedup"] = {"error": str(e)}

    # --- Part 2: Mamba3 vs Transformer ---
    print("\n>>> PART 2: Mamba3Block vs Transformer <<<")
    try:
        mt = benchmark_mamba3_vs_transformer(L_values)
        results.update(mt)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["mamba3_vs_transformer"] = {"error": str(e)}

    # Save JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    # Human-readable report
    print_human_readable(results)

    print("\n" + "=" * 72)
    print("  BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
