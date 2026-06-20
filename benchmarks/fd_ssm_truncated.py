#!/usr/bin/env python3
"""
fd_ssm_truncated.py — Verifica el claim "O(dS) total para L=64K".

El claim: "FD-SSM procesa L=64K tokens en O(dS) total".

Caminos de verificación:
  1. Truncamiento + paralelismo: K pasos → error ε(K). En GPU (L procesadores),
     wall time = O(K·dS) en vez de O(L·dS). Speedup = L/K.
  2. K crítico = ln(ε)/ln(a_max) para peor dimensión (a_k = exp(dt·λ_k)).
  3. Para dt=0.01, a_max=0.995 → K_1%=922, K_0.1%=1843.

Resultado: O(K·dS) con K ~ 1000 para 1% — NO O(dS) puro, pero speedup 64×
vs scan secuencial a L=64K. Con curvatura aprendida (κ), K efectivo es menor.
"""
import json
import math
import time
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_FILE = Path(__file__).parent / "fd_ssm_truncated_results.json"


# ─── Core implementations ───────────────────────────────────────────────

def full_scan(A: torch.Tensor, c: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Full diagonal SSM scan: O(L·dS) sequential."""
    L = c.size(0)
    h = h0.clone()
    h_states = []
    for t in range(L):
        h = A[t] * h + c[t]
        h_states.append(h)
    return torch.stack(h_states, dim=0)


def truncated_parallel(
    A: torch.Tensor, c: torch.Tensor, h0: torch.Tensor, K: int
) -> torch.Tensor:
    """
    Truncated SSM PARALELO: O(K·dS) por token, todos independientes.

    Formula correcta (alineada con full_scan):
      h̃[t] = Σ_{j=max(0,t-K+1)}^{t} (Π_{m=j+1}^{t} A[m]) · c[j]

    Incluye c[t] y producto hasta A[t].
    Para K >= t+1, reproduce full_scan exactamente (con h0=0).

    En GPU con L procesadores: wall time = O(K·dS), no O(L·dS).
    """
    L = c.size(0)
    dS = c.size(1)
    device = c.device

    h = torch.zeros(L, dS, device=device, dtype=c.dtype)

    for t in range(L):
        start = max(0, t - K + 1)
        running_prod = torch.ones(dS, device=device, dtype=c.dtype)
        acc = torch.zeros(dS, device=device, dtype=c.dtype)
        for j in range(t, start - 1, -1):
            acc += running_prod * c[j]
            running_prod *= A[j]
        h[t] = acc

    return h


# ─── Theoretical analysis ──────────────────────────────────────────────

def compute_hippo_eigenvalues(dS: int, dt: float) -> torch.Tensor:
    """Autovalores HiPPO discretizados: a_k = exp(dt * -(k + 0.5))"""
    k = torch.arange(dS, dtype=torch.float32)
    eig = -(k + 0.5)
    return torch.exp(dt * eig)


def theoretical_K_for_error(A_diag: torch.Tensor, eps: float = 0.01) -> tuple:
    """
    K teórico para error < eps en cada dimensión.
    a_k^K < eps → K > ln(eps) / ln(a_k)

    Returns: (K_per_dim, avg_K, max_K)
    """
    K_per_dim = torch.where(
        A_diag > 0,
        torch.ceil(torch.log(torch.tensor(eps)) / torch.log(A_diag)).int(),
        torch.ones_like(A_diag, dtype=torch.int),
    )
    K_per_dim = torch.clamp(K_per_dim, min=0, max=100000)
    valid = (A_diag > 0) & (A_diag < 1)

    if valid.any():
        avg_K = float(K_per_dim[valid].float().mean())
        max_K = int(K_per_dim[valid].max())
    else:
        avg_K = 0.0
        max_K = 0

    return K_per_dim, avg_K, max_K


# ─── Error measurement ─────────────────────────────────────────────────

def measure_error(
    h_trunc: torch.Tensor,
    h_full: torch.Tensor,
    h_full_norm: torch.Tensor,
) -> dict:
    """Mide error relativo L2 por token."""
    error_per_token = (h_trunc - h_full).norm(dim=1) / h_full_norm
    return {
        "mean_error": round(float(error_per_token.mean()), 6),
        "max_error": round(float(error_per_token.max()), 6),
        "median_error": round(float(error_per_token.median()), 6),
        "p95_error": round(float(error_per_token.quantile(0.95)), 6),
    }


# ─── Benchmarks ────────────────────────────────────────────────────────

def benchmark_truncation(
    dS: int = 64,
    L: int = 4096,
    dt: float = 0.01,
    K_values=None,
    seed: int = 42,
) -> dict:
    """
    Benchmark: para cada K, mide error vs scan completo.
    NOTA: No mide speedup real en CPU (truncado es más lento secuencialmente).
    Speedup es teórico: L/K en GPU con L procesadores.
    """
    if K_values is None:
        K_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    device = torch.device("cpu")

    # Autovalores HiPPO
    A_diag = compute_hippo_eigenvalues(dS, dt)
    _, K_avg, K_max = theoretical_K_for_error(A_diag, eps=0.01)
    _, K_avg_1p, K_max_1p = theoretical_K_for_error(A_diag, eps=0.001)
    print(f"  HiPPO eigenvalues (dt={dt}): a ∈ [{float(A_diag.min()):.4f}, "
          f"{float(A_diag.max()):.4f}]")
    print(f"  K 1% error: avg≈{K_avg:.0f}, max={K_max}")
    print(f"  K 0.1% error: avg≈{K_avg_1p:.0f}, max={K_max_1p}")

    # Input
    torch.manual_seed(seed)
    c = torch.randn(L, dS, device=device) * 0.1
    h0 = torch.zeros(dS, device=device)
    A = A_diag.unsqueeze(0).expand(L, -1).clone()

    # Full scan baseline
    print(f"\n  Full scan (L={L}, dS={dS})...", end=" ")
    t0 = time.perf_counter()
    h_full = full_scan(A, c, h0)
    t_full = time.perf_counter() - t0
    print(f"{t_full*1000:.1f} ms")

    h_full_norm = h_full.norm(dim=1)
    h_full_norm[h_full_norm < 1e-10] = 1.0

    results = {
        "config": {"dS": dS, "L": L, "dt": dt, "K_values": K_values},
        "theoretical": {
            "K_avg_1pct": round(K_avg, 1),
            "K_max_1pct": K_max,
            "K_avg_0.1pct": round(K_avg_1p, 1),
            "K_max_0.1pct": K_max_1p,
            "warning": (
                "K_max (peor dimensión) domina error. K_avg (promedio) "
                "no es suficiente para error < 1%."
            ),
        },
        "full_scan_time_ms": round(t_full * 1000, 4),
        "results": {},
    }

    for K in K_values:
        print(f"  K={K:>4}...", end=" ")
        t0 = time.perf_counter()
        h_trunc = truncated_parallel(A, c, h0, K)
        t_trunc = time.perf_counter() - t0

        err = measure_error(h_trunc, h_full, h_full_norm)

        # Sweep speedup (L=64K reference)
        L_ref = 65536
        gpu_speedup_64k = L_ref / max(K, 1)

        err["truncated_time_ms"] = round(t_trunc * 1000, 4)
        err["gpu_parallel_speedup_vs_full_L64K"] = round(gpu_speedup_64k, 1)
        err["effective_K"] = K

        results["results"][str(K)] = err

        qual = ""
        if err["mean_error"] < 0.01:
            qual = " ✅ <1%"
        elif err["mean_error"] < 0.05:
            qual = " ~ OK"
        print(f"err_mean={err['mean_error']:.4f}, "
              f"max={err['max_error']:.4f}{qual}")

    return results


def sweep_dt():
    """Barre dt para entender relación dt → K_needed."""
    dt_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    dS = 64

    print(f"\n{'dt':>6} | {'a_min':>8} | {'a_max':>8} | "
          f"{'K_avg_1%':>9} | {'K_max_1%':>9} | "
          f"{'GPUspd@64K':>10}")
    print("-" * 70)

    results = {}
    for dt in dt_values:
        A_diag = compute_hippo_eigenvalues(dS, dt)
        _, K_avg, K_max = theoretical_K_for_error(A_diag, eps=0.01)
        L_ref = 65536
        parallel_speedup = L_ref / max(K_max, 1)
        results[str(dt)] = {
            "a_min": round(float(A_diag.min()), 6),
            "a_max": round(float(A_diag.max()), 6),
            "K_avg_1pct": round(K_avg, 1),
            "K_max_1pct": K_max,
            "gpu_parallel_speedup_at_64K": round(parallel_speedup, 1),
        }
        print(f"{dt:>6.3f} | {float(A_diag.min()):>8.4f} | "
              f"{float(A_diag.max()):>8.4f} | {K_avg:>8.1f} | "
              f"{K_max:>8} | {parallel_speedup:>7.0f}x")

    return results


def main():
    print("=" * 70)
    print("FD-SSM TRUNCATED: Verification of O(dS) Claim")
    print("=" * 70)
    print()
    print("Claim: 'FD-SSM procesa L=64K tokens en O(dS) total'")
    print()
    print("Mecanismo: Truncamiento + paralelismo en GPU")
    print("  Cada h[t] = Σ_{j=t-K+1}^{t} Π A · c[j]  (K pasos)")
    print("  En GPU con L procesadores: wall time O(K·dS), no O(L·dS)")
    print("  Speedup = L/K si K << L")
    print()

    # 1. Sweep dt
    print("=" * 70)
    print("SWEEP: dt → K_needed → GPU parallel speedup @ 64K")
    print("=" * 70)
    dt_results = sweep_dt()

    # 2. Main: dS=64, L=4096
    print("\n" + "=" * 70)
    print("BENCHMARK: dS=64, L=4096, dt=0.01")
    print("=" * 70)
    main_results = benchmark_truncation(
        dS=64, L=4096, dt=0.01,
        K_values=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )

    # 3. Small dS: dS=16
    print("\n" + "=" * 70)
    print("BENCHMARK: dS=16, L=4096, dt=0.01")
    print("=" * 70)
    small_results = benchmark_truncation(
        dS=16, L=4096, dt=0.01,
        K_values=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )

    # Compile
    all_results = {
        "interpretation": (
            "El claim 'O(dS) total' es PARCIALMENTE CORRECTO:\n"
            "  ✅ Truncamiento + paralelismo GPU: wall time = O(K·dS) vs O(L·dS)\n"
            "  ❌ K no es constante: depende de dt y la dimensión más lenta\n"
            "  📊 Para dt=0.01: K_1%_max=922, speedup_GPU@64K=71x\n"
            "  📊 Para dt=0.05: K_1%_max=185, speedup_GPU@64K=354x\n"
            "  📊 Para dt=0.10: K_1%_max=93, speedup_GPU@64K=705x\n\n"
            "Claim corregido: 'FD-SSM en GPU logra O(K·dS) con K ~ 100-1000\n"
            "para 1% error, dando speedup de 70-700× sobre scan secuencial.'\n\n"
            "Con curvatura aprendida (κ ∈ [0,1]), el dt efectivo puede ser\n"
            "mayor para dimensiones rápidas, reduciendo K_max aún más."
        ),
        "dt_sweep": dt_results,
        "dS_64": main_results,
        "dS_16": small_results,
        "conclusion": _conclusion(main_results),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados guardados en {RESULTS_FILE}")
    _print_conclusion(all_results)


def _conclusion(r):
    """Generate conclusion."""
    good = [(int(k), v) for k, v in r["results"].items()
            if v["mean_error"] < 0.01]
    if good:
        best_K, best_v = good[0]
        L_ref = 65536
        spd = L_ref / best_K
        return (
            f"CLAIM VERIFICABLE: K={best_K} da mean_error={best_v['mean_error']:.4f}, "
            f"max_error={best_v['max_error']:.4f}. "
            f"En GPU (L procesadores) wall time = O({best_K}·dS). "
            f"Speedup sobre scan secuencial a L=64K: ~{spd:.0f}×."
        )
    return "CLAIM NO VERIFICABLE en K testeados."


def _print_conclusion(all_results):
    print("\n" + "=" * 70)
    print("CONCLUSIÓN")
    print("=" * 70)
    print(all_results["conclusion"])
    print()

    for tag, data in [("dS=64", all_results["dS_64"]),
                      ("dS=16", all_results["dS_16"])]:
        print(f"\n--- {tag} ---")
        print(f"{'K':>5} | {'Err Mean':>9} | {'Err Max':>8} | "
              f"{'P95':>8} | {'GPUspd@64K':>10}")
        print("-" * 50)
        for K_str, d in data["results"].items():
            K = int(K_str)
            if K <= 128:
                print(f"{K:>5} | {d['mean_error']:>8.4f} | "
                      f"{d['max_error']:>7.4f} | {d['p95_error']:>7.4f} | "
                      f"{d['gpu_parallel_speedup_vs_full_L64K']:>7.0f}x")

    print("\nInterpretación:")
    for line in all_results["interpretation"].split("\n"):
        print(f"  {line}")


if __name__ == "__main__":
    main()
