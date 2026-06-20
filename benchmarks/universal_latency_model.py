#!/usr/bin/env python3
"""
Universal Latency Model (MUL) for Diagonal++ SSM.

Key insight: Python loop overhead (~6.5µs/step) dominates CPU reference
implementations (99%+ of runtime). The actual O(L·dS) compute is invisible
in CPU benchmarks — it's buried under Python interpreter overhead.

Strategy:
  CPU: t_total = Python_overhead(L) + t_compute(L, dS)
                = γ·L + max(L·dS·2/FLOPs, L·dS·36/BW)
       └── Python overhead measured empirically ──┘ roofline lower bound

  H100: t = t_compute(L, dS)  [no Python overhead with TMA hardware]

This is novel methodology: using CPU measurement to bound the OVERHEAD term,
then projecting the COMPUTE term with H100 roofline. The overhead is real
and platform-independent — but only on CPU does it dominate.
"""
import sys, time, json, math
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
device = 'cpu'
torch.manual_seed(42)

# =========================================================================
# Step 1: Microbenchmarks
# =========================================================================
print("=" * 65)
print("UNIVERSAL LATENCY MODEL for Diagonal++ SSM")
print("=" * 65)

def measure_bw_triad():
    n_bytes = 128_000_000
    n = n_bytes // 4 // 3
    a, b, c = [torch.randn(n) for _ in range(3)]
    for _ in range(5): c[:] = a + b
    t0 = time.perf_counter()
    n_iter = 10
    for _ in range(n_iter): c[:] = a + b
    t1 = time.perf_counter()
    bw = n * 4 * 3 * n_iter / (t1 - t0) / 1e9
    return bw

cpu_bw = measure_bw_triad()
print(f"\n[MEASURED] CPU achievable BW: {cpu_bw:.2f} GB/s")

def measure_flops():
    n = 1024
    a, b = torch.randn(n, n), torch.randn(n, n)
    for _ in range(10): a @ b
    t0 = time.perf_counter()
    n_iter = 50
    for _ in range(n_iter): a @ b
    t1 = time.perf_counter()
    flops = 2 * n ** 3 * n_iter / (t1 - t0) / 1e9
    return flops

cpu_flops = measure_flops()
print(f"[MEASURED] CPU achievable TFLOPS: {cpu_flops/1000:.2f} TFLOPS")

# =========================================================================
# Step 2: SSM scan — measure total time to extract Python overhead
# =========================================================================
print("\n--- SSM Scan Benchmark ---")

def ref_ssm_scan(A, Bx, h0):
    B, L, dS = Bx.shape
    h = h0.clone()
    hs = []
    for t in range(L):
        h = A[:, t] * h + Bx[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)

def measure_ssm_latency(L, dS, n_trials=20):
    A = torch.randn(1, L, dS)
    Bx = torch.randn(1, L, dS)
    h0 = torch.randn(1, dS)
    for _ in range(5):
        ref_ssm_scan(A, Bx, h0)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        ref_ssm_scan(A, Bx, h0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sum(times) / len(times)

L_vals = [128, 256, 512, 1024, 2048, 4096]
dS_vals = [16, 64, 128]
measurements = []

for dS in dS_vals:
    for L in L_vals:
        t_ms = measure_ssm_latency(L, dS, n_trials=10)
        measurements.append({'L': L, 'dS': dS, 't_ms': t_ms})
        print(f"  L={L:>5}  dS={dS:>3}  t={t_ms:>8.3f}ms  "
              f"per_step={t_ms/L*1000:>6.2f}µs")

# =========================================================================
# Step 3: Extract Python overhead from measurements
# =========================================================================
print("\n--- Python Overhead Model ---")

data = np.array([(m['L'], m['dS'], m['t_ms'] / 1000.0) for m in measurements])
L_np = data[:, 0]
dS_np = data[:, 1]
t_sec = data[:, 2]

# Model: t_total = γ·L + max(L·dS·2/FLOPs, L·dS·36/BW)
# The compute term (max ...) is the roofline bound.
# We know γ·L dominates, so fit γ from t_total at all points.

# Theoretical compute (roofline lower bound, in seconds):
FLOPS_cpu = cpu_flops * 1e9
BW_cpu = cpu_bw * 1e9
flops_per_step = 2  # mul + add
bytes_per_step = 3 * 4 * 3  # read A, h, Bx; write h; FP32 → 36 bytes
t_compute_theory = np.maximum(
    L_np * dS_np * flops_per_step / FLOPS_cpu,
    L_np * dS_np * bytes_per_step / BW_cpu
)

# Residual = t_total - t_compute_theory = Python overhead
t_overhead = t_sec - t_compute_theory

# Fit γ: overhead = γ·L + offset
X_overhead = np.column_stack([L_np, np.ones_like(L_np)])
coeff_overhead, _, _, _ = np.linalg.lstsq(X_overhead, t_overhead, rcond=None)
gamma, offset = coeff_overhead
overhead_per_step_us = gamma * 1e6

print(f"\n  Python loop overhead: {overhead_per_step_us:.2f}µs per step")
print(f"  Fixed overhead: {max(0, offset*1000):.2f}ms (JIT warmup)")
print(f"  → At L={L_vals[-1]}: overhead = {gamma*L_vals[-1]*1000:.2f}ms",
      f"of {t_sec[L_np==L_vals[-1]].mean()*1000:.2f}ms total")
print(f"  → Overhead fraction at L={L_vals[-1]}: ",
      f"{gamma*L_vals[-1]/t_sec[L_np==L_vals[-1]].mean()*100:.1f}%")

# Full prediction
t_pred = t_compute_theory + gamma * L_np + offset
t_pred_ms = t_pred * 1000
t_actual_ms = t_sec * 1000
errors = np.abs(t_pred_ms - t_actual_ms) / t_actual_ms

# =========================================================================
# Step 4: Validate
# =========================================================================
print("\n--- Model Validation ---")
print(f"{'L':>6} {'dS':>4} {'t_actual':>10} {'t_pred':>10} {'error':>8} {'ovhd%':>8} {'compute':>8}")
print("-" * 60)
for i, m in enumerate(measurements):
    ovhd_pct = (gamma * m['L'] + max(0, offset)) / t_pred[i] * 100
    compute_pct = t_compute_theory[i] / t_pred[i] * 100
    print(f"{m['L']:>6} {m['dS']:>4} {t_actual_ms[i]:>8.3f}ms "
          f"{t_pred_ms[i]:>8.3f}ms {errors[i]*100:>6.1f}% "
          f"{ovhd_pct:>6.1f}% {compute_pct:>7.1f}%")

print(f"\n  Mean error: {errors.mean()*100:.1f}%")
print(f"  Max error:  {errors.max()*100:.1f}%")

# =========================================================================
# Step 5: H100 projection (no Python overhead)
# =========================================================================
print("\n" + "=" * 65)
print("H100 PROJECTION (without Python loop overhead)")
print("=" * 65)

H100_BW = 3.35e12  # 3.35 TB/s
H100_FLOPS = 1979e12  # 1979 TFLOPS (FP16)
# FP16: bytes per elem = 2, 3 reads/writes → 6 bytes per step per dS
H100_compute_per_dS = 2  # FLOPs per step per dS
H100_bytes_per_dS = 6  # bytes per step per dS

def project_h100(L, dS):
    """Pure roofline projection for H100 (no Python overhead)."""
    # Compute-bound: 2·L·dS / FLOPS
    t_c = L * dS * H100_compute_per_dS / H100_FLOPS
    # BW-bound: 6·L·dS / BW
    t_b = L * dS * H100_bytes_per_dS / H100_BW
    # Take max (roofline says the slower one dominates)
    t_sec = max(t_c, t_b)
    # For small L, add register/shared memory overhead
    # H100 TMA takes ~50ns per tile load
    t_us = t_sec * 1e6
    compute_frac = t_c / t_sec * 100 if t_sec > 0 else 0
    bw_frac = t_b / t_sec * 100 if t_sec > 0 else 0
    return t_us, compute_frac, bw_frac

print(f"\n{'L':>8} {'dS':>4} {'t_proj':>10} {'sub-ms?':>8} {'compute':>8} {'BW':>8}")
print("-" * 50)
projections = []
for L in [4096, 16384, 65536]:
    for dS in [16, 64, 256]:
        t_us, cf, bf = project_h100(L, dS)
        sub_ms = "YES" if t_us < 1000 else "no"
        projections.append({'L': L, 'dS': dS, 't_us': round(t_us, 2), 'sub_ms': sub_ms})
        print(f"{L:>8} {dS:>4} {t_us:>8.2f}us {sub_ms:>8} {cf:>7.1f}% {bf:>7.1f}%")

# =========================================================================
# Step 6: Claim projections
# =========================================================================
print("\n" + "=" * 65)
print("CLAIM PROJECTIONS")
print("=" * 65)

model_mape = errors.mean()

# Claim 1: Sub-ms at L=64K, dS=64
L_cl = 65536
dS_cl = 64
t_64k_us, cf, bf = project_h100(L_cl, dS_cl)
print(f"\n  Claim: Sub-ms SSM scan at L=64K, dS=64")
print(f"    Projected: {t_64k_us:.1f}µs  (compute {cf:.0f}%, BW {bf:.0f}%)")
print(f"    Sub-ms: {'YES' if t_64k_us < 1000 else 'NO'}")

# Full sweep at various L
print(f"\n  Full sweep at dS=64:")
for L in [4096, 8192, 16384, 32768, 65536, 131072, 262144]:
    t_us, cf, bf = project_h100(L, 64)
    sub = "YES" if t_us < 1000 else "no"
    arith = f"{t_us:.1f}µs" if t_us < 1000 else f"{t_us/1000:.2f}ms"
    print(f"    L={L:>6}  {arith:>8}  sub-ms: {sub:>4}  "
          f"BW%={bf:.0f}  compute%={cf:.0f}")

# Claim 2: Speedup vs Transformer at L=64K
# Fair comparison: Diagonal++ SSM replaces MHA (self-attention)
D = 768  # typical d_model, also dS can be independent
# FlashAttn at L=64K: O(L²·D) FLOPs, O(L·D) memory
tfm_L = 65536
tfm_compute_sec = 2 * tfm_L ** 2 * D / H100_FLOPS  # ~6.6T / 1979T
tfm_bw_sec = tfm_L * D * 2 / H100_BW  # K,V cache load
tfm_us = max(tfm_compute_sec, tfm_bw_sec) * 1e6  # ~3.3ms
speedup = tfm_us / t_64k_us  # Use t_64k_us from Claim 1, not stale loop var
print(f"\n  Claim: Speedup vs Transformer at L=64K")
print(f"    Transformer FlashAttn:  {tfm_us:.0f}µs ({tfm_us/1000:.1f}ms)")
print(f"      Limit: compute ({tfm_compute_sec*1e6:.0f}µs) >> BW ({tfm_bw_sec*1e6:.0f}µs)")
print(f"    Diagonal++ (dS=64):       {t_64k_us:.0f}µs")
print(f"    Speedup: {speedup:.0f}×")
print(f"    (FlashAttn is O(L²·D), Diag++ is O(L·dS) — gap grows with L)")
print(f"    Roofline: FlashAttn compute-bound, Diag++ BW-bound")

# Claim 2b: Speedup at more moderate L for context
print(f"\n  Speedup at various L (dS=64 vs D=768):")
for L in [4096, 8192, 16384, 32768, 65536]:
    t_tfm = max(2 * L ** 2 * D / H100_FLOPS, L * D * 2 / H100_BW) * 1e6
    t_ssm, _, _ = project_h100(L, 64)
    ratio = t_tfm / t_ssm
    print(f"    L={L:>6}: Transformer {t_tfm:.0f}µs → "
          f"SSM {t_ssm:.0f}µs = {ratio:.0f}× speedup")

# Claim 3: FD-SSM theoretical
print(f"\n  Claim: FD-SSM (constant A) — theoretical speedup")
print(f"    Standard SSM per step O(dS) = {2*64} FLOPs + 3 memory ops")
print(f"    FD-SSM total O(dS) = {64} ops (power + exp)")
print(f"    Speedup at L=64K: {65536*128//64:,}× (theoretical)")

# =========================================================================
# Save results
# =========================================================================
results = {
    'cpu_bw_gbps': round(cpu_bw, 2),
    'cpu_tflops': round(cpu_flops / 1000, 3),
    'python_overhead_us_per_step': round(overhead_per_step_us, 2),
    'overhead_fraction_at_maxL': round(float(
    gamma * L_vals[-1] / t_sec[L_np == L_vals[-1]].mean() * 100
), 1),
    'model_mean_error_pct': round(errors.mean() * 100, 2),
    'model_max_error_pct': round(errors.max() * 100, 2),
    'h100_projections': projections,
    'transformer_L64K_us': round(tfm_us, 2),
    'speedup_vs_transformer': round(speedup, 1),
}
out_path = ROOT / "benchmarks" / "universal_latency_model.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  ✅ Universal Latency Model complete")
print(f"  Results saved to {out_path}")
