"""
Kernel dispatcher for H100-optimised SSM operations.

Dispatch priority chain:
    1. TileLang  — proprietary H100 TMA compiler (requires internal build)
    2. Triton    — open-source GPU code generator (CUDA)
    3. PyTorch   — pure-PyTorch sequential fallback

Every availability check is *real* (actual import / device probe) rather
than a hardcoded flag.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from .triton_ssm import is_triton_available, triton_ssm_scan


# ---------------------------------------------------------------------------
# Runtime availability probes (real, not hardcoded)
# ---------------------------------------------------------------------------

def _tilelang_available() -> bool:
    """Return True only if TileLang can actually be imported."""
    try:
        import tilelang  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def best_ssm_backend() -> str:
    """Return the best available SSM backend name.

    Returns one of ``"tilelang"``, ``"triton"``, ``"pytorch"``.
    """
    if _tilelang_available():
        return "tilelang"
    if is_triton_available():
        return "triton"
    return "pytorch"


# ---------------------------------------------------------------------------
# TileLangOptimizer — kernel dispatcher with fallback chain
# ---------------------------------------------------------------------------

class TileLangOptimizer:
    """
    SSM kernel dispatcher.

    Wraps the three-tier dispatch (TileLang → Triton → PyTorch) into a
    single ``.compile_ssm_scan_kernel()`` call so that model code never
    needs to worry about which backend is actually in use.

    The ``.available`` property reports whether at least one GPU-accelerated
    backend (TileLang or Triton) is reachable.
    """

    def __init__(self):
        self._tl_avail = _tilelang_available()
        self._tr_avail = is_triton_available()
        self.compiled_kernels: Dict[str, Callable] = {}
        self.target_latency_ms = 2.0

        # H100 hardware profile (used for tiling decisions)
        self.h100_config = {
            "sm_count": 132,
            "shared_memory_per_sm": 228,  # KB
            "tensor_cores_version": 4,
            "tma_available": self._tl_avail,
            "clock_rate_ghz": 1.98,
        }

    # -- availability --------------------------------------------------------

    @property
    def available(self) -> bool:
        """At least one GPU-accelerated backend is reachable."""
        return self._tl_avail or self._tr_avail

    def _check_tilelang(self) -> bool:
        """Legacy alias for :meth:`_tilelang_available`."""
        return self._tl_avail

    # -- SSM scan kernel -----------------------------------------------------

    def compile_ssm_scan_kernel(self) -> Optional[Callable]:
        """Return a callable implementing the SSM scan.

        The returned callable has the same signature as
        ``triton_ssm_scan(A, Bx, h0)``.

        Fallback:
            1. TileLang (proprietary, H100 TMA-optimised)
            2. Triton   (open-source, CUDA)
            3. PyTorch  (pure-PyTorch loop)

        The result is cached in ``self.compiled_kernels["ssm_scan"]``.
        """
        if self._tl_avail:
            kernel = self._compile_tilelang_ssm()
        elif self._tr_avail:
            kernel = triton_ssm_scan
        else:
            kernel = self._optimized_ssm_scan_pytorch

        self.compiled_kernels["ssm_scan"] = kernel
        return kernel

    def _compile_tilelang_ssm(self) -> Callable:
        """TileLang compilation path (placeholder until TileLang is public)."""
        # When TileLang becomes publicly available this method will JIT-
        # compile an H100 TMA-optimised SSM scan and return it.
        # ── placeholder ─────────────────────────────────────────────────
        def compiled_ssm_scan(A, Bx, h0):
            return triton_ssm_scan(A, Bx, h0)

        return compiled_ssm_scan

    # -- PyTorch fallback ----------------------------------------------------

    @staticmethod
    def _optimized_ssm_scan_pytorch(
        A: torch.Tensor,
        Bx: torch.Tensor,
        h0: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch sequential SSM scan.

        Used when neither TileLang nor Triton is available.
        The loop is JIT-compiled with ``torch.compile`` when possible.
        """
        B, L, dS = Bx.shape
        device = A.device
        h = h0.clone()
        outputs = []

        def step(h, A_t, Bx_t):
            return (A_t @ h.unsqueeze(-1)).squeeze(-1) + Bx_t

        if hasattr(torch, "compile") and device.type == "cuda":
            try:
                step = torch.compile(step, mode="reduce-overhead")
            except Exception:
                pass

        for t in range(L):
            h = step(h, A[:, t], Bx[:, t])
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    # -- Lorentz distance kernel --------------------------------------------

    def compile_lorentz_distance_kernel(self) -> Optional[Callable]:
        """Return a callable for Lorentz distance (placeholder)."""
        if not self.available:
            return None

        def compiled_lorentz_distance(x, y):
            time_dot = -x[..., 0] * y[..., 0]
            space_dot = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
            dot = time_dot + space_dot
            dot = torch.clamp(-dot, min=1.0 + 1e-8)
            return torch.acosh(dot)

        self.compiled_kernels["lorentz_distance"] = compiled_lorentz_distance
        return compiled_lorentz_distance

    # -- MIMO convolution kernel --------------------------------------------

    def compile_mimo_conv_kernel(self) -> Optional[Callable]:
        """Return a callable for MIMO convolution (placeholder)."""
        if not self.available:
            return None

        def compiled_mimo_conv(x, weight, gates):
            B, L, d_inner = x.shape
            x_expanded = torch.matmul(x, weight)
            x_expanded = x_expanded.view(B, L, d_inner, 4)
            gates_expanded = gates.unsqueeze(2)
            output = (x_expanded * gates_expanded).sum(dim=-1)
            return output

        self.compiled_kernels["mimo_conv"] = compiled_mimo_conv
        return compiled_mimo_conv

    # -- Model-level optimisation -------------------------------------------

    def optimize_for_latency(self, model: nn.Module) -> nn.Module:
        """Compile critical kernels and attempt ``torch.compile`` on the model.

        Returns the (possibly modified) model.
        """
        self.compile_ssm_scan_kernel()
        self.compile_lorentz_distance_kernel()
        self.compile_mimo_conv_kernel()

        if not self.available:
            print("[TileLang] No GPU backend available — using PyTorch optimisations")
            return self._pytorch_optimizations(model)

        backend = best_ssm_backend()
        print(f"[TileLang] Using backend: {backend}")
        return self._pytorch_optimizations(model)

    @staticmethod
    def _pytorch_optimizations(model: nn.Module) -> nn.Module:
        """Apply PyTorch-level optimisations (torch.compile, etc.)."""
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception:
                pass
        return model

    # -- Benchmarking -------------------------------------------------------

    def benchmark_latency(
        self,
        model: nn.Module,
        input_size: tuple,
        n_runs: int = 100,
    ) -> Dict[str, Any]:
        """Measure average inference latency in milliseconds."""
        import time

        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)

        # warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        latency_ms = (elapsed / n_runs) * 1000.0

        return {
            "avg_latency_ms": latency_ms,
            "target_latency_ms": self.target_latency_ms,
            "target_met": latency_ms < self.target_latency_ms,
            "input_size": input_size,
            "device": str(device),
            "backend": best_ssm_backend(),
        }


# ---------------------------------------------------------------------------
# TMAOptimizer — specific to H100 Tensor Memory Accelerator
# ---------------------------------------------------------------------------

class TMAOptimizer:
    """
    Layout optimisations for H100 Tensor Memory Accelerator (TMA).

    These are purely PyTorch-level transformations (contiguity, alignment)
    and do *not* require TileLang.
    """

    def __init__(self):
        self.block_sizes = {
            "small": (64, 64),
            "medium": (128, 128),
            "large": (256, 256),
        }

    @staticmethod
    def optimize_memory_layout(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.stride(-1) != 1:
            tensor = tensor.contiguous()
        return tensor

    @staticmethod
    def compute_optimal_block_size(dim: int) -> int:
        for size in (256, 128, 64):
            if dim % size == 0:
                return size
        return 64


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def create_optimized_model(
    model: nn.Module,
    use_tilelang: bool = True,
) -> nn.Module:
    """Build an optimised copy of *model*.

    When ``use_tilelang=True`` the dispatcher tries TileLang first, then
    Triton, then PyTorch.  When ``False`` it skips TileLang.
    """
    optimizer = TileLangOptimizer()
    backend = best_ssm_backend() if use_tilelang else "pytorch"

    if backend == "tilelang" and use_tilelang:
        print("[create_optimized_model] TileLang backend")
    elif backend == "triton":
        print("[create_optimized_model] Triton backend")
    else:
        print("[create_optimized_model] PyTorch backend")

    return optimizer.optimize_for_latency(model)


def verify_optimization() -> bool:
    """Print a status report and return True."""
    optimizer = TileLangOptimizer()

    print("=" * 60)
    print("TileLang / H100  —  optimisation status")
    print("=" * 60)
    print(f"  TileLang  (H100 TMA)  : {optimizer._tl_avail}")
    print(f"  Triton    (CUDA)      : {optimizer._tr_avail}")
    print(f"  Backend selected      : {best_ssm_backend()}")
    print(f"  Target latency        : {optimizer.target_latency_ms} ms")
    print()
    print("Hardware profile:")
    for k, v in optimizer.h100_config.items():
        print(f"  {k}: {v}")
    print()
    print("Registered kernels:")
    for name in ("ssm_scan", "lorentz_distance", "mimo_conv"):
        avail = name in optimizer.compiled_kernels
        print(f"  {name:<20s} {'✓' if avail else '—'}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    verify_optimization()
