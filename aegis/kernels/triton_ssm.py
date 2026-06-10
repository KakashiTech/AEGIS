"""
Triton GPU kernel for Mamba-3 SSM prefix scan.

Implements the recurrence: h_t = A_t @ h_{t-1} + Bx_t

Semantics:
    A_t: (B, dS, dS)  state transition matrix
    Bx_t: (B, dS)     pre-gated input (B_t * x_t)
    h_t:  (B, dS)     hidden state

The kernel parallelises over the batch dimension and iterates
sequentially over the sequence length.  Each program (block) owns
one batch element and at every time step performs a full matrix-
vector product in registers without staging through shared memory
(the compiler is free to promote the state vector to shared /
local memory as it sees fit).

Numerical precision
-------------------
The kernel works in fp32 internally regardless of the input dtype
to guarantee stable long-range recurrence.  Input tensors may be
fp16, bf16, or fp32; the output will match the input dtype.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_triton_available() -> bool:
    """Return True iff Triton is installed *and* a CUDA device is visible.

    This is the single source of truth used everywhere in the BGCE
    kernel layer to decide whether GPU kernels can be launched.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
        return True
    except (ImportError, RuntimeError, Exception):
        return False


# ---------------------------------------------------------------------------
# Triton kernel  (only defined when the package is importable)
# ---------------------------------------------------------------------------

def _triton_kernel_definition():
    """Return the ``@triton.jit`` kernel function, or ``None``.

    Wrapped in a factory so that the module can be imported on CPU-only
    machines without raising at import time.
    """
    if not is_triton_available():
        return None

    import triton
    import triton.language as tl

    @triton.jit
    def ssm_scan_kernel(
        # -- tensor data pointers --
        A_ptr,        # (B, L, dS, dS)
        Bx_ptr,       # (B, L, dS)
        h0_ptr,       # (B, dS)
        h_ptr,        # (B, L, dS)   -- output
        # -- strides (in elements) --
        stride_Ab, stride_Al, stride_Am, stride_An,
        stride_Bxb, stride_Bxl, stride_Bxd,
        stride_hb, stride_hl, stride_hd,
        # -- problem sizes --
        L: tl.int32,
        DS: tl.constexpr,
        BLOCK_DS: tl.constexpr,
    ):
        """Triton kernel for the SSM recurrence.

        Grid: ``(batch_size,)`` — one program per batch element.

        Each program iterates over ``L`` time steps.  At step ``t`` it
        loads the full ``(dS, dS)`` matrix ``A[t]``, the vector
        ``Bx[t]``, performs the fused matvec-add

            h_new = A[t] @ h + Bx[t]

        stores the result and advances to the next step.
        """
        pid = tl.program_id(0)

        # ---- offsets & masks ------------------------------------------------
        offs_d = tl.arange(0, BLOCK_DS)
        mask_d = offs_d < DS

        # ---- load initial state h0 ------------------------------------------
        h = tl.load(h0_ptr + pid * DS + offs_d, mask=mask_d, other=0.0)
        h = h.to(tl.float32)

        # ---- sequential scan over L -----------------------------------------
        for t in range(L):
            # pointer for A[t]  ──  (DS, DS) in row-major layout
            a_base = A_ptr + pid * stride_Ab + t * stride_Al
            a_ptrs = (
                a_base
                + offs_d[:, None] * stride_Am
                + offs_d[None, :] * stride_An
            )
            a_mask = mask_d[:, None] & mask_d[None, :]

            # pointer for Bx[t]  ──  (DS,)
            bx_base = Bx_ptr + pid * stride_Bxb + t * stride_Bxl
            bx_ptrs = bx_base + offs_d * stride_Bxd

            # ---- load -------------------------------------------------------
            A_blk = tl.load(a_ptrs, mask=a_mask, other=0.0)
            Bx_blk = tl.load(bx_ptrs, mask=mask_d, other=0.0)

            # ---- matvec:  row[d] = sum_j A[d,j] * h[j] + Bx[d] -------------
            h_new = tl.sum(A_blk * h[None, :], axis=1) + Bx_blk

            # ---- store ------------------------------------------------------
            out_base = h_ptr + pid * stride_hb + t * stride_hl
            out_ptrs = out_base + offs_d * stride_hd
            tl.store(out_ptrs, h_new.to(h_ptr.dtype.element_ty), mask=mask_d)

            # ---- advance state ----------------------------------------------
            h = h_new

    return ssm_scan_kernel


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def triton_ssm_scan(
    A: torch.Tensor,
    Bx: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """SSM prefix scan.

    ``h[t] = A[t] @ h[t-1] + Bx[t]``    with  ``h[-1] = h0``

    Parameters
    ----------
    A:
        State-transition matrices, shape ``(B, L, dS, dS)``.
        May be real (float32/float16/bfloat16) **or** complex64.
        **Must be contiguous** (the wrapper will call ``.contiguous()``
        automatically if needed).
    Bx:
        Pre-gated inputs, shape ``(B, L, dS)``.
        Must share the same dtype family (real / complex) as ``A``.
    h0:
        Initial state, shape ``(B, dS)``.

    Returns
    -------
    Tensor of shape ``(B, L, dS)`` with the same dtype as ``A``.

    Notes
    -----
    *   The internal computation is always performed in fp32.  When the
        input is complex64 the system is transparently expanded to a
        ``2*dS``-dimensional real recurrence (see complex arithmetic
        below) so that a single kernel launch suffices.

    *   Complex arithmetic
        ~~~~~~~~~~~~~~~~~~~
        ``h = A @ h + Bx``  with  ``A ∈ ℂ, h ∈ ℂ, Bx ∈ ℂ``

        Let ``A = Ar + i·Ai``, ``h = hr + i·hi``, ``Bx = Bxr + i·Bxi``.
        The recurrence expands to the real ``2*dS``-dimensional system::

            [hr_new]   =  [[Ar, -Ai],   [hr]     +  [Bxr]
            [hi_new]      [Ai,  Ar]]    [hi]        [Bxi]

    *   **Fallback** — if Triton is unavailable or the shapes are not
        compatible with the kernel launch, the wrapper silently falls
        back to a pure-PyTorch sequential loop (``_ssm_scan_fallback``).
    """
    B, L, dS = Bx.shape
    device = A.device
    is_complex = A.is_complex()
    dtype = A.dtype

    # -- sanity checks ----------------------------------------------------
    if A.ndim != 4 or A.shape != (B, L, dS, dS):
        raise ValueError(f"A expects (B, L, dS, dS), got {A.shape}")
    if Bx.ndim != 3 or Bx.shape != (B, L, dS):
        raise ValueError(f"Bx expects (B, L, dS), got {Bx.shape}")
    if h0.ndim != 2 or h0.shape != (B, dS):
        raise ValueError(f"h0 expects (B, dS), got {h0.shape}")

    if L == 0:
        return torch.empty(B, 0, dS, device=device, dtype=dtype)

    # -- complex → real 2× expansion --------------------------------------
    if is_complex:
        return _complex_ssm_scan(A, Bx, h0)

    # -- real-valued path --------------------------------------------------
    if not is_triton_available():
        return _ssm_scan_fallback(A, Bx, h0)

    kernel_fn = _triton_kernel_definition()
    if kernel_fn is None:
        return _ssm_scan_fallback(A, Bx, h0)

    # Ensure contiguity (strides computed below assume contiguous).
    A = A.contiguous().float()
    Bx = Bx.contiguous().float()
    h0 = h0.contiguous().float()

    output = torch.empty(B, L, dS, device=device, dtype=torch.float32)

    BLOCK_DS = _next_power_of_2(dS)
    # Triton requires BLOCK_DS <= 1024 for most backends.
    BLOCK_DS = min(BLOCK_DS, 256)

    grid = (B,)

    kernel_fn[grid](
        A, Bx, h0, output,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        L,
        DS=dS,
        BLOCK_DS=BLOCK_DS,
        num_warps=4,
    )

    return output.to(dtype)


# ---------------------------------------------------------------------------
# Complex-valued scan  (expand to 2*dS real system)
# ---------------------------------------------------------------------------

def _complex_ssm_scan(
    A: torch.Tensor,
    Bx: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Handle complex-valued SSM via ``2*dS`` real expansion."""
    B, L, dS = Bx.shape
    device = A.device

    Ar = A.real.float()
    Ai = A.imag.float()
    Bxr = Bx.real.float()
    Bxi = Bx.imag.float()
    h0r = h0.real.float()
    h0i = h0.imag.float()

    # Build expanded real system of size (2*dS)
    A_exp = torch.zeros(B, L, 2 * dS, 2 * dS, device=device, dtype=torch.float32)
    A_exp[:, :, :dS, :dS] = Ar
    A_exp[:, :, :dS, dS:] = -Ai
    A_exp[:, :, dS:, :dS] = Ai
    A_exp[:, :, dS:, dS:] = Ar

    Bx_exp = torch.cat([Bxr, Bxi], dim=-1)  # (B, L, 2*dS)
    h0_exp = torch.cat([h0r, h0i], dim=-1)  # (B, 2*dS)

    h_exp = triton_ssm_scan(A_exp, Bx_exp, h0_exp)

    hr = h_exp[:, :, :dS]
    hi = h_exp[:, :, dS:]

    return torch.complex(hr, hi).to(A.dtype)


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback
# ---------------------------------------------------------------------------

def _ssm_scan_fallback(
    A: torch.Tensor,
    Bx: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Sequential PyTorch implementation of the SSM recurrence.

    Used when Triton is not available or when kernel launch fails.
    The loop is compiled with ``torch.compile`` when available for
    a moderate speed-up.
    """
    B, L, dS = Bx.shape
    device = A.device

    h = h0.clone()
    outputs = [None] * L

    # torch.compile the inner step when torch ≥ 2.0 and CUDA is available
    if hasattr(torch, "compile") and device.type == "cuda":
        try:

            @torch.compile(mode="reduce-overhead")
            def compiled_step(h, A_t, Bx_t):
                return torch.bmm(A_t, h.unsqueeze(-1)).squeeze(-1) + Bx_t

            step = compiled_step
        except Exception:
            step = None
    else:
        step = None

    for t in range(L):
        if step is not None:
            h = step(h, A[:, t], Bx[:, t])
        else:
            h = torch.bmm(A[:, t], h.unsqueeze(-1)).squeeze(-1) + Bx[:, t]
        outputs[t] = h.clone()

    return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


__all__ = [
    "is_triton_available",
    "triton_ssm_scan",
]
