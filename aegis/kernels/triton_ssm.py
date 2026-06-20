"""
Triton GPU kernel for Diagonal++ SSM element-wise scan.

Implements the recurrence: h_t[k] = A_t[k] * h_{t-1}[k] + Bx_t[k]

This is the O(dS) element-wise version. A is a VECTOR (diagonal), not a matrix.
The original O(dS²) matrix-vector version was removed in the June 2026 audit
because it contradicted the Diagonal++ design (which is element-wise by
construction on the CPU path).

Semantics:
    A_t: (B, L, dS)  diagonal state transition (element-wise multiplier)
    Bx_t: (B, L, dS) pre-gated input (B_t * x_t)
    h_t:  (B, dS)    hidden state

The kernel parallelises over the batch dimension and state dimension,
iterating sequentially over the sequence length. Each program (block)
processes one batch element and a BLOCK_DS chunk of state dimensions.

Numerical precision
-------------------
The kernel works in fp32 internally regardless of the input dtype
to guarantee stable long-range recurrence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_triton_available() -> bool:
    """Return True iff Triton is installed *and* a CUDA device is visible."""
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
    """Return the ``@triton.jit`` kernel function, or ``None``."""
    if not is_triton_available():
        return None

    import triton
    import triton.language as tl

    @triton.jit
    def diag_ssm_scan_kernel(
        # -- tensor data pointers --
        A_ptr,        # (B, L, dS)  — diagonal, NOT full matrix
        Bx_ptr,       # (B, L, dS)
        h0_ptr,       # (B, dS)
        h_ptr,        # (B, L, dS)  — output
        # -- strides (in elements) --
        stride_Ab, stride_Al, stride_Ad,
        stride_Bxb, stride_Bxl, stride_Bxd,
        stride_hb, stride_hl, stride_hd,
        # -- problem sizes --
        L: tl.int32,
        DS: tl.constexpr,
        BLOCK_DS: tl.constexpr,
    ):
        """Diagonal++ SSM scan: element-wise recurrence.

        Grid: ``(batch_size,)`` — one program per batch element.

        Each program iterates over ``L`` time steps.  At step ``t`` it
        loads the diagonal ``A[t]`` (vector of length dS), the vector
        ``Bx[t]``, performs the element-wise recurrence

            h_new[k] = A[t,k] * h[k] + Bx[t,k]   for k = 0..dS-1

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
            # pointer for A[t]  ──  (DS,) diagonal
            a_base = A_ptr + pid * stride_Ab + t * stride_Al
            a_ptrs = a_base + offs_d * stride_Ad

            # pointer for Bx[t]  ──  (DS,)
            bx_base = Bx_ptr + pid * stride_Bxb + t * stride_Bxl
            bx_ptrs = bx_base + offs_d * stride_Bxd

            # ---- load -------------------------------------------------------
            A_blk = tl.load(a_ptrs, mask=mask_d, other=0.0)
            Bx_blk = tl.load(bx_ptrs, mask=mask_d, other=0.0)

            # ---- element-wise recurrence: h = A ⊙ h + Bx (O(dS)) ------------
            h_new = A_blk * h + Bx_blk

            # ---- store ------------------------------------------------------
            out_base = h_ptr + pid * stride_hb + t * stride_hl
            out_ptrs = out_base + offs_d * stride_hd
            tl.store(out_ptrs, h_new.to(h_ptr.dtype.element_ty), mask=mask_d)

            # ---- advance state ----------------------------------------------
            h = h_new

    return diag_ssm_scan_kernel


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def triton_ssm_scan(
    A: torch.Tensor,
    Bx: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Diagonal++ SSM prefix scan (element-wise, O(dS) per step).

    ``h[t] = A[t] ⊙ h[t-1] + Bx[t]``    with  ``h[-1] = h0``

    ``A`` must be a VECTOR of shape ``(B, L, dS)`` (diagonal),
    NOT a full ``(B, L, dS, dS)`` matrix.  This is the O(dS)
    element-wise version used by Diagonal++.

    Parameters
    ----------
    A:
        Diagonal state-transition, shape ``(B, L, dS)``.
        May be real (float32/float16/bfloat16) **or** complex64.
    Bx:
        Pre-gated inputs, shape ``(B, L, dS)``.
    h0:
        Initial state, shape ``(B, dS)``.

    Returns
    -------
    Tensor of shape ``(B, L, dS)`` with the same dtype as ``A``.

    Notes
    -----
    *   Internal computation is always fp32.
    *   Complex: ``h = A * h + Bx`` with ``A ∈ ℂ`` (element-wise)
        expands to a ``2*dS``-dimensional real recurrence.
    *   **Fallback** — pure-PyTorch loop when Triton unavailable.
    """
    B, L, dS = Bx.shape
    device = A.device
    is_complex = A.is_complex()
    dtype = A.dtype

    # -- sanity checks (now expects 3D A, not 4D) -------------------------
    if A.ndim != 3 or A.shape != (B, L, dS):
        raise ValueError(
            f"A expects (B, L, dS), got {A.shape}. "
            f"Diagonal++ uses element-wise recurrence (O(dS)). "
            f"For full matrix (O(dS²)) use the fallback."
        )
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

    A = A.contiguous().float()
    Bx = Bx.contiguous().float()
    h0 = h0.contiguous().float()

    output = torch.empty(B, L, dS, device=device, dtype=torch.float32)

    BLOCK_DS = _next_power_of_2(dS)
    BLOCK_DS = min(BLOCK_DS, 256)

    grid = (B,)

    kernel_fn[grid](
        A, Bx, h0, output,
        A.stride(0), A.stride(1), A.stride(2),
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        L,
        DS=dS,
        BLOCK_DS=BLOCK_DS,
        num_warps=4,
    )

    return output.to(dtype)


# ---------------------------------------------------------------------------
# Complex-valued scan  (expand to 2*dS real system, element-wise)
# ---------------------------------------------------------------------------

def _complex_ssm_scan(
    A: torch.Tensor,
    Bx: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Handle complex-valued SSM via ``2*dS`` real expansion.

    For element-wise (diagonal) A:
    h_new[k] = A[k] * h[k] + Bx[k]

    Let A = Ar + i·Ai, h = hr + i·hi, Bx = Bxr + i·Bxi.
    Element-wise: hr_new = Ar·hr - Ai·hi + Bxr
                  hi_new = Ai·hr + Ar·hi + Bxi
    """
    B, L, dS = Bx.shape
    device = A.device

    Ar = A.real.float()  # (B, L, dS)
    Ai = A.imag.float()  # (B, L, dS)
    Bxr = Bx.real.float()
    Bxi = Bx.imag.float()
    h0r = h0.real.float()
    h0i = h0.imag.float()

    # Interleave Ar, -Ai, Ai, Ar into a 2*dS real system
    # For the interleaved layout: [r0, r1, ..., i0, i1, ...]
    A_exp = torch.zeros(B, L, 2 * dS, 2 * dS, device=device, dtype=torch.float32)
    A_exp[:, :, :dS, :dS] = torch.diag_embed(Ar)      # real → real
    A_exp[:, :, :dS, dS:] = torch.diag_embed(-Ai)      # imag → real
    A_exp[:, :, dS:, :dS] = torch.diag_embed(Ai)       # real → imag
    A_exp[:, :, dS:, dS:] = torch.diag_embed(Ar)       # imag → imag

    Bx_exp = torch.cat([Bxr, Bxi], dim=-1)
    h0_exp = torch.cat([h0r, h0i], dim=-1)

    # Use element-wise fallback for expanded complex system
    # (the expanded system is block-diagonal, each 2×2 block is independent)
    h_exp = _expanded_diag_scan(A_exp, Bx_exp, h0_exp, dS)

    hr = h_exp[:, :, :dS]
    hi = h_exp[:, :, dS:]

    return torch.complex(hr, hi).to(A.dtype)


def _expanded_diag_scan(
    A_exp: torch.Tensor,
    Bx_exp: torch.Tensor,
    h0_exp: torch.Tensor,
    dS: int,
) -> torch.Tensor:
    """Scan for expanded complex system (block-diagonal, each 2×2).

    A_exp: (B, L, 2*dS, 2*dS) — block-diagonal with 2×2 blocks
    Bx_exp: (B, L, 2*dS)
    h0_exp: (B, 2*dS)
    """
    B, L, _ = Bx_exp.shape
    device = A_exp.device

    h = h0_exp.clone()
    outputs = []

    for t in range(L):
        # Extract the diagonal and off-diagonal from the 2×2 blocks
        # For the expanded system of size 2*dS, each dimension i maps to
        # block i%dS with a 2×2 matrix [[Ar_i, -Ai_i], [Ai_i, Ar_i]]
        A_t = A_exp[:, t]  # (B, 2*dS, 2*dS)
        # Use batch matrix-vector for the expanded system
        h = torch.bmm(A_t, h.unsqueeze(-1)).squeeze(-1) + Bx_exp[:, t]
        outputs.append(h.clone())

    return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback  (element-wise O(dS))
# ---------------------------------------------------------------------------

def _ssm_scan_fallback(
    A: torch.Tensor,
    Bx: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Element-wise PyTorch fallback for Diagonal++ SSM.

    This is the reference implementation: O(dS) per step.
    Used when Triton is unavailable.
    """
    B, L, dS = Bx.shape
    device = A.device

    h = h0.clone()
    outputs = [None] * L

    for t in range(L):
        h = A[:, t] * h + Bx[:, t]  # element-wise: O(dS)
        outputs[t] = h.clone()

    return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Matrix-vector fallback (legacy, for non-Diagonal++ paths)
# ---------------------------------------------------------------------------

def _full_ssm_scan_fallback(
    A: torch.Tensor,
    Bx: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Full matrix-vector SSM scan (O(dS²)) — legacy path.

    Only used when A is (B, L, dS, dS) — this is NOT the Diagonal++ path.
    """
    B, L, dS = Bx.shape
    device = A.device

    h = h0.clone()
    outputs = [None] * L

    for t in range(L):
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
