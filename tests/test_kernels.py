"""Tests for GPU kernel layer (CPU fallback path)"""
import sys, torch
sys.path.insert(0, '.')
from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig


def test_kernel_dispatcher_cpu():
    config = SSMConfig(d_model=32, d_state=4, d_inner=64, n_layers=1,
                       use_diagonal_ssm=True, device='cpu')
    model = Mamba3MIMO(config).eval()
    x = torch.randint(0, 50000, (1, 16))
    out = model(x, return_hidden=True)
    assert out.shape == (1, 16, 32)


def test_kernel_dispatcher_cpu_standard_ssm():
    """Standard SSM path (TrapezoidalDiscretization)."""
    config = SSMConfig(d_model=32, d_state=4, d_inner=64, n_layers=1,
                       use_diagonal_ssm=False, device='cpu')
    model = Mamba3MIMO(config).eval()
    x = torch.randint(0, 50000, (1, 16))
    out = model(x, return_hidden=True)
    assert out.shape == (1, 16, 32)


def test_kernel_long_sequence():
    config = SSMConfig(d_model=32, d_state=4, d_inner=64, n_layers=1,
                       use_diagonal_ssm=True, device='cpu')
    model = Mamba3MIMO(config).eval()
    x = torch.randint(0, 50000, (1, 256))
    out = model(x, return_hidden=True)
    assert out.shape == (1, 256, 32)


def test_triton_fallback():
    from aegis.kernels.triton_ssm import triton_ssm_scan
    # Function should exist and be callable (even if triton not installed)
    assert callable(triton_ssm_scan)
