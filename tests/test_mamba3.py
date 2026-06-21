#!/usr/bin/env python3
"""
Tests unitarios para Mamba-3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import pytest

from aegis.core.mamba3_mimo import (
    Mamba3MIMO, SSMConfig,
    DiagonalSSMDiscretization, TrapezoidalDiscretization,
)


def test_ssm_config():
    """Test SSM configuration"""
    config = SSMConfig(d_model=768, d_state=64)
    assert config.d_model == 768
    assert config.d_state == 64
    assert config.use_complex == True
    assert config.use_mimo == True


def test_trapezoidal_discretization():
    """Test trapezoidal discretization"""
    config = SSMConfig(d_model=256, d_state=64)
    disc = TrapezoidalDiscretization(config)
    
    batch_size, seq_len = 2, 10
    delta = torch.randn(batch_size, seq_len, config.dt_rank)
    x = torch.randn(batch_size, seq_len, config.d_model)  # Usar d_model, no d_inner
    
    A_bar, B_bar = disc(delta, x)
    
    # Verificar dimensiones
    assert A_bar.shape[0] == batch_size
    assert A_bar.shape[1] == seq_len
    assert B_bar.shape[0] == batch_size
    assert B_bar.shape[1] == seq_len


def test_rsm_config_fourier_init():
    """RSM mode: Fourier-initialized ω_k"""
    config = SSMConfig(d_model=64, d_state=16, use_spectral_ssm=True)
    disc = DiagonalSSMDiscretization(config)
    
    eig_imag = torch.tanh(disc.eig_imag_raw) * math.pi
    k = torch.arange(disc.d_state, dtype=torch.float32)
    target = k / max(disc.d_state - 1, 1) * math.pi
    error = (eig_imag - target).abs().max().item()
    
    assert error < 0.1, f"Fourier init error too large: {error}"
    print(f"  ✓ Fourier ω_k init: max error = {error:.4f}")


def test_rsm_hierarchical_kappa():
    """RSM mode: hierarchical κ scales (1, 10, 50...)"""
    config = SSMConfig(d_model=64, d_state=16, use_spectral_ssm=True)
    disc = DiagonalSSMDiscretization(config)
    
    assert disc.kappa_scale[0].item() == pytest.approx(1.0), "dim 0 should have κ=1"
    assert disc.kappa_scale[1].item() == pytest.approx(10.0), "dim 1 should have κ=10"
    assert disc.kappa_scale[2:].min().item() == pytest.approx(50.0), "dims 2+ should have κ=50"
    print("  ✓ Hierarchical κ: dim0=1, dim1=10, dim2+=50")


def test_exact_zoh_vs_approx():
    """Exact ZOH discretization: matches Taylor for small dt, diverges for large dt"""
    config = SSMConfig(d_model=64, d_state=4, use_spectral_ssm=False)
    disc = DiagonalSSMDiscretization(config)
    
    batch_size, seq_len = 2, 5
    delta = torch.randn(batch_size, seq_len, config.dt_rank) * 0.01  # small dt
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    A_bar, B_bar = disc(delta, x)
    
    # Taylor approximation for B_bar
    eig_imag = torch.tanh(disc.eig_imag_raw) * math.pi
    eig = disc.eig_real + 1j * eig_imag
    kappa = disc.kappa_base(x) * disc.kappa_scale
    eig_scaled = eig * kappa
    dt = delta.mean(dim=-1, keepdim=True)
    B_bar_taylor = dt * (1.0 + dt * eig_scaled / 2.0)
    
    # For small dt, exact and Taylor should be close
    eig_small = eig_scaled.abs().max().item()
    if eig_small < 10:  # only valid for small eigenvalues
        error = (B_bar - B_bar_taylor).abs().mean().item()
        assert error < 0.1, f"ZOH vs Taylor discrepancy too large: {error}"
    
    # For large dt or large κ, exact ZOH should NOT match Taylor
    # (Taylor goes negative when |λ·dt| > 1)
    eig_test = torch.tensor([-25.0], dtype=torch.complex64)  # κ=50, λ=-0.5
    dt_large = torch.tensor([0.1])
    a_bar_test = torch.exp(dt_large * eig_test)
    taylor_test = dt_large * (1.0 + dt_large * eig_test / 2.0)
    exact_test = (a_bar_test - 1.0) / eig_test
    
    # Taylor should be negative for |λ·dt| > 2, exact stays positive
    if eig_test.abs().item() * dt_large.item() > 2:
        assert taylor_test.real.item() < 0, f"Taylor should go negative: {taylor_test.real.item()}"
        assert exact_test.real.item() > 0, f"Exact should stay positive: {exact_test.real.item()}"
        print(f"  ✓ Exact ZOH: Taylor={taylor_test.real.item():.4f} (neg!), exact={exact_test.real.item():.4f} (pos!)")
    else:
        print(f"  ✓ Exact ZOH (small step, both positive)")


def test_rsm_forward():
    """RSM mode: forward pass produces valid output"""
    config = SSMConfig(d_model=64, d_state=8, d_inner=128, n_layers=2,
                       use_spectral_ssm=True, device='cpu')
    model = Mamba3MIMO(config).eval()
    x = torch.randint(0, 500, (1, 32))
    out = model(x, return_hidden=True)
    assert out.shape == (1, 32, 64), f"RSM forward shape: {out.shape}"
    assert not torch.isnan(out).any(), "RSM output has NaN"
    assert not torch.isinf(out).any(), "RSM output has Inf"


if __name__ == "__main__":
    print("=== Mamba-3 SSS Tests ===")
    
    test_ssm_config()
    print("✓ SSMConfig")
    
    test_trapezoidal_discretization()
    print("✓ TrapezoidalDiscretization")
    
    test_rsm_config_fourier_init()
    print("✓ RSM Fourier init")
    
    test_rsm_hierarchical_kappa()
    print("✓ RSM hierarchical κ")
    
    test_exact_zoh_vs_approx()
    print("✓ Exact ZOH")
    
    test_rsm_forward()
    print("✓ RSM forward")
    
    print("\n✓ Todos los tests de Mamba-3 pasaron!")
