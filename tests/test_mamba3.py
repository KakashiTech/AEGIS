#!/usr/bin/env python3
"""
Tests unitarios para Mamba-3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_complex_value_dynamics():
    """ComplexValueDynamics module was removed (dead code, never called)."""
    pass


if __name__ == "__main__":
    print("=== Mamba-3 SSS Tests ===")
    
    test_ssm_config()
    print("SSMConfig")
    
    test_trapezoidal_discretization()
    print("TrapezoidalDiscretization")
    
    test_mimo_conv1d()
    print("MIMOConv1d")
    
    test_mamba3_forward()
    print("Mamba3MIMO forward")
    
    test_mamba3_hidden_states()
    print("Mamba3MIMO hidden states")
    
    test_gradient_flow()
    print("Gradient flow")
    
    print("\n✓ Todos los tests de Mamba-3 pasaron!")
