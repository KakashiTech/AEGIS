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
    TrapezoidalDiscretization, 
    ComplexValueDynamics,
    MIMOConv1d
)


def test_ssm_config():
    """Test de configuración SSM"""
    config = SSMConfig(d_model=768, d_state=64)
    assert config.d_model == 768
    assert config.d_state == 64
    assert config.use_complex == True
    assert config.use_mimo == True


def test_trapezoidal_discretization():
    """Test de discretización trapezoidal"""
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
    """Test de dinámica de valor complejo"""
    config = SSMConfig(d_model=768, d_state=64)
    cvd = ComplexValueDynamics(config)
    
    batch_size, seq_len = 2, 10
    h = torch.randn(batch_size, seq_len, config.d_state, dtype=torch.complex64)
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    h_new = cvd(h, x)
    
    # Verificar que es complejo
    assert h_new.dtype == torch.complex64
    assert h_new.shape == h.shape


def test_mimo_conv1d():
    """Test de convolución MIMO"""
    config = SSMConfig(d_model=768, d_inner=1536)
    mimo = MIMOConv1d(config)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.d_inner)
    
    output = mimo(x)
    
    # Verificar dimensiones
    assert output.shape == x.shape


def test_mamba3_forward():
    """Test de forward pass Mamba-3"""
    config = SSMConfig(d_model=256, d_state=32, n_layers=2)
    model = Mamba3MIMO(config)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward
    logits = model(input_ids)
    
    # Verificar dimensiones
    assert logits.shape == (batch_size, seq_len, 50000)
    
    # Verificar que no hay NaN
    assert not torch.isnan(logits).any()


def test_mamba3_hidden_states():
    """Test de obtención de estados ocultos"""
    config = SSMConfig(d_model=256, d_state=32, n_layers=2)
    model = Mamba3MIMO(config)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    hidden = model.get_hidden_states(input_ids)
    
    # Verificar dimensiones
    assert hidden.shape == (batch_size, seq_len, config.d_model)
    assert not torch.isnan(hidden).any()


def test_gradient_flow():
    """Test de flujo de gradientes"""
    config = SSMConfig(d_model=128, d_state=16, n_layers=2)
    model = Mamba3MIMO(config)
    
    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward
    logits = model(input_ids)
    loss = logits.mean()
    
    # Backward
    loss.backward()
    
    # Verificar que hay gradientes
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad


if __name__ == '__main__':
    print("Ejecutando tests de Mamba-3...")
    
    test_ssm_config()
    print("✓ SSMConfig")
    
    test_trapezoidal_discretization()
    print("✓ TrapezoidalDiscretization")
    
    test_complex_value_dynamics()
    print("✓ ComplexValueDynamics")
    
    test_mimo_conv1d()
    print("✓ MIMOConv1d")
    
    test_mamba3_forward()
    print("✓ Mamba3MIMO forward")
    
    test_mamba3_hidden_states()
    print("✓ Mamba3MIMO hidden states")
    
    test_gradient_flow()
    print("✓ Gradient flow")
    
    print("\n✓ Todos los tests de Mamba-3 pasaron!")
