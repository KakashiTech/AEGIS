#!/usr/bin/env python3
"""
Unit tests for Lorentz layers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
import math

from aegis.geometry.lorentz_layers import (
    LorentzManifold, LorentzLinear, LorentzProjection,
    PoincareProjection, LorentzAttention
)


def test_lorentz_manifold():
    """Test Lorentz manifold"""
    manifold = LorentzManifold(curvature=1.0, dim=4)
    
    # Crear vectores en Lorentz (componente temporal dominante)
    x = torch.tensor([[3.0, 1.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[2.5, 0.5, 0.5, 0.5, 0.5]])
    
    # Projection
    x_proj = manifold.proj(x)
    y_proj = manifold.proj(y)
    
    # Verificar que no hay NaN ni Inf
    assert not torch.isnan(x_proj).any()
    assert not torch.isinf(x_proj).any()
    
    # Verificar propiedad: producto de Minkowski debe ser negativo
    minkowski_norm = manifold.minkowski_norm(x_proj)
    assert minkowski_norm.item() < 0


def test_minkowski_dot():
    """Test Minkowski dot product"""
    manifold = LorentzManifold(curvature=1.0, dim=3)
    
    # x = (x_0, x_1, x_2, x_3)
    x = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[2.0, 0.5, 0.5, 0.5]])
    
    dot = manifold.minkowski_dot(x, y)
    
    # <x, y>_L = -x_0*y_0 + x_1*y_1 + x_2*y_2 + x_3*y_3
    expected = -2.0*2.0 + 1.0*0.5 + 1.0*0.5 + 1.0*0.5
    
    assert abs(dot.item() - expected) < 0.01


def test_lorentzian_distance():
    """Test Lorentzian distance"""
    manifold = LorentzManifold(curvature=1.0, dim=3)
    
    # Dos puntos cercanos en Lorentz
    x = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[2.1, 1.0, 1.0, 1.0]])
    
    dist = manifold.lorentzian_distance(x, y)
    
    # La distancia debe ser positiva y finita
    assert dist.item() > 0
    assert not torch.isnan(dist)
    assert not torch.isinf(dist)


def test_lorentz_projection():
    """Test Lorentz projection"""
    proj = LorentzProjection(euclidean_dim=64, lorentz_dim=64, curvature=1.0)
    
    batch_size = 4
    x = torch.randn(batch_size, 64)
    
    x_lorentz = proj(x)
    
    # Verificar dimensiones
    assert x_lorentz.shape == (batch_size, 65)  # +1 for time dimension
    
    # Verify it is on the manifold
    manifold = LorentzManifold(curvature=1.0, dim=64)
    norm = manifold.minkowski_norm(x_lorentz)
    
    # Should be close to -1/κ
    assert abs(norm.mean().item() + 1.0) < 0.1


def test_lorentz_linear():
    """Test Lorentz linear layer"""
    layer = LorentzLinear(in_features=64, out_features=100, curvature=1.0)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 65)  # 64 + 1 time dimension
    
    output = layer(x)
    
    # Verificar dimensiones
    assert output.shape == (batch_size, seq_len, 100)
    
    # Verificar que no hay NaN
    assert not torch.isnan(output).any()


def test_poincare_projection():
    """Test Poincare projection"""
    proj = PoincareProjection(dim=64, curvature=1.0)
    
    # Punto en Lorentz
    x_lorentz = torch.tensor([[2.0, 0.5, 0.3, 0.4]])
    
    # Project to Poincare
    x_poincare = proj.lorentz_to_poincare(x_lorentz)
    
    # La norma debe ser < 1 (dentro del disco)
    assert torch.norm(x_poincare).item() < 1.0
    
    # Projection inversa
    x_back = proj.poincare_to_lorentz(x_poincare)
    
    # Verificar consistencia
    assert not torch.isnan(x_back).any()


def test_lorentz_attention():
    """Test Lorentz attention"""
    attn = LorentzAttention(dim=64, num_heads=4, curvature=1.0)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 65)
    
    output = attn(x)
    
    # Verificar dimensiones
    assert output.shape == (batch_size, seq_len, 64)
    
    # Verificar que no hay NaN
    assert not torch.isnan(output).any()


if __name__ == '__main__':
    print("Running Lorentz tests...")
    
    test_lorentz_manifold()
    print("✓ LorentzManifold")
    
    test_minkowski_dot()
    print("✓ Minkowski dot product")
    
    test_lorentzian_distance()
    print("✓ Lorentzian distance")
    
    test_lorentz_projection()
    print("✓ LorentzProjection")
    
    test_lorentz_linear()
    print("✓ LorentzLinear")
    
    test_poincare_projection()
    print("✓ PoincareProjection")
    
    test_lorentz_attention()
    print("✓ LorentzAttention")
    
    print("\n✓ All Lorentz tests passed!")
