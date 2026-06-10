#!/usr/bin/env python3
"""
Tests unitarios para capas de Lorentz
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
    """Test de variedad de Lorentz"""
    manifold = LorentzManifold(curvature=1.0, dim=4)
    
    # Crear vectores en Lorentz (componente temporal dominante)
    x = torch.tensor([[3.0, 1.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[2.5, 0.5, 0.5, 0.5, 0.5]])
    
    # Proyección
    x_proj = manifold.proj(x)
    y_proj = manifold.proj(y)
    
    # Verificar que no hay NaN ni Inf
    assert not torch.isnan(x_proj).any()
    assert not torch.isinf(x_proj).any()
    
    # Verificar propiedad: producto de Minkowski debe ser negativo
    minkowski_norm = manifold.minkowski_norm(x_proj)
    assert minkowski_norm.item() < 0


def test_minkowski_dot():
    """Test de producto interno de Minkowski"""
    manifold = LorentzManifold(curvature=1.0, dim=3)
    
    # x = (x_0, x_1, x_2, x_3)
    x = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[2.0, 0.5, 0.5, 0.5]])
    
    dot = manifold.minkowski_dot(x, y)
    
    # <x, y>_L = -x_0*y_0 + x_1*y_1 + x_2*y_2 + x_3*y_3
    expected = -2.0*2.0 + 1.0*0.5 + 1.0*0.5 + 1.0*0.5
    
    assert abs(dot.item() - expected) < 0.01


def test_lorentzian_distance():
    """Test de distancia lorentziana"""
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
    """Test de proyección a Lorentz"""
    proj = LorentzProjection(euclidean_dim=64, lorentz_dim=64, curvature=1.0)
    
    batch_size = 4
    x = torch.randn(batch_size, 64)
    
    x_lorentz = proj(x)
    
    # Verificar dimensiones
    assert x_lorentz.shape == (batch_size, 65)  # +1 por dimensión temporal
    
    # Verificar que está en la variedad
    manifold = LorentzManifold(curvature=1.0, dim=64)
    norm = manifold.minkowski_norm(x_lorentz)
    
    # Debería ser cercano a -1/κ
    assert abs(norm.mean().item() + 1.0) < 0.1


def test_lorentz_linear():
    """Test de capa lineal de Lorentz"""
    layer = LorentzLinear(in_features=64, out_features=100, curvature=1.0)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 65)  # 64 + 1 dimensión temporal
    
    output = layer(x)
    
    # Verificar dimensiones
    assert output.shape == (batch_size, seq_len, 100)
    
    # Verificar que no hay NaN
    assert not torch.isnan(output).any()


def test_poincare_projection():
    """Test de proyección Poincaré"""
    proj = PoincareProjection(dim=64, curvature=1.0)
    
    # Punto en Lorentz
    x_lorentz = torch.tensor([[2.0, 0.5, 0.3, 0.4]])
    
    # Proyectar a Poincaré
    x_poincare = proj.lorentz_to_poincare(x_lorentz)
    
    # La norma debe ser < 1 (dentro del disco)
    assert torch.norm(x_poincare).item() < 1.0
    
    # Proyección inversa
    x_back = proj.poincare_to_lorentz(x_poincare)
    
    # Verificar consistencia
    assert not torch.isnan(x_back).any()


def test_lorentz_attention():
    """Test de atención en Lorentz"""
    attn = LorentzAttention(dim=64, num_heads=4, curvature=1.0)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 65)
    
    output = attn(x)
    
    # Verificar dimensiones
    assert output.shape == (batch_size, seq_len, 64)
    
    # Verificar que no hay NaN
    assert not torch.isnan(output).any()


if __name__ == '__main__':
    print("Ejecutando tests de Lorentz...")
    
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
    
    print("\n✓ Todos los tests de Lorentz pasaron!")
