#!/usr/bin/env python3
"""
Tests unitarios para VSA y Abstract-CoT
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from aegis.cognition.abstract_cot import (
    AbstractCoT, AbstractCoTConfig, VSAModule,
    HyperdimensionalEncoder, CircularConvolution,
    AbstractTokenizer
)


def test_circular_convolution():
    """Test de convolución circular"""
    conv = CircularConvolution(dim=64)
    
    a = torch.randn(2, 64)
    b = torch.randn(2, 64)
    
    # Binding
    bound = conv(a, b)
    
    # Verificar propiedad conmutativa: a ⊛ b ≈ b ⊛ a
    bound_ba = conv(b, a)
    
    assert torch.allclose(bound, bound_ba, atol=1e-5)
    
    # Verificar dimensiones
    assert bound.shape == (2, 64)


def test_circular_convolution_inverse():
    """Test de desvinculación"""
    conv = CircularConvolution(dim=64)
    
    a = torch.randn(2, 64)
    b = torch.randn(2, 64)
    
    # Binding
    bound = conv(a, b)
    
    # Desvincular
    b_recovered = conv.inverse(bound, a)
    
    # b_recovered debería ser cercano a b
    assert torch.allclose(b, b_recovered, atol=0.5)  # Tolerancia alta por ruido


def test_hyperdimensional_encoder():
    """Test de codificador hiperdimensional"""
    encoder = HyperdimensionalEncoder(vocab_size=1000, dim=256, num_positions=64)
    
    batch_size, seq_len = 2, 10
    symbols = torch.randint(0, 1000, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    encoded = encoder.encode(symbols, positions)
    
    # Verificar dimensiones
    assert encoded.shape == (batch_size, seq_len, 256)
    
    # Verificar que no hay NaN
    assert not torch.isnan(encoded).any()


def test_vsa_bind():
    """Test de binding VSA"""
    config = AbstractCoTConfig(binding_dim=128, use_vsa=True)
    vsa = VSAModule(config)
    
    role = torch.randn(2, 128)
    entity = torch.randn(2, 128)
    
    bound = vsa.bind(role, entity)
    
    # Verificar dimensiones
    assert bound.shape == (2, 128)
    
    # Verificar que no hay NaN
    assert not torch.isnan(bound).any()


def test_vsa_bundle():
    """Test de bundling VSA"""
    config = AbstractCoTConfig(binding_dim=128, use_vsa=True)
    vsa = VSAModule(config)
    
    vectors = [torch.randn(2, 128) for _ in range(5)]
    
    bundled = vsa.bundle(vectors)
    
    # Verificar dimensiones
    assert bundled.shape == (2, 128)
    
    # Verificar que no hay NaN
    assert not torch.isnan(bundled).any()
    
    # La norma debería estar normalizada
    assert torch.norm(bundled, dim=-1).mean().item() > 10


def test_vsa_unbind():
    """Test de desvinculación VSA"""
    config = AbstractCoTConfig(binding_dim=128, use_vsa=True)
    vsa = VSAModule(config)
    
    role = torch.randn(2, 128)
    entity = torch.randn(2, 128)
    
    # Binding
    bound = vsa.bind(role, entity)
    
    # Guardar en memoria
    memory = bound.unsqueeze(1)  # (B, 1, dim)
    
    # Recuperar
    recovered = vsa.unbind_and_query(memory, role)
    
    # Verificar dimensiones
    assert recovered.shape == (2, 128)


def test_abstract_tokenizer():
    """Test de tokenizador abstracto"""
    tokenizer = AbstractTokenizer(num_tokens=256)
    
    # Verificar tokens especiales
    assert tokenizer.encode_thought('start') == 256
    assert tokenizer.encode_thought('end') == 257
    
    # Verificar tokens abstractos
    token = tokenizer.get_abstract_token(100)
    assert token == 100


def test_abstract_cot_config():
    """Test de configuración Abstract-CoT"""
    config = AbstractCoTConfig(
        num_abstract_tokens=256,
        d_model=768,
        max_reasoning_steps=32
    )
    
    assert config.num_abstract_tokens == 256
    assert config.d_model == 768
    assert config.max_reasoning_steps == 32
    assert config.use_vsa == True


def test_abstract_cot_forward():
    """Test de forward pass Abstract-CoT"""
    config = AbstractCoTConfig(
        num_abstract_tokens=64,
        d_model=128,
        max_reasoning_steps=8,
        use_vsa=True,
        binding_dim=256
    )
    
    cot = AbstractCoT(config)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    context = torch.randn(batch_size, seq_len, config.d_model)
    
    # Forward con razonamiento
    outputs = cot(input_ids, context, use_reasoning=True)
    
    # Verificar que tiene las claves esperadas
    assert 'output' in outputs
    assert 'abstract_tokens' in outputs
    assert 'reasoning_states' in outputs
    
    # Verificar dimensiones del output
    assert outputs['output'].shape == (batch_size, seq_len, config.d_model)
    
    # Verificar que hay tokens abstractos
    assert outputs['abstract_tokens'] is not None
    assert outputs['abstract_tokens'].shape[0] == batch_size


def test_abstract_cot_efficiency():
    """Test de cálculo de eficiencia"""
    config = AbstractCoTConfig()
    cot = AbstractCoT(config)
    
    # Simular secuencia de 50 tokens verbales
    verbal_tokens = torch.randint(0, 1000, (1, 50))
    
    # Simular secuencia de 5 tokens abstractos
    abstract_tokens = torch.randint(0, 256, (1, 5))
    
    ratio = cot._compute_efficiency_ratio(verbal_tokens, abstract_tokens)
    
    # La eficiencia debería ser 50/5 = 10
    assert ratio == 10.0


def test_abstract_cot_decode():
    """Test de decodificación de tokens abstractos"""
    config = AbstractCoTConfig(num_abstract_tokens=256)
    cot = AbstractCoT(config)
    
    tokens = torch.tensor([[10, 20, 258, 259]])  # A10, A20, <think>, <end>
    
    decoded = cot.decode_abstract(tokens)
    
    assert len(decoded) == 1
    assert '[A10]' in decoded[0]
    assert '[A20]' in decoded[0]


if __name__ == '__main__':
    print("Ejecutando tests de VSA...")
    
    test_circular_convolution()
    print("✓ CircularConvolution")
    
    test_circular_convolution_inverse()
    print("✓ CircularConvolution inverse")
    
    test_hyperdimensional_encoder()
    print("✓ HyperdimensionalEncoder")
    
    test_vsa_bind()
    print("✓ VSA bind")
    
    test_vsa_bundle()
    print("✓ VSA bundle")
    
    test_vsa_unbind()
    print("✓ VSA unbind")
    
    test_abstract_tokenizer()
    print("✓ AbstractTokenizer")
    
    test_abstract_cot_config()
    print("✓ AbstractCoTConfig")
    
    test_abstract_cot_forward()
    print("✓ AbstractCoT forward")
    
    test_abstract_cot_efficiency()
    print("✓ AbstractCoT efficiency")
    
    test_abstract_cot_decode()
    print("✓ AbstractCoT decode")
    
    print("\n✓ Todos los tests de VSA pasaron!")
