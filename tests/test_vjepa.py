#!/usr/bin/env python3
"""
Unit tests for VJEPA (Vicinal Joint-Embedding Predictive Architecture)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from aegis.learning.vjepa import (
    VJEPA, VJEPAConfig, TargetEncoder, Predictor,
    EBM_energy, MaskingStrategy
)
from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig
from aegis.engine.bgce_engine import BGCEngine, BGCEConfig


def test_vjepa_config():
    """VJEPAConfig creates with correct defaults"""
    config = VJEPAConfig(d_model=768, d_pred=384, mask_ratio=0.75)
    assert config.d_model == 768
    assert config.d_pred == 384
    assert config.mask_ratio == 0.75
    assert config.ema_decay == 0.9998


def test_target_encoder():
    """TargetEncoder holds a frozen EMA copy of the encoder"""
    ssm_config = SSMConfig(d_model=128, d_state=16, n_layers=2)
    encoder = Mamba3MIMO(ssm_config)
    
    target_enc = TargetEncoder(encoder, ema_decay=0.9)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward
    with torch.no_grad():
        output = target_enc(input_ids)
    
    # Verify dimensions
    assert output.shape == (batch_size, seq_len, ssm_config.d_model)
    
    # Verify target encoder is frozen (gradients disabled)
    for param in target_enc.core.parameters():
        assert not param.requires_grad


def test_target_encoder_update():
    """EMA update modifies target encoder weights"""
    ssm_config = SSMConfig(d_model=64, d_state=16, n_layers=1)
    online_encoder = Mamba3MIMO(ssm_config)
    
    target_enc = TargetEncoder(online_encoder, ema_decay=0.9)
    
    # Save initial weights
    initial_weight = list(target_enc.core.parameters())[0].data.clone()
    
    # Modify online encoder
    for param in online_encoder.parameters():
        param.data += 0.1
    
    # Update target encoder
    target_enc.update(online_encoder)
    
    # Verify change
    new_weight = list(target_enc.core.parameters())[0].data
    assert not torch.allclose(initial_weight, new_weight)


def test_predictor():
    """Predictor produces correct output shapes"""
    config = VJEPAConfig(d_model=128, d_pred=64, predictor_depth=2)
    predictor = Predictor(config)
    
    batch_size = 2
    context_len = 10
    num_masks = 5
    
    context = torch.randn(batch_size, context_len, config.d_model)
    mask_indices = torch.randint(0, 100, (batch_size, num_masks))
    
    z_pred, variance = predictor(context, mask_indices, return_variance=True)
    
    # Verify dimensions
    assert z_pred.shape == (batch_size, num_masks, config.d_model)
    assert variance.shape == (batch_size, num_masks, config.d_model)


def test_ebm_energy():
    """EBM energy function produces scalar per sample"""
    ebm = EBM_energy(dim=64, hidden_dim=128)
    
    batch_size = 4
    z_pred = torch.randn(batch_size, 64)
    z_target = torch.randn(batch_size, 64)
    
    energy = ebm(z_pred, z_target)
    
    # Verify dimensions
    assert energy.shape == (batch_size,)
    
    # Verify no NaN
    assert not torch.isnan(energy).any()


def test_masking_strategies():
    """All masking strategies produce correct shapes"""
    seq_len = 20
    mask_ratio = 0.5
    
    # Block mask
    mask_block = MaskingStrategy.block_mask(seq_len, mask_ratio, block_size=4)
    assert mask_block.shape == (seq_len,)
    assert mask_block.sum().item() >= seq_len * mask_ratio * 0.8
    
    # Random mask
    mask_random = MaskingStrategy.random_mask(seq_len, mask_ratio)
    assert mask_random.shape == (seq_len,)
    
    # Causal mask
    mask_causal = MaskingStrategy.causal_mask(seq_len, mask_ratio)
    assert mask_causal.shape == (seq_len,)
    assert mask_causal[-int(seq_len * mask_ratio):].all()


def test_vjepa_forward():
    """VJEPA forward returns dict with expected keys"""
    ssm_config = SSMConfig(d_model=128, d_state=16, n_layers=2)
    encoder = Mamba3MIMO(ssm_config)
    
    vjepa_config = VJEPAConfig(d_model=128, d_pred=64, mask_ratio=0.5)
    vjepa = VJEPA(encoder, vjepa_config)
    
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward
    outputs = vjepa(input_ids)
    
    # Verify expected keys
    assert 'z_pred' in outputs
    assert 'z_target' in outputs
    assert 'variance' in outputs


def test_vjepa_loss():
    """VJEPA compute_loss returns non-negative scalar"""
    ssm_config = SSMConfig(d_model=64, d_state=16, n_layers=1)
    encoder = Mamba3MIMO(ssm_config)
    
    vjepa_config = VJEPAConfig(d_model=64, d_pred=32)
    vjepa = VJEPA(encoder, vjepa_config)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 50, (batch_size, seq_len))
    
    # Forward
    outputs = vjepa(input_ids)
    
    # Compute loss
    loss = vjepa.compute_loss(outputs)
    
    # Verify scalar output
    assert loss.dim() == 0
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_vjepa_train_step():
    """Single VJEPA train_step returns metrics dict"""
    ssm_config = SSMConfig(d_model=64, d_state=16, n_layers=1)
    encoder = Mamba3MIMO(ssm_config)
    
    vjepa_config = VJEPAConfig(d_model=64, d_pred=32)
    vjepa = VJEPA(encoder, vjepa_config)
    
    optimizer = torch.optim.AdamW(
        list(vjepa.online_encoder.parameters()) + list(vjepa.predictor.parameters()),
        lr=1e-4
    )
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 50, (batch_size, seq_len))
    
    # Train step
    metrics = vjepa.train_step(input_ids, optimizer)
    
    # Verify metrics
    assert 'loss' in metrics
    assert 'ema_loss' in metrics
    assert metrics['loss'] >= 0


def test_vjepa_with_bgce_backbone():
    """VJEPA works when BGCEngine (dict-returning) is the backbone [CPU]"""
    bgce_config = BGCEConfig(d_model=64, n_layers=1, vocab_size=5000,
        ssm_config=SSMConfig(d_model=64, d_state=4, d_inner=128, dt_rank=2,
                             use_diagonal_ssm=True, device='cpu'))
    backbone = BGCEngine(bgce_config)
    
    vjepa_config = VJEPAConfig(d_model=64, d_pred=32, mask_ratio=0.5)
    vjepa = VJEPA(backbone, vjepa_config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 5000, (batch_size, seq_len))
    
    outputs = vjepa(input_ids)
    assert 'z_pred' in outputs
    assert 'z_target' in outputs
    assert outputs['z_pred'].shape == (batch_size, seq_len // 2, 64)
    assert outputs['z_target'].shape == (batch_size, seq_len // 2, 64)


if __name__ == '__main__':
    print("Running VJEPA tests...")
    
    test_vjepa_config()
    print("✓ VJEPAConfig")
    
    test_target_encoder()
    print("✓ TargetEncoder")
    
    test_target_encoder_update()
    print("✓ TargetEncoder EMA update")
    
    test_predictor()
    print("✓ Predictor")
    
    test_ebm_energy()
    print("✓ EBM_energy")
    
    test_masking_strategies()
    print("✓ MaskingStrategy")
    
    test_vjepa_forward()
    print("✓ VJEPA forward")
    
    test_vjepa_loss()
    print("✓ VJEPA loss")
    
    test_vjepa_train_step()
    print("✓ VJEPA train_step")
    
    test_vjepa_with_bgce_backbone()
    print("✓ VJEPA + BGCEngine integration")
    
    print("\n✓ All VJEPA tests passed!")
