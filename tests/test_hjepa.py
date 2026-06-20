#!/usr/bin/env python3
"""
Tests for H-JEPA (Hierarchical Joint-Embedding Predictive Architecture)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from aegis.learning.hjepa import (
    HJEPAConfig, HierarchicalLevel, CausalTimePrior,
    MentalRolloutSimulator, HJEPA
)


def test_hjepa_config():
    config = HJEPAConfig(d_model=64, n_hierarchical_levels=2)
    assert config.d_model == 64
    assert config.n_hierarchical_levels == 2
    assert config.temporal_horizon is None


def test_hierarchical_level_encode():
    level = HierarchicalLevel(level_id=0, d_model=64, temporal_scale=2)
    x = torch.randn(2, 16, 64)
    encoded = level.encode(x)
    # temporal_scale=2 subsamples: 16 → 8
    assert encoded.shape == (2, 8, 64)
    assert not torch.isnan(encoded).any()


def test_hierarchical_level_predict():
    level = HierarchicalLevel(level_id=0, d_model=64, temporal_scale=2)
    x = torch.randn(2, 8, 64)
    encoded = level.encode(x)
    preds = level.predict_future(encoded, n_steps=3)
    assert len(preds) == 3
    assert preds[0].shape == (2, 1, 64)


def test_causal_time_prior():
    ctp = CausalTimePrior(state_dim=16, action_dim=4, n_causal_vars=4)
    state = torch.randn(2, 16)
    actions = torch.randn(2, 5, 4)
    traj = ctp.generate_causal_trajectory(state, actions, n_steps=5)
    assert traj.shape == (2, 5, 16)


def test_hjepa_forward():
    config = HJEPAConfig(d_model=32, n_hierarchical_levels=2, state_dim=16, action_dim=4)
    model = HJEPA(config)
    obs = torch.randn(2, 16, 32)
    actions = torch.randn(2, 16, 4)
    out = model(obs, actions=actions)
    assert isinstance(out, dict)
    assert 'level_representations' in out
    assert len(out['level_representations']) == 2


def test_hjepa_hierarchical_encode():
    config = HJEPAConfig(d_model=32, n_hierarchical_levels=2, state_dim=16, action_dim=4)
    model = HJEPA(config)
    x = torch.randn(2, 16, 32)
    encoded = model.hierarchical_encode(x)
    assert len(encoded) == 2


def test_hjepa_zero_shot_control():
    config = HJEPAConfig(d_model=32, n_hierarchical_levels=2, state_dim=16, action_dim=4)
    model = HJEPA(config)
    target = torch.randn(16)
    obs = torch.randn(1, 8, 32)
    action = model.zero_shot_control(target, obs)
    assert action.shape == (1, 4)


def test_hjepa_get_stats():
    config = HJEPAConfig(d_model=32, n_hierarchical_levels=2, state_dim=16, action_dim=4)
    model = HJEPA(config)
    stats = model.get_stats()
    assert isinstance(stats, dict)


if __name__ == '__main__':
    test_hjepa_config()
    print("✓ HJEPAConfig")
    test_hierarchical_level_encode()
    print("✓ HierarchicalLevel.encode")
    test_hierarchical_level_predict()
    print("✓ HierarchicalLevel.predict_future")
    test_causal_time_prior()
    print("✓ CausalTimePrior")
    test_hjepa_forward()
    print("✓ HJEPA forward")
    test_hjepa_hierarchical_encode()
    print("✓ HJEPA hierarchical_encode")
    test_hjepa_zero_shot_control()
    print("✓ HJEPA zero_shot_control")
    test_hjepa_get_stats()
    print("✓ HJEPA get_stats")
    print("\n✓ All H-JEPA tests passed!")
