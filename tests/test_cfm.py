#!/usr/bin/env python3
"""
Tests for CFM (Causal Foundation Model)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from aegis.causality.cfm import (
    CFMConfig, PartialCausalGraph, StructuralEquation,
    CausalAttention, ATEEstimator, CausalFoundationModel
)


def test_cfm_config():
    config = CFMConfig(d_model=64, n_causal_vars=8)
    assert config.d_model == 64
    assert config.n_causal_vars == 8


def test_structural_equation():
    eq = StructuralEquation(d_model=64, max_parents=5)
    parents = torch.randn(2, 5, 64)
    out = eq(parents)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_partial_causal_graph_forward():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    pcg = PartialCausalGraph(config)
    vars = torch.randn(2, 4, 64)
    out = pcg(vars)
    assert out.shape == (2, 4, 64)
    assert not torch.isnan(out).any()


def test_partial_causal_graph_intervention():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    pcg = PartialCausalGraph(config)
    vars = torch.randn(2, 4, 64)
    out = pcg(vars, intervention_idx=1, intervention_value=0.5)
    assert out.shape == (2, 4, 64)


def test_causal_attention():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    attn = CausalAttention(config)
    x = torch.randn(2, 4, 64)
    out = attn(x, x, x)
    assert out.shape == (2, 4, 64)


def test_ate_estimator():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    estimator = ATEEstimator(config)
    vars = torch.randn(2, 4, 64)
    out = estimator(vars)
    assert out.shape == (2, 1)


def test_causal_foundation_model_forward():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    model = CausalFoundationModel(config)
    obs = torch.randn(2, 4, 64)
    out = model(obs)
    assert isinstance(out, dict)
    assert 'causal_graph' in out


def test_causal_foundation_model_intervene():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    model = CausalFoundationModel(config)
    obs = torch.randn(2, 4, 64)
    out = model.intervene(obs, {1: 0.5})
    assert out.shape == (2, 4, 64)


def test_causal_foundation_model_ate():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    model = CausalFoundationModel(config)
    obs = torch.randn(2, 4, 64)
    out = model(obs, treatment_idx=0, outcome_idx=1, return_ate=True)
    assert isinstance(out, dict)


def test_cfm_gradient_flow():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    model = CausalFoundationModel(config)
    obs = torch.randn(2, 4, 64)
    out = model(obs, treatment_idx=0, outcome_idx=1)
    loss = out['ate'].sum() if 'ate' in out else sum(v.sum() for v in out.values() if isinstance(v, torch.Tensor))
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_cfm_multiple_interventions():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    model = CausalFoundationModel(config)
    obs = torch.randn(2, 4, 64)
    out = model.intervene(obs, {0: 0.5, 2: -0.3})
    assert out.shape == (2, 4, 64)


def test_partial_causal_graph_no_parents():
    config = CFMConfig(d_model=64, n_causal_vars=2)
    config.causal_graph = [[], []]  # No edges
    pcg = PartialCausalGraph(config)
    vars = torch.randn(2, 2, 64)
    out = pcg(vars)
    assert out.shape == (2, 2, 64)


def test_ate_estimator_gradient():
    config = CFMConfig(d_model=64, n_causal_vars=4)
    estimator = ATEEstimator(config)
    vars = torch.randn(2, 4, 64)
    out = estimator(vars)
    loss = out.sum()
    loss.backward()
    # Check at least one parameter got gradient
    has_grad = any(p.grad is not None and p.grad.norm() > 0
                    for p in estimator.parameters())
    assert has_grad


if __name__ == '__main__':
    test_cfm_config()
    print("✓ CFMConfig")
    test_structural_equation()
    print("✓ StructuralEquation")
    test_partial_causal_graph_forward()
    print("✓ PartialCausalGraph forward")
    test_partial_causal_graph_intervention()
    print("✓ PartialCausalGraph intervention")
    test_causal_attention()
    print("✓ CausalAttention")
    test_ate_estimator()
    print("✓ ATEEstimator")
    test_causal_foundation_model_forward()
    print("✓ CausalFoundationModel forward")
    test_causal_foundation_model_intervene()
    print("✓ CausalFoundationModel intervene")
    test_causal_foundation_model_ate()
    print("✓ CausalFoundationModel ATE")
    print("\n✓ All CFM tests passed!")
