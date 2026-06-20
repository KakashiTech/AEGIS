"""Tests for AEGIS security module"""
import sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, '.')
from aegis.security.aegis_cyber import (
    AEGISCyberDefense, AEGISCyberConfig,
    FlowPhysicsEncoder, TVDHyperbolicLiquidSSM, TunnelDetector
)


def test_aegis_config():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    assert cfg.d_model == 16
    assert cfg.sequence_length == 8
    assert len(cfg.tunnel_types) == 4


def test_flow_encoder_forward():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    enc = FlowPhysicsEncoder(cfg)
    x = torch.randn(2, 8, 5)
    out = enc.encode_flow(x)
    assert out.shape == (2, 8, 17)  # d_model + 1


def test_flow_encoder_dynamic_features():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    enc = FlowPhysicsEncoder(cfg)
    x3 = torch.randn(2, 8, 3)
    x5 = torch.randn(2, 8, 5)
    o3 = enc.encode_flow(x3)
    o5 = enc.encode_flow(x5)
    assert o3.shape == (2, 8, 17)
    assert o5.shape == (2, 8, 17)
    assert len(enc.feature_encoders) == 2  # one per n_features


def test_tvd_hl_ssm_forward():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    ssm = TVDHyperbolicLiquidSSM(cfg)
    x = torch.randn(2, 8, 17)  # hyperbolic space
    out = ssm(x)
    assert out.shape == (2, 8, 17)


def test_tunnel_detector_forward():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    det = TunnelDetector(cfg)
    x = torch.randn(2, 8, 17)
    is_tunnel, tunnel_type, metrics = det.detect(x)
    assert is_tunnel.shape == (2, 1)
    assert tunnel_type.shape == (2, 4)
    assert 0 <= is_tunnel.min().item() <= is_tunnel.max().item() <= 1


def test_aegis_training_step():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    model = AEGISCyberDefense(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 8, 5)
    y = torch.FloatTensor([[0], [0], [1], [1]])
    fr = model.flow_encoder.encode_flow(x)
    pr = model.tvd_hl_ssm(fr)
    rs, _, _ = model.tunnel_detector.detect(pr)
    loss = F.binary_cross_entropy(rs, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert loss.item() > 0
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    assert params_with_grad > 0, "At least some parameters should receive gradients"


def test_aegis_analyze_traffic():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    model = AEGISCyberDefense(cfg).eval()
    x = torch.randn(2, 8, 5)
    is_mal, ttype, analysis = model.analyze_traffic(x)
    assert is_mal.shape == (2, 1)
    assert 'detection_threshold' in analysis


def test_update_metrics():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    model = AEGISCyberDefense(cfg)
    pred = torch.tensor([[1], [0], [1], [0]])
    gt = torch.tensor([[1], [1], [0], [0]])
    model.update_metrics(pred, gt)
    stats = model.get_detection_stats()
    assert stats['true_positive_rate'] == 0.5
    assert stats['false_positive_rate'] == 0.5
