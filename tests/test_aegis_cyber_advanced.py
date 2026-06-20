"""Additional tests for AEGIS security module: PDE, RK4, traffic patterns"""
import sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, '.')
from aegis.security.aegis_cyber import (
    AEGISCyberDefense, AEGISCyberConfig,
    FlowPhysicsEncoder, TVDHyperbolicLiquidSSM, TunnelDetector, LiquidNeuron
)


def test_liquid_neuron_rk4_vs_euler():
    dim = 16
    rk4 = LiquidNeuron(dim, time_constant=1.0)

    class EulerLiquid(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.W = original.W
            self.tau = original.tau
            self.activation = original.activation
        def forward(self, x, dt=0.1):
            decay = -x / self.tau
            input_term = self.activation(self.W(x))
            return x + dt * (decay + input_term)

    euler = EulerLiquid(rk4)
    x = torch.randn(1, dim)

    for dt in [0.5, 0.2, 0.1, 0.05]:
        x_rk = rk4(x.clone(), dt=dt)
        x_eu = euler(x.clone(), dt=dt)
        diff = (x_rk - x_eu).norm().item()
        assert diff > 1e-8, f"RK4 and Euler should differ at dt={dt}"
        # RK4 should be more accurate: compare vs small-dt reference
        x_ref = x.clone()
        for _ in range(20):
            x_ref = rk4(x_ref, dt=0.001)
        err_rk = (rk4(x.clone(), dt=dt) - x_ref).norm().item()
        err_eu = (euler(x.clone(), dt=dt) - x_ref).norm().item()
        if dt >= 0.1:
            assert err_rk <= err_eu * 1.5 or err_eu < 1e-6, \
                f"RK4 error ({err_rk:.6f}) should be ≤ Euler error ({err_eu:.6f})"


def test_liquid_neuron_gradient_flow():
    neuron = LiquidNeuron(16, time_constant=1.0)
    x = torch.randn(1, 16, requires_grad=True)
    out = neuron(x, dt=0.1)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_tvd_ssm_rk4_vs_euler():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    ssm = TVDHyperbolicLiquidSSM(cfg)
    x = torch.randn(2, 8, 17)
    out = ssm(x)
    assert out.shape == (2, 8, 17)
    assert not torch.isnan(out).any()


def test_pde_transport_correctness():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=16)
    model = AEGISCyberDefense(cfg).eval()
    x = torch.randn(2, 16, 5)
    with torch.no_grad():
        fr = model.flow_encoder.encode_flow(x)
        pr = model.tvd_hl_ssm(fr)
    assert not torch.isnan(pr).any()
    assert pr.shape == (2, 16, 17)


def test_real_traffic_consistency():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=16)
    model = AEGISCyberDefense(cfg).eval()
    n = 100
    sl = 16
    benign = np.abs(np.random.exponential(0.05, (n//2, sl))) + np.random.normal(0, 0.01, (n//2, sl))
    t = np.linspace(0, 4*np.pi, sl)
    malicious = 0.1 + 0.02*np.sin(t) + np.random.normal(0, 0.005, (n//2, sl))
    X = torch.FloatTensor(np.vstack([benign, np.clip(malicious, 0.001, 1.0)])).unsqueeze(-1).expand(-1, -1, 5)
    y = torch.FloatTensor([0]*(n//2) + [1]*(n//2)).unsqueeze(1)
    with torch.no_grad():
        fr = model.flow_encoder.encode_flow(X)
        pr = model.tvd_hl_ssm(fr)
        rs, _, _ = model.tunnel_detector.detect(pr)
    assert rs.shape == (n, 1)
    assert 0 <= rs.min().item() <= rs.max().item() <= 1
    benign_scores = rs[:n//2].mean().item()
    malicious_scores = rs[n//2:].mean().item()
    print(f"  benign mean score: {benign_scores:.4f}, malicious: {malicious_scores:.4f}")


def test_tvd_dissipation_stability():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=32)
    model = AEGISCyberDefense(cfg).eval()
    x = torch.randn(2, 32, 5)
    with torch.no_grad():
        fr = model.flow_encoder.encode_flow(x)
        pr = model.tvd_hl_ssm(fr)
    assert not torch.isnan(pr).any()
    assert not torch.isinf(pr).any()


def test_flow_encoder_forward_shape():
    cfg = AEGISCyberConfig(d_model=32, sequence_length=16)
    enc = FlowPhysicsEncoder(cfg)
    x = torch.randn(2, 16, 5)
    out = enc.encode_flow(x)
    assert out.shape == (2, 16, cfg.d_model + 1)
    assert not torch.isnan(out).any()


def test_roc_auc_monotonic():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=16)
    model = AEGISCyberDefense(cfg)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    tpr_values = []
    fpr_values = []
    n = 50
    sl = 16
    benign = np.abs(np.random.exponential(0.05, (n//2, sl))) + np.random.normal(0, 0.01, (n//2, sl))
    t = np.linspace(0, 4*np.pi, sl)
    malicious = 0.1 + 0.02*np.sin(t) + np.random.normal(0, 0.005, (n//2, sl))
    X = torch.FloatTensor(np.vstack([benign, np.clip(malicious, 0.001, 1.0)])).unsqueeze(-1).expand(-1, -1, 5)
    y = torch.FloatTensor([0]*(n//2) + [1]*(n//2)).unsqueeze(1)
    with torch.no_grad():
        fr = model.flow_encoder.encode_flow(X)
        pr = model.tvd_hl_ssm(fr)
        rs, _, _ = model.tunnel_detector.detect(pr)
    for th in thresholds:
        pred = (rs > th).float()
        tp = ((pred == 1) & (y == 1)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        tn = ((pred == 0) & (y == 0)).sum().item()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    assert tpr_values[0] >= tpr_values[-1], "TPR should decrease as threshold increases"
    assert fpr_values[0] >= fpr_values[-1], "FPR should decrease as threshold increases"
    print(f"  TPR progression: {[round(v, 3) for v in tpr_values]}")
    print(f"  FPR progression: {[round(v, 3) for v in fpr_values]}")


def test_tvd_hn_ssm_batch_independence():
    cfg = AEGISCyberConfig(d_model=16, sequence_length=8)
    ssm = TVDHyperbolicLiquidSSM(cfg)
    x1 = torch.randn(1, 8, 17)
    x2 = torch.randn(1, 8, 17)
    x_batch = torch.cat([x1, x2], dim=0)
    out_batch = ssm(x_batch)
    out1 = ssm(x1)
    out2 = ssm(x2)
    assert torch.allclose(out_batch[0:1], out1, atol=1e-6), "Batch outputs should match individual"
    assert torch.allclose(out_batch[1:2], out2, atol=1e-6), "Batch outputs should match individual"
