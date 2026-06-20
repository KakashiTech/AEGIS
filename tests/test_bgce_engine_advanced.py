"""Additional tests for BGCEngine: VJEPA, liquid neurons, Lorentz, inference"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from aegis.engine.bgce_engine import BGCEngine, BGCEConfig, ContinualLiquidNeurons
from aegis.core.mamba3_mimo import SSMConfig
from aegis.learning.vjepa import VJEPAConfig


def _make_vjepa_cfg(d_model=32):
    return VJEPAConfig(d_model=d_model, d_pred=d_model//2, predictor_depth=2,
                       mask_ratio=0.5, input_dim=d_model, n_causal_features=d_model)


def _make_cfg(use_vjepa=False):
    vjepa_cfg = _make_vjepa_cfg(32) if use_vjepa else None
    return BGCEConfig(
        d_model=32, n_layers=1, vocab_size=100,
        use_vjepa=use_vjepa,
        vjepa_config=vjepa_cfg if use_vjepa else BGCEConfig(
            d_model=32, n_layers=1, vocab_size=100,
            ssm_config=SSMConfig(d_model=32, d_state=4, d_inner=64, n_layers=1)
        ).vjepa_config,
        ssm_config=SSMConfig(d_model=32, d_state=4, d_inner=64, n_layers=1)
    )


def test_bgce_vjepa_integration():
    cfg = _make_cfg(use_vjepa=True)
    model = BGCEngine(cfg)
    assert hasattr(model, 'vjepa')
    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert 'logits' in out
    assert out['logits'].shape == (2, 16, 100)


def test_bgce_liquid_rk4():
    cfg = _make_cfg(use_vjepa=False)
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert not torch.isnan(out['logits']).any()


def test_liquid_neuron_rk4_euler_comparison():
    dim = 32
    rk4 = ContinualLiquidNeurons(dim)
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
    for dt in [0.5, 0.2, 0.1]:
        x_rk = rk4(x.clone(), dt=dt)
        x_eu = euler(x.clone(), dt=dt)
        assert (x_rk - x_eu).norm().item() > 1e-6


def test_bgce_lorentz_projection():
    cfg = BGCEConfig(d_model=32, n_layers=1, vocab_size=100, use_lorentz=True,
                     ssm_config=SSMConfig(d_model=32, d_state=4, d_inner=64, n_layers=1))
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert 'logits' in out
    assert out['logits'].shape == (2, 16, 100)


def test_bgce_generate_no_vjepa():
    cfg = _make_cfg(use_vjepa=False)
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 13)
    assert out.dtype == torch.long


def test_bgce_gradient_lm_head():
    cfg = _make_cfg(use_vjepa=False)
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (4, 16))
    out = model(x)
    loss = F.cross_entropy(out['logits'].transpose(1, 2), x)
    loss.backward()
    lm_grads = [p.grad for n, p in model.lm_head.named_parameters() if p.grad is not None]
    backbone_grads = [p.grad for n, p in model.backbone.named_parameters() if p.grad is not None]
    assert len(lm_grads) > 0
    assert len(backbone_grads) > 0
