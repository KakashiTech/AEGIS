import sys, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from aegis.engine.bgce_engine import BGCEngine, BGCEConfig
from aegis.core.mamba3_mimo import SSMConfig


def _make_config():
    return BGCEConfig(d_model=32, n_layers=1, vocab_size=100,
                      ssm_config=SSMConfig(d_model=32, d_state=4, d_inner=64, n_layers=1))


def test_bgce_config():
    cfg = _make_config()
    assert cfg.d_model == 32
    assert cfg.n_layers == 1
    assert cfg.vocab_size == 100


def test_bgce_forward():
    cfg = _make_config()
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert 'logits' in out
    assert out['logits'].shape == (2, 16, 100)


def test_bgce_gradient_flow():
    cfg = _make_config()
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    loss = out['logits'].sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "Gradient should flow through all parameters"


def test_bgce_empty_batch():
    cfg = _make_config()
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (0, 16))
    out = model(x)
    assert out['logits'].shape[0] == 0


def test_bgce_generate():
    cfg = _make_config()
    model = BGCEngine(cfg)
    model.eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 13)
    assert out.dtype == torch.long


def test_bgce_generate_greedy():
    cfg = _make_config()
    model = BGCEngine(cfg)
    model.eval()
    x = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=3, temperature=1e-8)
    assert out.shape[0] == 1
    assert out.shape[1] >= 8


def test_bgce_hidden_states():
    cfg = _make_config()
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = model(x, return_hidden=True)
    assert 'hidden_states' in out
    assert out['hidden_states'].shape == (2, 16, 32)


def test_bgce_loss():
    cfg = _make_config()
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    logits = out['logits']
    loss = F.cross_entropy(logits.transpose(1, 2), x)
    assert loss > 0
    assert not torch.isnan(loss)


def test_bgce_train_step():
    cfg = _make_config()
    model = BGCEngine(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 100, (4, 16))
    out = model(x)
    loss = F.cross_entropy(out['logits'].transpose(1, 2), x)
    loss.backward()
    opt.step()
    opt.zero_grad()
    out2 = model(x)
    loss2 = F.cross_entropy(out2['logits'].transpose(1, 2), x)
    assert not torch.isnan(loss2)


def test_bgce_ssm_diagonal_flag():
    cfg = BGCEConfig(d_model=32, n_layers=1, vocab_size=100,
                     ssm_config=SSMConfig(d_model=32, d_state=4, d_inner=64,
                                          n_layers=1, use_diagonal_ssm=True))
    model = BGCEngine(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert out['logits'].shape == (2, 16, 100)
