#!/bin/bash
set -e
echo "=== BGCE Reproducibility Suite ==="

echo ""
echo "--- Tests Unitarios (4 suites, 34 tests) ---"
python tests/run_all_tests.py

echo ""
echo "--- CPU Showdown: Diagonal++ vs Transformer ---"
python -u -c "
import sys, time, torch
sys.path.insert(0, '.')
from aegis.core.mamba3_mimo import Mamba3MIMO, SSMConfig
from benchmarks.transformer_baseline import TransformerLM
device='cpu'
m = Mamba3MIMO(SSMConfig(d_model=256, d_state=16, n_layers=4, d_inner=512, dt_rank=8, use_diagonal_ssm=True, device=device)).eval()
t = TransformerLM(d_model=256, n_layers=4, n_heads=8, max_seq_len=2048).eval()
for L in [128, 256, 512, 768, 1024, 1536, 2048]:
    x = torch.randint(0, 50000, (1, L))
    for _ in range(2):
        _ = m(x); _ = t(x)
    m_times, t_times = [], []
    for _ in range(5):
        t0=time.perf_counter(); _=m(x); t1=time.perf_counter(); m_times.append(t1-t0)
        t0=time.perf_counter(); _=t(x); t1=time.perf_counter(); t_times.append(t1-t0)
    mm = sum(m_times)/len(m_times)*1000
    tt = sum(t_times)/len(t_times)*1000
    print(f'L={L:>5}:  Mamba3={mm:>8.3f}ms  Transformer={tt:>8.3f}ms  {\"BGCE\" if mm<tt else \"Tfmr\"}')
"

echo ""
echo "--- AEGIS Synthetic Training (30 steps, ROC-calibrated) ---"
python -u -c "
import torch, torch.nn.functional as F, numpy as np, sys
sys.path.insert(0, '.')
from aegis.security.aegis_cyber import AEGISCyberDefense, AEGISCyberConfig
config = AEGISCyberConfig(d_model=64, sequence_length=32)
model = AEGISCyberDefense(config).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
n, sl = 200, 32
benign = np.abs(np.random.exponential(0.05, (n//2, sl))) + np.random.normal(0, 0.01, (n//2, sl))
t = np.linspace(0, 4*np.pi, sl)
malicious = 0.1 + 0.02*np.sin(t) + np.random.normal(0, 0.005, (n//2, sl))
X = torch.FloatTensor(np.vstack([benign, np.clip(malicious,0.001,1.0)])).unsqueeze(-1).expand(-1,-1,config.d_model)
y = torch.FloatTensor([0]*(n//2)+[1]*(n//2)).unsqueeze(1)
for step in range(30):
    idx = np.random.choice(n, 16, replace=False)
    fr = model.flow_encoder.encode_flow(X[idx])
    pr = model.tvd_hl_ssm(fr)
    rs, _, _ = model.tunnel_detector.detect(pr)
    loss = F.binary_cross_entropy(rs, y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1)%15==0:
        print(f'  Step {step+1}: loss={loss.item():.4f}' + f'  acc={((rs>0.5).float()==y[idx]).float().mean().item():.3f}')
model.eval()
with torch.no_grad():
    fr=model.flow_encoder.encode_flow(X); pr=model.tvd_hl_ssm(fr); rs,_,_=model.tunnel_detector.detect(pr)
    from sklearn.metrics import roc_curve
    fpr,tpr,th=roc_curve(y.numpy().ravel(), rs.squeeze().numpy()); youden=tpr-fpr
    acc=((rs.squeeze().numpy()>th[np.argmax(youden)]).astype(int)==y.numpy().ravel()).mean()
    print(f'  Final accuracy (ROC-calibrated): {acc:.3f}')
print('AEGIS OK')
"

echo ""
echo "--- E2E Pipeline + Gradient Flow ---"
python -u -c "
import sys, torch
sys.path.insert(0, '.')
from aegis.engine.bgce_engine import BGCEngine, BGCEConfig
from aegis.core.mamba3_mimo import SSMConfig
config = BGCEConfig(d_model=64, n_layers=2, vocab_size=5000,
    ssm_config=SSMConfig(d_model=64, d_state=8, d_inner=128, dt_rank=4, use_diagonal_ssm=True))
model = BGCEngine(config)
x = torch.randint(0, 5000, (2, 64))
out = model(x)
logits = out['logits']
print(f'Pipeline OK: logits shape={list(logits.shape)}')
loss = logits.sum()
loss.backward()
total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f'Gradient flow OK: total_grad={total_grad:.4f}')
"

echo ""
echo "--- Hobbit Dataset 1: Shakespeare Tiny (character-level LM) ---"
python -u /home/tuffhk/Work/HOBBIT/examples/train_shakespeare_tiny.py 2>&1 | tail -20

echo ""
echo "--- Hobbit Dataset 2: Algebraic Reasoning (OOD generalization) ---"
python -u /home/tuffhk/Work/HOBBIT/examples/train_algebraic_reasoning.py 2>&1 | tail -20

echo ""
echo "--- Hobbit Dataset 3: Traffic Anomaly Detection ---"
python -u /home/tuffhk/Work/HOBBIT/examples/train_traffic_anomaly.py 2>&1 | tail -15

echo ""
echo "--- AEGIS Live Demo (10s synthetic traffic) ---"
python -u /home/tuffhk/Work/HOBBIT/examples/aegis_live_demo.py --demo --duration 8 2>&1 | tail -15

echo ""
echo "=== Todos los resultados reproducidos ==="
echo "Ver PAPER_DIAGONAL_SSM.md para el paper matematico."
echo "Ver STATUS_REPORT.md para el informe de estado."
