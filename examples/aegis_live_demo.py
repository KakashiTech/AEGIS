#!/usr/bin/env python3
"""
aegis_live_demo.py — Live demo of AEGIS for anomalous traffic detection.

Captura paquetes de red con scapy, extrae features de flujo,
and classifies each flow as benign or malicious in real time.

Modos:
  --demo     : Generate simulated traffic (no root required)
  --live     : Capture real traffic (requires root)
  --pcap FILE: Analiza un archivo pcap
"""
import sys, time, json, math, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.security.aegis_cyber import AEGISCyberDefense, AEGISCyberConfig

# ═══════════════════════════════════════════════════════════════════════════
# 0. Synthetic data generation (same format as FlowBuffer)
# ═══════════════════════════════════════════════════════════════════════════
def gen_web(n, seq_len=64):
    """Benign web traffic: features = [iat, pkt_len, ttl, sin(flags), cos(flags)]"""
    data = []
    for _ in range(n):
        iat = np.abs(np.random.exponential(0.05, seq_len)) * (1 + np.random.normal(0, 0.1, seq_len))
        pkt = np.where(np.random.rand(seq_len) < 0.7,
                       np.random.randint(40, 100, seq_len),
                       np.random.randint(500, 1500, seq_len))
        ttl = np.random.randint(48, 64, seq_len)
        flags = np.random.choice([0, 1], seq_len, p=[0.8, 0.2])
        data.append(np.column_stack([iat, pkt, ttl, np.sin(flags), np.cos(flags)]))
    return np.array(data, dtype=np.float32)

def gen_attack(n, seq_len=64):
    """Malicious traffic: features = [iat, pkt_len, ttl, sin(flags), cos(flags)]"""
    data = []
    for _ in range(n):
        t = np.linspace(0, 4*np.pi, seq_len)
        iat = np.clip(0.05 + 0.008*np.sin(t) + np.random.normal(0, 0.003, seq_len), 0.001, 1.0)
        pkt = np.random.randint(100, 300, seq_len)
        ttl = np.random.randint(64, 128, seq_len)
        flags = np.random.choice([2, 4, 16], seq_len, p=[0.6, 0.3, 0.1])
        data.append(np.column_stack([iat, pkt, ttl, np.sin(flags), np.cos(flags)]))
    return np.array(data, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Train AEGIS model
# ═══════════════════════════════════════════════════════════════════════════
def train_aegis(d_model=64, seq_len=64, steps=50):
    """Train AEGIS on synthetic data, return trained model + threshold."""
    np.random.seed(42)
    torch.manual_seed(42)

    Xb, Xa = gen_web(500, seq_len), gen_attack(500, seq_len)
    X_all = np.vstack([Xb, Xa])
    y_all = np.array([0]*500 + [1]*500, dtype=np.float32).reshape(-1, 1)

    config = AEGISCyberConfig(d_model=d_model, sequence_length=seq_len)
    model = AEGISCyberDefense(config).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    Xt = torch.FloatTensor(X_all)
    yt = torch.FloatTensor(y_all)

    for step in range(steps):
        idx = np.random.choice(1000, 64, replace=False)
        fr = model.flow_encoder.encode_flow(Xt[idx])
        pr = model.tvd_hl_ssm(fr)
        rs, _, _ = model.tunnel_detector.detect(pr)
        loss = F.binary_cross_entropy(rs, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()

    # Calibrar threshold con ROC
    with torch.no_grad():
        Xv = torch.FloatTensor(np.vstack([gen_web(100, seq_len), gen_attack(100, seq_len)]))
        fr = model.flow_encoder.encode_flow(Xv)
        pr = model.tvd_hl_ssm(fr)
        scores, _, _ = model.tunnel_detector.detect(pr)
        scores = scores.squeeze().numpy()
    yv = np.array([0]*100 + [1]*100, dtype=np.float32)
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(yv, scores)
    youden = tpr - fpr
    threshold = th[np.argmax(youden)]

    print(f"AEGIS model trained: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"ROC-calibrated threshold: {threshold:.3f}")
    return model, config, threshold

# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature extraction
# ═══════════════════════════════════════════════════════════════════════════
class FlowBuffer:
    """Buffer circular para construir ventanas de flujo desde paquetes."""
    def __init__(self, window_size=64):
        self.window_size = window_size
        self.packets = []
        self.last_time = None

    def add_packet(self, pkt_len, timestamp, ttl=64, flags=0):
        """Add packet and return window if full, None otherwise."""
        if self.last_time is not None:
            iat = timestamp - self.last_time
        else:
            iat = 0.05  # default
        self.last_time = timestamp
        self.packets.append([iat, pkt_len, ttl, math.sin(flags), math.cos(flags)])

        if len(self.packets) >= self.window_size:
            window = np.array(self.packets[-self.window_size:], dtype=np.float32)
            return window[np.newaxis, :, :]  # (1, W, 5)
        return None

    def clear(self):
        self.packets = []
        self.last_time = None

# ═══════════════════════════════════════════════════════════════════════════
# 3. Demo mode: synthetic traffic simulation
# ═══════════════════════════════════════════════════════════════════════════
def demo_synthetic(model, config, threshold, duration_seconds=30):
    """Generate synthetic traffic and classify live."""
    print(f"\n{'='*60}")
    print(f"DEMO MODE: Generating synthetic traffic... (threshold={threshold:.3f})")
    print(f"{'='*60}")
    print(f"{'Time':>8} {'Source':<12} {'Score':>8} {'Prediction':<12} {'Confidence':>8}")
    print("-" * 52)

    buffer = FlowBuffer(config.sequence_length)
    t = 0.0
    benign_traffic = True
    switch_interval = 5.0

    # Pre-generate series para ataque (necesita t para sin(t))
    attack_start_t = 0.0

    while t < duration_seconds:
        if benign_traffic:
            iat = abs(np.random.exponential(0.05)) * (1 + np.random.normal(0, 0.1))
            pkt_len = np.random.randint(40, 100) if np.random.rand() < 0.7 else np.random.randint(500, 1500)
            ttl = np.random.randint(48, 64)
            flags = 0 if np.random.rand() < 0.8 else 1
            source = "web"
        else:
            local_t = t - attack_start_t
            iat = 0.05 + 0.008 * math.sin(local_t * 2) + np.random.normal(0, 0.003)
            iat = max(0.001, iat)
            pkt_len = np.random.randint(100, 300)
            ttl = np.random.randint(64, 128)
            r = np.random.rand()
            flags = 2 if r < 0.6 else (4 if r < 0.9 else 16)
            source = "C2"

        t += abs(iat)
        window = buffer.add_packet(pkt_len, t, ttl, flags)

        if window is not None:
            with torch.no_grad():
                fr = model.flow_encoder.encode_flow(torch.FloatTensor(window))
                pr = model.tvd_hl_ssm(fr)
                score, _, _ = model.tunnel_detector.detect(pr)
                s = score.item()
                pred = "⚠️ MALICIOUS" if s > threshold else "✓ benign"
                conf = s if s > threshold else (1 - s)
                print(f"{t:>7.2f}s {source:<12} {s:>7.3f}  {pred:<12} {conf:>7.1%}")

        if t > switch_interval:
            benign_traffic = not benign_traffic
            switch_interval += 5.0
            attack_start_t = t if not benign_traffic else attack_start_t
            buffer.clear()
            print("-" * 52)

        time.sleep(0.05)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Live capture mode (requires root)
# ═══════════════════════════════════════════════════════════════════════════
def live_capture(model, config, threshold, interface=None, duration=60):
    """Captura paquetes reales y los clasifica."""
    try:
        from scapy.all import sniff, IP, TCP, UDP
    except ImportError:
        print("ERROR: scapy no instalado. Usa --demo o --pcap.")
        return

    if interface:
        print(f"Escuchando en {interface}...")
    else:
        print("Escuchando en todas las interfaces...")

    buffer = FlowBuffer(config.sequence_length)
    flows = defaultdict(lambda: FlowBuffer(config.sequence_length))
    packet_count = 0
    start_time = time.time()

    def process_packet(pkt):
        nonlocal packet_count
        packet_count += 1
        now = time.time() - start_time

        if IP in pkt:
            src = pkt[IP].src
            pkt_len = len(pkt)
            ttl = pkt[IP].ttl
            flags = 0
            if TCP in pkt:
                flags = pkt[TCP].flags
            elif UDP in pkt:
                flags = 0

            window = flows[src].add_packet(pkt_len, now, ttl, flags)
            if window is not None:
                with torch.no_grad():
                    fr = model.flow_encoder.encode_flow(torch.FloatTensor(window))
                    pr = model.tvd_hl_ssm(fr)
                    score, ttype, _ = model.tunnel_detector.detect(pr)
                    s = score.item()
                    pred = "⚠️ ANOMALY" if s > threshold else "✓ benign"
                    tunnel_name = config.tunnel_types[ttype.argmax().item()]
                    print(f"[{now:>6.1f}s] {src:<18} score={s:.3f} {pred} ({tunnel_name})")

        if time.time() - start_time > duration:
            return True  # stop sniffing

    try:
        sniff(prn=process_packet, store=0, timeout=duration)
    except PermissionError:
        print("ERROR: Se necesita root para captura en vivo.")
        print("Usa: sudo python examples/aegis_live_demo.py --live")
        return

    print(f"\nPaquetes capturados: {packet_count}")
    print(f"Flujos monitoreados: {len(flows)}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PCAP analysis mode
# ═══════════════════════════════════════════════════════════════════════════
def analyze_pcap(model, config, threshold, pcap_path):
    """Analiza un archivo pcap guardado."""
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        print("ERROR: scapy no instalado.")
        return

    print(f"Analizando {pcap_path}...")
    packets = rdpcap(pcap_path)
    print(f"Paquetes: {len(packets)}")

    buffer = FlowBuffer(config.sequence_length)
    results = []

    start_time = packets[0].time if packets else 0
    for pkt in packets:
        if IP not in pkt:
            continue
        now = pkt.time - start_time
        pkt_len = len(pkt)
        ttl = pkt[IP].ttl
        flags = 0
        if TCP in pkt:
            flags = pkt[TCP].flags

        window = buffer.add_packet(pkt_len, now, ttl, flags)
        if window is not None:
            with torch.no_grad():
                fr = model.flow_encoder.encode_flow(torch.FloatTensor(window))
                pr = model.tvd_hl_ssm(fr)
                score, _, _ = model.tunnel_detector.detect(pr)
                results.append((now, score.item()))

    # Summary
    scores = [s for _, s in results]
    if scores:
        print(f"\nVentanas analizadas: {len(scores)}")
        print(f"Score promedio: {np.mean(scores):.4f}")
        print(f"Max score: {np.max(scores):.4f}")
        pct_anomalous = np.mean([1 for s in scores if s > threshold])
        print(f"Anomalous traffic: {pct_anomalous:.1%}")
    else:
        print("No se pudieron formar ventanas de flujo.")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS Live Traffic Detection")
    parser.add_argument("--demo", action="store_true", help="Demo with synthetic traffic")
    parser.add_argument("--live", action="store_true", help="Captura en vivo (requiere root)")
    parser.add_argument("--pcap", type=str, help="Analizar archivo pcap")
    parser.add_argument("--interface", type=str, default=None, help="Interface de red")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    args = parser.parse_args()

    # Train model
    print("Inicializando AEGIS...")
    model, config, threshold = train_aegis()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    if args.demo:
        demo_synthetic(model, config, threshold, duration_seconds=args.duration)
    elif args.live:
        live_capture(model, config, threshold, interface=args.interface, duration=args.duration)
    elif args.pcap:
        analyze_pcap(model, config, threshold, args.pcap)
    else:
        # Default: run synthetic demo for 10 seconds
        print("Default mode: synthetic demo. Use --live or --pcap for real data.")
        demo_synthetic(model, config, threshold, duration_seconds=10)
