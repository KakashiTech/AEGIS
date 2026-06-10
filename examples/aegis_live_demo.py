#!/usr/bin/env python3
"""
aegis_live_demo.py — Demo en vivo de AEGIS para detección de tráfico anómalo.

Captura paquetes de red con scapy, extrae features de flujo,
y clasifica cada flujo como benigno o malicioso en tiempo real.

Modos:
  --demo     : Genera tráfico simulado (no requiere root)
  --live     : Captura tráfico real (requiere root)
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
# 1. Entrenar modelo AEGIS
# ═══════════════════════════════════════════════════════════════════════════
def train_aegis(d_model=64, seq_len=64, steps=50):
    """Entrena AEGIS en datos sintéticos y retorna modelo entrenado."""
    np.random.seed(42)
    torch.manual_seed(42)

    def gen_web(n):
        data = []
        for _ in range(n):
            iat = np.abs(np.random.exponential(0.03, seq_len)) + np.random.normal(0, 0.005, seq_len)
            pkt = np.where(np.random.rand(seq_len) < 0.7, np.random.randint(40, 100, seq_len), np.random.randint(500, 1500, seq_len))
            ttl = np.random.randint(48, 64, seq_len)
            data.append(np.column_stack([iat, pkt, ttl, np.random.rand(seq_len), np.random.rand(seq_len)]))
        return np.array(data, dtype=np.float32)

    def gen_attack(n):
        data = []
        for _ in range(n):
            t = np.linspace(0, 8*np.pi, seq_len)
            iat = np.clip(0.05 + 0.01*np.sin(t) + np.random.normal(0, 0.002, seq_len), 0.001, 1.0)
            pkt = np.random.randint(100, 300, seq_len)
            ttl = np.random.randint(64, 128, seq_len)
            data.append(np.column_stack([iat, pkt, ttl, np.sin(t*0.5), np.cos(t*0.5)]))
        return np.array(data, dtype=np.float32)

    Xb, Xa = gen_web(200), gen_attack(200)
    X_all = np.vstack([Xb, Xa])
    y_all = np.array([0]*200 + [1]*200, dtype=np.float32).reshape(-1, 1)

    config = AEGISCyberConfig(d_model=d_model, sequence_length=seq_len)
    model = AEGISCyberDefense(config).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    Xt = torch.FloatTensor(X_all)
    yt = torch.FloatTensor(y_all)

    for step in range(steps):
        idx = np.random.choice(400, 16, replace=False)
        fr = model.flow_encoder.encode_flow(Xt[idx])
        pr = model.tvd_hl_ssm(fr)
        rs, _, _ = model.tunnel_detector.detect(pr)
        loss = F.binary_cross_entropy(rs, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    print(f"Modelo AEGIS entrenado: {sum(p.numel() for p in model.parameters()):,} params")
    return model, config

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
        """Añade un paquete y retorna ventana si está llena, None si no."""
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
def demo_synthetic(model, config, duration_seconds=30):
    """Genera tráfico sintético y lo clasifica en vivo."""
    print(f"\n{'='*60}")
    print("MODO DEMO: Generando tráfico sintético...")
    print(f"{'='*60}")
    print(f"{'Tiempo':>8} {'Origen':<12} {'Score':>8} {'Predicción':<12} {'Confianza':>8}")
    print("-" * 52)

    buffer = FlowBuffer(config.sequence_length)
    t = 0.0
    benign_traffic = True
    switch_interval = 5.0  # switch every 5 seconds

    while t < duration_seconds:
        # Generate synthetic packet
        if benign_traffic:
            iat = np.abs(np.random.exponential(0.05)) * (1 + np.random.normal(0, 0.1))
            pkt_len = np.random.randint(40, 1500)
            ttl = np.random.randint(48, 64)
            flags = 0
            source = "web"
        else:
            iat = 0.05 + 0.01 * math.sin(t * 2) + np.random.normal(0, 0.002)
            iat = max(0.001, iat)
            pkt_len = np.random.randint(100, 300)
            ttl = np.random.randint(64, 128)
            flags = 2  # SYN flag emulation
            source = "C2"

        t += abs(iat)
        window = buffer.add_packet(pkt_len, t, ttl, flags)

        if window is not None:
            with torch.no_grad():
                fr = model.flow_encoder.encode_flow(torch.FloatTensor(window))
                pr = model.tvd_hl_ssm(fr)
                score, _, _ = model.tunnel_detector.detect(pr)
                s = score.item()
                pred = "⚠️ MALICIOUS" if s > 0.5 else "✓ benigno"
                conf = s if s > 0.5 else (1 - s)
                print(f"{t:>7.2f}s {source:<12} {s:>7.3f}  {pred:<12} {conf:>7.1%}")

        # Switch traffic type
        if t > switch_interval:
            benign_traffic = not benign_traffic
            switch_interval += 5.0
            buffer.clear()
            print("-" * 52)

        time.sleep(0.05)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Live capture mode (requires root)
# ═══════════════════════════════════════════════════════════════════════════
def live_capture(model, config, interface=None, duration=60):
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
                    pred = "⚠️ ANOMALY" if s > 0.5 else "✓ benign"
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
def analyze_pcap(model, config, pcap_path):
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
        print(f"Score máximo: {np.max(scores):.4f}")
        pct_anomalous = np.mean([1 for s in scores if s > 0.5])
        print(f"Tráfico anómalo: {pct_anomalous:.1%}")
    else:
        print("No se pudieron formar ventanas de flujo.")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS Live Traffic Detection")
    parser.add_argument("--demo", action="store_true", help="Demo con tráfico sintético")
    parser.add_argument("--live", action="store_true", help="Captura en vivo (requiere root)")
    parser.add_argument("--pcap", type=str, help="Analizar archivo pcap")
    parser.add_argument("--interface", type=str, default=None, help="Interface de red")
    parser.add_argument("--duration", type=int, default=30, help="Duración en segundos")
    args = parser.parse_args()

    # Train model
    print("Inicializando AEGIS...")
    model, config = train_aegis()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    if args.demo:
        demo_synthetic(model, config, duration_seconds=args.duration)
    elif args.live:
        live_capture(model, config, interface=args.interface, duration=args.duration)
    elif args.pcap:
        analyze_pcap(model, config, args.pcap)
    else:
        # Default: run synthetic demo for 10 seconds
        print("Modo por defecto: demo sintética. Usa --live o --pcap para datos reales.")
        demo_synthetic(model, config, duration_seconds=10)
