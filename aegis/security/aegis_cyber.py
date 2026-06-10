"""
AEGIS Cyber: Especialización de Defensa Térmica
TVD-HL-SSM (Total Variation Diminishing - Hyperbolic Liquid - State Space Model)
Modela la "física del flujo" de red en lugar de leer bytes
Objetivo: 99.50% Tasa de Verdaderos Positivos contra túneles criptográficos (VLESS Reality)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class AEGISCyberConfig:
    """Configuración AEGIS Cyber"""
    d_model: int = 768
    n_flow_layers: int = 8
    detection_threshold: float = 0.5  # Se calibra con ROC durante entrenamiento
    tvd_coefficient: float = 0.1  # Coeficiente de disipación TVD
    hyperbolic_curvature: float = 1.0  # Curvatura espacio hiperbólico
    liquid_time_constant: float = 0.5  # τ para neuronas líquidas
    sequence_length: int = 256  # Ventana de análisis de flujo
    n_classes: int = 2  # Benigno / Malicioso
    tunnel_types: List[str] = None
    
    def __post_init__(self):
        if self.tunnel_types is None:
            self.tunnel_types = ['vless_reality', 'shadowsocks', 'trojan', 'wireguard']


class FlowPhysicsEncoder(nn.Module):
    """
    Codificador de física del flujo de red
    Convierte tráfico de red a representación de flujo continuo
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        
        # Se determina dinámicamente n_features del input en forward
        self.feature_encoder_fc = None  # Se crea en el primer forward
        
        # Positional encoding temporal para secuencias de flujo
        self.temporal_encoding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.d_model) * 0.02
        )
        
        # Proyección a espacio hiperbólico
        self.to_hyperbolic = nn.Linear(config.d_model, config.d_model + 1)
    
    def encode_flow(self, flow_data: torch.Tensor) -> torch.Tensor:
        """
        Codificar datos de flujo de red
        
        Args:
            flow_data: (B, L, n_features) - características de paquetes
        
        Returns:
            encoded: (B, L, d_model+1) - representación en espacio hiperbólico
        """
        batch_size, seq_len, n_features = flow_data.shape
        
        # Crear feature_encoder dinámicamente si es primera vez o cambia n_features
        if self.feature_encoder_fc is None or self.feature_encoder_fc[0].in_features != n_features:
            self.feature_encoder_fc = nn.Sequential(
                nn.Linear(n_features, self.config.d_model // 2),
                nn.LayerNorm(self.config.d_model // 2),
                nn.GELU(),
                nn.Linear(self.config.d_model // 2, self.config.d_model)
            ).to(flow_data.device, flow_data.dtype)
        
        # Codificar características
        encoded = self.feature_encoder_fc(flow_data)
        
        # Añadir encoding temporal
        if seq_len <= self.config.sequence_length:
            encoded = encoded + self.temporal_encoding[:, :seq_len, :]
        
        # Proyectar a espacio hiperbólico
        hyperbolic = self.to_hyperbolic(encoded)
        
        return hyperbolic


class TVDHyperbolicLiquidSSM(nn.Module):
    """
    TVD-HL-SSM: Modelo de espacio de estado líquido hiperbólico
    con disipación de variación total
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Parámetros de flujo hiperbólico (d_model+1 para espacio hiperbólico)
        self.flow_velocity = nn.Parameter(torch.randn(config.d_model + 1) * 0.1)
        self.flow_viscosity = nn.Parameter(torch.tensor(0.1))
        
        # Matriz de disipación TVD (opera en espacio hiperbólico: d_model+1)
        self.tvd_dissipation = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model + 1),
            nn.Sigmoid()
        )
        
        # Neuronas líquidas (CfC - Continuous-time Cellular Automata)
        # Operan en espacio hiperbólico (d_model+1)
        self.liquid_neurons = nn.ModuleList([
            LiquidNeuron(config.d_model + 1, config.liquid_time_constant)
            for _ in range(config.n_flow_layers)
        ])
        
        # Métrica de Minkowski para espacio hiperbólico
        self.register_buffer('minkowski_metric', 
                           torch.diag(torch.tensor([-1.0] + [1.0] * config.d_model)))
    
    def minkowski_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Producto interno de Minkowski"""
        # x, y: (..., d_model+1)
        time_component = -x[..., 0] * y[..., 0]
        space_component = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        return time_component + space_component
    
    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distancia en espacio hiperbólico"""
        dot = -self.minkowski_product(x, y)
        dot = torch.clamp(dot, min=1.0 + 1e-8)
        return torch.acosh(dot)
    
    def forward(self, flow_repr: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Procesar flujo con TVD-HL-SSM
        
        Args:
            flow_repr: (B, L, d_model+1) en espacio hiperbólico
            dt: Paso de tiempo
        
        Returns:
            processed: (B, L, d_model+1)
        """
        batch_size, seq_len, _ = flow_repr.shape
        x = flow_repr
        
        # Capas de flujo líquido
        for layer_idx, liquid_layer in enumerate(self.liquid_neurons):
            # Evolución temporal continua
            new_states = []
            
            for t in range(seq_len):
                x_t = x[:, t, :]
                
                # Ecuación de flujo hiperbólico con disipación TVD
                # ∂u/∂t + v·∇u = ν∇²u - λ·TVD(u)
                
                # Término de transporte
                if t > 0:
                    transport = self.flow_velocity * (x_t - x[:, t-1, :])
                else:
                    transport = torch.zeros_like(x_t)
                
                # Término de disipación TVD
                tvd_term = self.flow_viscosity * self.tvd_dissipation(x_t)
                
                # Actualización líquida
                dx_dt = -transport + tvd_term
                x_new = x_t + dt * dx_dt
                
                # Aplicar neurona líquida
                x_new = liquid_layer(x_new.unsqueeze(1), dt).squeeze(1)
                
                # Proyectar de vuelta a hiperboloide
                x_new = self._project_to_hyperboloid(x_new)
                
                new_states.append(x_new)
            
            x = torch.stack(new_states, dim=1)
        
        return x
    
    def _project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Proyectar a hiperboloide unitario"""
        # Asegurar que x_0 > ||x_space||
        x_0 = torch.sqrt(torch.norm(x[:, 1:], dim=-1, keepdim=True)**2 + 1.0)
        
        x_proj = torch.cat([x_0, x[:, 1:]], dim=-1)
        return x_proj


class LiquidNeuron(nn.Module):
    """
    Neurona Líquida con dinámica de tiempo continuo
    CfC: Continuous-time Cellular Automata
    """
    
    def __init__(self, dim: int, time_constant: float):
        super().__init__()
        self.dim = dim
        self.tau = time_constant
        
        # Parámetros de ODE
        self.W = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        
        # No-linealidad
        self.activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Solución de tiempo continuo
        dx/dt = -x/τ + f(W·x + U·input)
        """
        # Término de decaimiento
        decay = -x / self.tau
        
        # Término de entrada
        input_term = self.activation(self.W(x))
        
        # Actualización
        dx = decay + input_term
        x_new = x + dt * dx
        
        return x_new


class TunnelDetector(nn.Module):
    """
    Detector especializado en túneles criptográficos
    Entrenado para identificar VLESS Reality, Shadowsocks, Trojan, WireGuard
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        
        # Cabeza de clasificación de túneles
        self.tunnel_classifier = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, len(config.tunnel_types))
        )
        
        # Detector binario (benigno/malicioso)
        self.binary_detector = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Analizador de patrones de ofuscación
        # FIX: num_heads=1 porque embed_dim=769 (d_model+1) no es divisible por 4
        self.obfuscation_analyzer = nn.MultiheadAttention(
            embed_dim=config.d_model + 1,
            num_heads=1,
            batch_first=True
        )
    
    def detect(self, flow_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Detectar túneles criptográficos
        
        Returns:
            is_tunnel: Probabilidad de ser túnel (B, 1)
            tunnel_type: Clasificación por tipo (B, n_types)
            metrics: Estadísticas de detección
        """
        batch_size = flow_repr.size(0)
        
        # Analizar ofuscación
        attended_repr, attention_weights = self.obfuscation_analyzer(
            flow_repr, flow_repr, flow_repr
        )
        
        # Promediar sobre la secuencia
        pooled = attended_repr.mean(dim=1)  # (B, d_model+1)
        
        # Detección binaria
        is_tunnel = self.binary_detector(pooled)
        
        # Clasificación de tipo
        tunnel_type_logits = self.tunnel_classifier(pooled)
        tunnel_type_probs = F.softmax(tunnel_type_logits, dim=-1)
        
        # Métricas
        metrics = {
            'attention_entropy': -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1).mean().item(),
            'max_attention': attention_weights.max().item(),
            'detection_confidence': is_tunnel.mean().item()
        }
        
        return is_tunnel, tunnel_type_probs, metrics


class AEGISCyberDefense(nn.Module):
    """
    Sistema completo de Defensa Térmica AEGIS
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        
        # Componentes principales
        self.flow_encoder = FlowPhysicsEncoder(config)
        self.tvd_hl_ssm = TVDHyperbolicLiquidSSM(config)
        self.tunnel_detector = TunnelDetector(config)
        
        # Umbral de detección adaptativo
        self.detection_threshold = nn.Parameter(torch.tensor(config.detection_threshold))
        
        # Estadísticas de rendimiento
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.total_detected = 0
    
    def analyze_traffic(self, flow_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Analizar tráfico de red
        
        Args:
            flow_data: (B, L, n_features) características de flujo
        
        Returns:
            is_malicious: (B, 1) probabilidad de ser malicioso
            tunnel_type: (B, n_types) clasificación de túnel
            analysis: métricas detalladas
        """
        # Codificar flujo
        flow_repr = self.flow_encoder.encode_flow(flow_data)
        
        # Procesar con TVD-HL-SSM
        processed = self.tvd_hl_ssm(flow_repr)
        
        # Detectar túneles
        is_tunnel, tunnel_type, detection_metrics = self.tunnel_detector.detect(processed)
        
        # Decisión final
        is_malicious = (is_tunnel > self.detection_threshold).float()
        
        analysis = {
            **detection_metrics,
            'detection_threshold': self.detection_threshold.item(),
            'tvd_dissipation_active': True,
            'hyperbolic_curvature': self.config.hyperbolic_curvature
        }
        
        self.total_detected += flow_data.size(0)
        
        return is_malicious, tunnel_type, analysis
    
    def update_metrics(self, predictions: torch.Tensor, ground_truth: torch.Tensor):
        """Actualizar métricas de rendimiento"""
        pred = predictions.bool()
        gt = ground_truth.bool()
        
        self.true_positives += (pred & gt).sum().item()
        self.false_positives += (pred & ~gt).sum().item()
        self.false_negatives += (~pred & gt).sum().item()
        self.true_negatives += (~pred & ~gt).sum().item()
    
    def get_detection_stats(self) -> Dict:
        """Obtener estadísticas de detección"""
        total = self.true_positives + self.false_positives + \
                self.false_negatives + self.true_negatives
        
        if total == 0:
            return {}
        
        tpr = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        fpr = self.false_positives / max(self.false_positives + self.true_negatives, 1)
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        
        return {
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'precision': precision,
            'target_tpr': self.config.detection_threshold,
            'target_achieved': tpr >= self.config.detection_threshold,
            'total_analyzed': self.total_detected
        }
    
    def detect_vless_reality(self, flow_data: torch.Tensor) -> Tuple[bool, float]:
        """Detección específica de VLESS Reality (desafío principal)"""
        is_malicious, tunnel_type, _ = self.analyze_traffic(flow_data)
        
        # VLESS Reality suele ser el índice 0 en tunnel_types
        vless_prob = tunnel_type[0, 0].item()
        
        return is_malicious[0, 0].item() > 0.5, vless_prob
