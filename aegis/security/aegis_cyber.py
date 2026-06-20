"""
AEGIS Cyber Defense
TVD-HL-SSM (Total Variation Diminishing - Hyperbolic Liquid - State Space Model)
Models network "flow physics" instead of reading raw bytes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class AEGISCyberConfig:
    """AEGIS Cyber Configuration"""
    d_model: int = 768
    n_flow_layers: int = 8
    detection_threshold: float = 0.5  # Calibrated via ROC during training
    tvd_coefficient: float = 0.1  # TVD dissipation coefficient
    hyperbolic_curvature: float = 1.0  # Hyperbolic space curvature
    liquid_time_constant: float = 0.5  # τ for liquid neurons
    sequence_length: int = 256  # Flow analysis window
    n_classes: int = 2  # Benign / Malicious
    tunnel_types: List[str] = None
    
    def __post_init__(self):
        if self.tunnel_types is None:
            self.tunnel_types = ['vless_reality', 'shadowsocks', 'trojan', 'wireguard']


class FlowPhysicsEncoder(nn.Module):
    """
    Network flow physics encoder
    Converts network traffic to continuous flow representation
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        
        # Dynamic feature encoder registered as ModuleDict for DataParallel compat
        self.feature_encoders = nn.ModuleDict()
        
        # Temporal positional encoding for flow sequences
        self.temporal_encoding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.d_model) * 0.02
        )
        
        # Projection to hyperbolic space
        self.to_hyperbolic = nn.Linear(config.d_model, config.d_model + 1)
    
    def encode_flow(self, flow_data: torch.Tensor) -> torch.Tensor:
        """
        Encode network flow data
        
        Args:
            flow_data: (B, L, n_features) - packet features
        
        Returns:
            encoded: (B, L, d_model+1) - hyperbolic space representation
        """
        batch_size, seq_len, n_features = flow_data.shape
        
        # Create feature_encoder dynamically (registered in ModuleDict)
        key = f"n{n_features}"
        if key not in self.feature_encoders:
            encoder = nn.Sequential(
                nn.Linear(n_features, self.config.d_model // 2),
                nn.LayerNorm(self.config.d_model // 2),
                nn.GELU(),
                nn.Linear(self.config.d_model // 2, self.config.d_model)
            ).to(flow_data.device, flow_data.dtype)
            self.feature_encoders[key] = encoder
        feature_encoder_fc = self.feature_encoders[key]
        
        # Encode features
        encoded = feature_encoder_fc(flow_data)
        
        # Add temporal encoding
        if seq_len <= self.config.sequence_length:
            encoded = encoded + self.temporal_encoding[:, :seq_len, :]
        
        # Project to hyperbolic space
        hyperbolic = self.to_hyperbolic(encoded)
        
        return hyperbolic


class TVDHyperbolicLiquidSSM(nn.Module):
    """
    TVD-HL-SSM: Hyperbolic Liquid State Space Model
    with Total Variation Diminishing dissipation
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Hyperbolic flow parameters (d_model+1 for hyperbolic space)
        self.flow_velocity = nn.Parameter(torch.randn(config.d_model + 1) * 0.1)
        self.flow_viscosity = nn.Parameter(torch.tensor(0.1))
        self.diffusivity = nn.Parameter(torch.tensor(0.01))
        
        # TVD dissipation matrix (operates in hyperbolic space: d_model+1)
        self.tvd_dissipation = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model + 1),
            nn.Sigmoid()
        )
        
        # Liquid neurons (CfC - Continuous-time Cellular Automata)
        # Operate in hyperbolic space (d_model+1)
        self.liquid_neurons = nn.ModuleList([
            LiquidNeuron(config.d_model + 1, config.liquid_time_constant)
            for _ in range(config.n_flow_layers)
        ])
        
        # Minkowski metric for hyperbolic space
        self.register_buffer('minkowski_metric', 
                           torch.diag(torch.tensor([-1.0] + [1.0] * config.d_model)))
    
    def minkowski_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Minkowski inner product"""
        # x, y: (..., d_model+1)
        time_component = -x[..., 0] * y[..., 0]
        space_component = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        return time_component + space_component
    
    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distance in hyperbolic space"""
        dot = -self.minkowski_product(x, y)
        dot = torch.clamp(dot, min=1.0 + 1e-8)
        return torch.acosh(dot)
    
    def forward(self, flow_repr: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Process flow with TVD-HL-SSM
        
        Args:
            flow_repr: (B, L, d_model+1) in hyperbolic space
            dt: Time step
        
        Returns:
            processed: (B, L, d_model+1)
        """
        batch_size, seq_len, _ = flow_repr.shape
        x = flow_repr
        
        # Liquid flow layers
        for layer_idx, liquid_layer in enumerate(self.liquid_neurons):
            # Continuous-time evolution
            new_states = []
            
            for t in range(seq_len):
                x_t = x[:, t, :]
                
                # Hyperbolic flow equation with TVD dissipation
                # ∂u/∂t = -v·∇u + ν∇²u - λ·TVD(u)
                
                # Transport term: backward difference approx of v·∇u
                if t > 0:
                    transport = self.flow_velocity * (x_t - x[:, t-1, :])
                else:
                    transport = torch.zeros_like(x_t)
                
                # Diffusion term: second-order central difference
                if t > 0 and t < seq_len - 1:
                    diffusion = x[:, t-1, :] - 2 * x_t + x[:, t+1, :]
                else:
                    diffusion = torch.zeros_like(x_t)
                
                # TVD dissipation term (negative sign per PDE)
                tvd_term = self.flow_viscosity * self.tvd_dissipation(x_t)
                
                # PDE-correct update: ∂u/∂t = -transport + ν·diffusion - λ·TVD
                dx_dt = -transport + self.diffusivity * diffusion - tvd_term
                x_new = x_t + dt * dx_dt
                
                # Apply liquid neuron
                x_new = liquid_layer(x_new.unsqueeze(1), dt).squeeze(1)
                
                # Project back to hyperboloid
                x_new = self._project_to_hyperboloid(x_new)
                
                new_states.append(x_new)
            
            x = torch.stack(new_states, dim=1)
        
        return x
    
    def _project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project to unit hyperboloid"""
        # Ensure x_0 > ||x_space||
        x_0 = torch.sqrt(torch.norm(x[:, 1:], dim=-1, keepdim=True)**2 + 1.0)
        
        x_proj = torch.cat([x_0, x[:, 1:]], dim=-1)
        return x_proj


class LiquidNeuron(nn.Module):
    """
    Liquid Neuron with continuous-time RK4 dynamics
    CfC: Continuous-time Cellular Automata
    dx/dt = -x/τ + tanh(W·x)
    Local truncation error: O(dt⁵)
    """
    
    def __init__(self, dim: int, time_constant: float):
        super().__init__()
        self.dim = dim
        self.tau = time_constant
        
        # ODE parameters
        self.W = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        
        # Non-linearity
        self.activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Continuous-time solution via RK4
        dx/dt = -x/τ + f(W·x + U·input)
        """
        def f(state):
            decay = -state / self.tau
            input_term = self.activation(self.W(state))
            return decay + input_term
        
        k1 = f(x)
        k2 = f(x + dt/2.0 * k1)
        k3 = f(x + dt/2.0 * k2)
        k4 = f(x + dt * k3)
        x_new = x + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
        return x_new


class TunnelDetector(nn.Module):
    """
    Cryptographic tunnel detector
    Trained to identify VLESS Reality, Shadowsocks, Trojan, WireGuard
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        
        # Tunnel classification head
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
        
        # Binary detector (benign/malicious)
        self.binary_detector = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Obfuscation pattern analyzer
        # FIX: num_heads=1 because embed_dim=769 (d_model+1) not divisible by 4
        self.obfuscation_analyzer = nn.MultiheadAttention(
            embed_dim=config.d_model + 1,
            num_heads=1,
            batch_first=True
        )
    
    def detect(self, flow_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Detect cryptographic tunnels
        
        Returns:
            is_tunnel: Tunnel probability (B, 1)
            tunnel_type: Classification by type (B, n_types)
            metrics: Detection statistics
        """
        batch_size = flow_repr.size(0)
        
        # Analyze obfuscation
        attended_repr, attention_weights = self.obfuscation_analyzer(
            flow_repr, flow_repr, flow_repr
        )
        
        # Average over sequence
        pooled = attended_repr.mean(dim=1)  # (B, d_model+1)
        
        # Binary detection
        is_tunnel = self.binary_detector(pooled)
        
        # Type classification
        tunnel_type_logits = self.tunnel_classifier(pooled)
        tunnel_type_probs = F.softmax(tunnel_type_logits, dim=-1)
        
        # Metrics
        metrics = {
            'attention_entropy': -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1).mean().item(),
            'max_attention': attention_weights.max().item(),
            'detection_confidence': is_tunnel.mean().item()
        }
        
        return is_tunnel, tunnel_type_probs, metrics


class AEGISCyberDefense(nn.Module):
    """
    Complete AEGIS Thermal Defense System
    """
    
    def __init__(self, config: AEGISCyberConfig):
        super().__init__()
        self.config = config
        
        # Main components
        self.flow_encoder = FlowPhysicsEncoder(config)
        self.tvd_hl_ssm = TVDHyperbolicLiquidSSM(config)
        self.tunnel_detector = TunnelDetector(config)
        
        # Adaptive detection threshold
        self.detection_threshold = nn.Parameter(torch.tensor(config.detection_threshold))
        
        # Performance statistics
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.total_detected = 0
    
    def analyze_traffic(self, flow_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Analyze network traffic
        
        Args:
            flow_data: (B, L, n_features) flow features
        
        Returns:
            is_malicious: (B, 1) probability of being malicious
            tunnel_type: (B, n_types) tunnel classification
            analysis: detailed metrics
        """
        # Encode flow
        flow_repr = self.flow_encoder.encode_flow(flow_data)
        
        # Process with TVD-HL-SSM
        processed = self.tvd_hl_ssm(flow_repr)
        
        # Detect tunnels
        is_tunnel, tunnel_type, detection_metrics = self.tunnel_detector.detect(processed)
        
        # Final decision
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
        """Update performance metrics"""
        pred = predictions.bool()
        gt = ground_truth.bool()
        
        self.true_positives += (pred & gt).sum().item()
        self.false_positives += (pred & ~gt).sum().item()
        self.false_negatives += (~pred & gt).sum().item()
        self.true_negatives += (~pred & ~gt).sum().item()
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
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
        """Specific VLESS Reality detection (primary challenge)"""
        is_malicious, tunnel_type, _ = self.analyze_traffic(flow_data)
        
        # VLESS Reality is usually index 0 in tunnel_types
        vless_prob = tunnel_type[0, 0].item()
        
        return is_malicious[0, 0].item() > 0.5, vless_prob
