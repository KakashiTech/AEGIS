"""
Geometric Layers: Lorentz Neural Networks (LNN)

Lorentz space and Minkowski metric implementation
For hierarchical structures without distortion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LorentzManifold:
    """
    Lorentz manifold with Minkowski metric
    diag(-1, 1, 1, ..., 1)
    """
    
    def __init__(self, curvature: float = 1.0, dim: int = 768):
        self.curvature = curvature
        self.dim = dim
        self.eps = 1e-8
        
    def minkowski_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Minkowski inner product
        <x, y>_L = -x_0*y_0 + sum(x_i*y_i for i=1 to D)
        """
        # x, y: (..., D+1)
        # Componente temporal (negativa)
        time_dot = -x[..., 0] * y[..., 0]
        # Componentes espaciales (positivas)
        space_dot = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        return time_dot + space_dot
    
    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Minkowski norm (can be negative)"""
        return self.minkowski_dot(x, x)
    
    def lorentzian_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Distance in Lorentz manifold
        d_L(x, y) = arccosh(-<x, y>_L)
        """
        dot = -self.minkowski_dot(x, y)
        # Asegurar que dot >= 1 para arccosh
        dot = torch.clamp(dot, min=1.0 + self.eps)
        return torch.acosh(dot) / math.sqrt(self.curvature)
    
    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from origin
        """
        v_norm = torch.sqrt(torch.clamp(
            self.minkowski_norm(v), min=self.eps
        ))
        
        # Componente temporal
        x_0 = torch.cosh(v_norm)
        # Componentes espaciales
        x_space = v[..., 1:] * (torch.sinh(v_norm) / v_norm).unsqueeze(-1)
        
        return torch.cat([x_0.unsqueeze(-1), x_space], dim=-1)
    
    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map to origin
        """
        x_norm = torch.sqrt(torch.clamp(
            self.minkowski_norm(x), min=self.eps
        ))
        
        # Componente temporal
        v_0 = x[..., 0]
        # Componentes espaciales
        v_space = x[..., 1:] * (torch.acosh(torch.clamp(x[..., 0], min=1.0 + self.eps)) / 
                                 torch.sqrt(torch.clamp(x_norm, min=self.eps))).unsqueeze(-1)
        
        return torch.cat([v_0.unsqueeze(-1), v_space], dim=-1)
    
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projection to Lorentz manifold
        """
        # Normalizar para estar en el hiperboloide
        x_norm = torch.sqrt(torch.clamp(
            torch.abs(self.minkowski_norm(x)), min=self.eps
        ))
        
        # Asegurar componente temporal positiva
        x_proj = x / x_norm.unsqueeze(-1)
        x_proj = torch.abs(x_proj)
        x_proj[..., 0] = -x_proj[..., 0]  # Negative by signature convention
        
        return x_proj


class LorentzLinear(nn.Module):
    """
    Linear layer in Lorentz space
    
    Implements distance to hyperplane:
    d_signed = (1/sqrt(κ)) * arcsinh(sqrt(κ) * <x, v>_L)
    """
    
    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature
        self.manifold = LorentzManifold(curvature, in_features)
        self.eps = 1e-8
        
        # Pesos en espacio de Minkowski
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features + 1) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Inicializar pesos en la variedad
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights respecting Lorentz geometry"""
        with torch.no_grad():
            # Proyectar pesos a la variedad
            for i in range(self.out_features):
                w = self.weight[i]
                # Asegurar que w_0 > ||w_space||
                w_0 = torch.norm(w[1:]) + 0.1
                w_norm = torch.sqrt(torch.abs(-w_0**2 + torch.norm(w[1:])**2) + self.eps)
                self.weight[i, 0] = w_0 / w_norm
                self.weight[i, 1:] = w[1:] / w_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ..., in_features+1) - ya en coordenadas de Lorentz
        Returns:
            output: (B, ..., out_features)
        """
        batch_shape = x.shape[:-1]
        
        # Producto interno de Lorentz vectorizado: <x, w_i>_L para cada w_i
        # x: (B, ..., in_features+1), weight: (out_features, in_features+1)
        # Expandir dimensiones para broadcast correcto
        x_expanded = x.unsqueeze(-2)  # (B, ..., 1, in_features+1)
        w_expanded = self.weight.unsqueeze(0)  # (1, out_features, in_features+1)
        
        # Automatic broadcast: (B, ..., out_features)
        time_dot = -x_expanded[..., 0] * w_expanded[..., 0]  # (B, ..., out_features)
        space_dot = (x_expanded[..., 1:] * w_expanded[..., 1:]).sum(dim=-1)  # (B, ..., out_features)
        dot = time_dot + space_dot
        
        # Formula: d_signed = (1/sqrt(κ)) * arcsinh(sqrt(κ) * dot)
        scaled_dot = math.sqrt(self.curvature) * dot
        output = (1.0 / math.sqrt(self.curvature)) * torch.asinh(scaled_dot)
        
        # Add bias
        output = output + self.bias
        
        return output


class LorentzProjection(nn.Module):
    """
    Projection from Euclidean to Lorentz
    """
    
    def __init__(self, euclidean_dim: int, lorentz_dim: int, curvature: float = 1.0):
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self.lorentz_dim = lorentz_dim
        self.curvature = curvature
        self.manifold = LorentzManifold(curvature, lorentz_dim)
        
        # Mapeo a espacio tangente
        self.to_tangent = nn.Linear(euclidean_dim, lorentz_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ..., euclidean_dim)
        Returns:
            x_lorentz: (B, ..., lorentz_dim+1)
        """
        # Mapear a espacio tangente en el origen
        v = self.to_tangent(x)  # (B, ..., lorentz_dim)
        
        # Extend with time dimension (component 0)
        v_ext = F.pad(v, (1, 0), value=0.0)  # (B, ..., lorentz_dim+1)
        
        # Aplicar mapeo exponencial
        x_lorentz = self.manifold.expmap0(v_ext)
        
        # Proyectar a la variedad
        x_lorentz = self.manifold.proj(x_lorentz)
        
        return x_lorentz


class PoincareProjection(nn.Module):
    """
    Lorentz to Poincare disk projection
    """
    
    def __init__(self, dim: int, curvature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.curvature = curvature
        self.eps = 1e-8
    
    def lorentz_to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stereographic projection: Lorentz -> Poincare
        x_poincare = x_space / (x_0 + 1)
        """
        x_0 = x[..., 0:1]
        x_space = x[..., 1:]
        
        return x_space / (x_0 + 1 + self.eps)
    
    def poincare_to_lorentz(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse: Poincare -> Lorentz
        """
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        
        # x_0 = (1 + ||x||^2) / (1 - ||x||^2)
        x_0 = (1 + x_norm_sq) / (1 - x_norm_sq + self.eps)
        
        # x_space = 2x / (1 - ||x||^2)
        x_space = 2 * x / (1 - x_norm_sq + self.eps)
        
        return torch.cat([x_0, x_space], dim=-1)
    
    def poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Distance in Poincare disk
        """
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        
        # Poincare distance
        num = 2 * torch.norm(x - y, dim=-1, keepdim=True) ** 2
        den = (1 - x_norm ** 2) * (1 - y_norm ** 2)
        
        return torch.acosh(1 + num / (den + self.eps))


class LorentzAttention(nn.Module):
    """
    Attention mechanism in Lorentz space
    """
    
    def __init__(self, dim: int, num_heads: int = 8, curvature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.curvature = curvature
        self.manifold = LorentzManifold(curvature, dim // num_heads)
        
        self.q_proj = LorentzLinear(dim, dim, curvature)
        self.k_proj = LorentzLinear(dim, dim, curvature)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, dim+1) en coordenadas de Lorentz
        Returns:
            output: (B, L, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Proyecciones
        q = self.q_proj(x)  # (B, L, dim)
        k = self.k_proj(x)  # (B, L, dim)
        v = self.v_proj(x[..., 1:])  # (B, L, dim) - usar componentes espaciales
        
        # Reshape para multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention with Lorentz distances — vectorized O(B·heads·L²)
        # Each head computes pairwise Lorentzian distances in batch.
        # q: (B, heads, L, D), k: (B, heads, L, D)
        # Expand: (B, h, L, 1, D) x (B, h, 1, L, D) → (B, h, L, L)
        q_exp = q.unsqueeze(3)   # (B, H, L, 1, D)
        k_exp = k.unsqueeze(2)   # (B, H, 1, L, D)
        
        # Lorentz inner product: -q0·k0 + q1·k1 + ... + qD·kD
        # Minkowski metric: diag(-1, 1, 1, ..., 1)
        minkowski_dot = -q_exp[..., 0] * k_exp[..., 0] + (q_exp[..., 1:] * k_exp[..., 1:]).sum(dim=-1)
        
        # Lorentzian distance: arccosh(-<q,k>_L)
        # Clamp to avoid NaN from acosh(x<1)
        minkowski_dot = torch.clamp(-minkowski_dot, min=1.0 + 1e-8)
        lorentz_dist = torch.acosh(minkowski_dot)
        
        # Convert to attention scores (smaller distance = higher attention)
        attn_weights = -lorentz_dist / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(output)
