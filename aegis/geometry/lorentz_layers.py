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
        Exponential map from origin.
        
        For tangent vector v at origin (v₀ = 0):
          x₀ = cosh(||v_space||)
          x_space = v_space · sinh(||v_space||) / ||v_space||
        """
        v_space = v[..., 1:]
        v_norm = torch.sqrt(torch.clamp(
            (v_space ** 2).sum(dim=-1, keepdim=True), min=self.eps
        ))
        
        x_0 = torch.cosh(v_norm).squeeze(-1)
        x_space = v_space * (torch.sinh(v_norm) / v_norm)
        
        return torch.cat([x_0.unsqueeze(-1), x_space], dim=-1)
    
    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map to origin.
        
        For x on the hyperboloid (x₀ > 0, -x₀² + ||x_space||² = -1/κ):
          v₀ = 0  (tangent vectors at origin have zero time component)
          v_space = x_space · acosh(√κ · x₀) / √(κ · x₀² - 1)
        """
        sqrt_k = math.sqrt(self.curvature)
        x0 = x[..., 0:1]
        x_space = x[..., 1:]
        
        # acosh(√κ · x₀) with clamp for numerical safety
        acosh_arg = torch.clamp(sqrt_k * x0, min=1.0 + self.eps)
        angle = torch.acosh(acosh_arg)
        
        # sinh(acosh(z)) = √(z² - 1)
        sinh_arg = torch.sqrt(torch.clamp(acosh_arg.pow(2) - 1.0, min=self.eps))
        
        # Tangent vector at origin: v₀ = 0
        v_0 = torch.zeros_like(x0)
        v_space = x_space * (angle / sinh_arg)
        
        return torch.cat([v_0, v_space], dim=-1)
    
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projection to Lorentz manifold (x₀ > 0, future-directed sheet)
        """
        x_norm = torch.sqrt(torch.clamp(
            torch.abs(self.minkowski_norm(x)), min=self.eps
        ))
        
        x_proj = x / x_norm.unsqueeze(-1)
        # Ensure future-directed sheet: x₀ must be > 0 for acosh in logmap0
        x_proj = torch.where(x_proj[..., 0:1] > 0, x_proj, -x_proj)
        
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


# NOTE: LorentzAttention was removed in v0.4 — it computed Minkowski inner
# products over Euclidean Q/K vectors, which is geometrically unsound.
# A proper hyperbolic attention layer would require Q/K to live on the
# Lorentz manifold. Re-implement if needed.
