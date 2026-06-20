"""
Mamba-3 MIMO Core
State Space Model with exponential trapezoidal discretization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math


@dataclass
class SSMConfig:
    """Configuration for Mamba-3 SSM"""
    d_model: int = 768
    d_state: int = 64
    d_inner: int = 1536
    dt_rank: int = 16
    dt_min: float = 0.001
    dt_max: float = 0.1
    n_layers: int = 24
    use_complex: bool = True
    use_mimo: bool = True
    curvature_kappa: float = 1.0
    use_diagonal_ssm: bool = True  # Diagonal++ SSM: O(dS) per step instead of O(dS²)
    vocab_size: int = 50000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TrapezoidalDiscretization(nn.Module):
    """
    Exponential trapezoidal discretization to preserve relative structure
    
    h_t = Ā * h_{t-1} + B̄ * x_t
    Ā = exp(Δ * A ⊙ K(κ))
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        self.d_state = config.d_state
        self.curvature_kappa = config.curvature_kappa
        
        # Matriz de estado continua A (HiPPO initialization)
        self.register_buffer('A_real', self._init_A_real())
        self.register_buffer('A_imag', self._init_A_imag())
        
        # Hyperbolic curvature function K(κ)
        # Sigmoid [0,1] ensures positive curvature. Tanh [-1,1] can give κ<0
        # → eigenvalues -(k+½)·κ positive → exp(large) → NaN.
        self.kappa_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_state),
            nn.Sigmoid()
        )
        
    def _init_A_real(self) -> torch.Tensor:
        """HiPPO initialization for real state matrix"""
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        return torch.diag(A) - torch.tril(torch.ones(self.d_state, self.d_state), -1)
    
    def _init_A_imag(self) -> torch.Tensor:
        """Imaginary component for complex dynamics"""
        return torch.randn(self.d_state, self.d_state) * 0.01
    
    def forward(self, delta: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            delta: Steps de tiempo (B, L, dt_rank)
            x: Entrada (B, L, d_model)
        Returns:
            Ā: Discretized transition matrix
            B̄: Discretized input matrix
        """
        batch_size, seq_len, _ = delta.shape
        device = delta.device
        
        # Apply curvature function
        kappa_weights = self.kappa_transform(x)  # (B, L, d_state)
        
        # Expandir A a batch
        A_complex = self.A_real + 1j * self.A_imag
        A_expanded = A_complex.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        # Apply hyperbolic curvature
        A_curved = A_expanded * kappa_weights.unsqueeze(-1)
        
        # Trapezoidal discretization
        delta_expanded = delta.unsqueeze(-1)  # (B, L, dt_rank, 1)
        
        # Compute Ā using first-order discretization
        # Ā ≈ I + ΔA·κ  (avoids matrix_exp O(dS³))
        dt = delta_expanded.mean(dim=-2, keepdim=True).real  # Media sobre dt_rank
        
        # Expandir dt para todas las dimensiones de estado
        dt_full = dt.expand(-1, -1, self.d_state, self.d_state)
        
        # Exponential discretization with curvature
        # FIX: first-order approximation instead of matrix_exp O(dS³)
        # exp(ΔA) ≈ I + ΔA, valid for small Δ (dt ∈ [0.001, 0.1])
        # Mamba-2 uses ZOH with the same approximation.
        A_dt = A_curved * dt_full
        I = torch.eye(self.d_state, device=device, dtype=torch.complex64)
        A_bar = I + A_dt  # First-order approximation, avoids matrix_exp O(dS³)
        
        # B̄ para entrada (trapezoidal: Δ·(I + ΔA/2))
        B_bar = dt_full[..., :1] * (1.0 + A_dt / 2.0)
        
        return A_bar, B_bar


class MIMOConv1d(nn.Module):
    """
    MIMO (Multi-Input Multi-Output) formulation
    Transition from outer products to matrix multiplication
    Increases arithmetic intensity 4x without increasing latency
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.use_mimo = config.use_mimo
        
        if self.use_mimo:
            # MIMO projection: simple matrix transformation
            self.mimo_proj = nn.Linear(config.d_inner, config.d_inner * 4)
            self.mimo_gate = nn.Linear(config.d_inner, 4)
        else:
            self.conv1d = nn.Conv1d(
                in_channels=config.d_inner,
                out_channels=config.d_inner,
                kernel_size=4,
                groups=config.d_inner,
                padding=3
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_inner)
        Returns:
            output: (B, L, d_inner)
        """
        if self.use_mimo:
            batch_size, seq_len, _ = x.shape
            
            # Projection to expanded space
            x_expanded = self.mimo_proj(x)  # (B, L, d_inner * 4)
            x_expanded = x_expanded.view(batch_size, seq_len, self.d_inner, 4)
            
            # Gates para combinar
            gates = torch.softmax(self.mimo_gate(x), dim=-1)  # (B, L, 4)
            gates = gates.unsqueeze(2)  # (B, L, 1, 4)
            
            # Combinar con gates
            output = (x_expanded * gates).sum(dim=-1)  # (B, L, d_inner)
            
            return output
        else:
            # Fallback a conv1d tradicional
            x_t = x.transpose(1, 2)  # (B, d_inner, L)
            out = self.conv1d(x_t)[:, :, :x.size(1)]
            return out.transpose(1, 2)


class DiagonalSSMDiscretization(nn.Module):
    """
    Diagonal discretization for Diagonal++ SSM.
    
    En vez de la matriz HiPPO completa (dS×dS), usa solo sus AUTOVALORES.
    La recurrencia se vuelve ELEMENT-WISE: O(dS) por paso en vez de O(dS²).
    
    λ_k = -(k + ½)  para k = 0, ..., dS-1  (autovalores de HiPPO)
    ω_k = aprendido de datos  (frecuencia compleja)
    
    Ā_t[k] = exp(Δ_t * (λ_k + i * ω_k))
    h_t[k] = Ā_t[k] * h_{t-1}[k] + B̄_t[k] * x_t[k]
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        self.d_state = config.d_state
        self.d_model = config.d_model
        
        # Autovalores de HiPPO: λ_k = -(k + ½)
        k = torch.arange(self.d_state, dtype=torch.float32)
        eigenvalues_real = -(k + 0.5)
        self.register_buffer('eig_real', eigenvalues_real)
        
        # Frecuencias complejas aprendidas (inicializadas cerca de cero)
        # Escaladas con tanh para evitar divergencia: eig_imag ∈ [-π, π]
        self.eig_imag_raw = nn.Parameter(torch.randn(self.d_state) * 0.01)
        
        # Hyperbolic curvature function K(κ) - same as original
        # FIX: Sigmoid en vez de Tanh para asegurar curvatura positiva.
        # Si κ < 0, los autovalores -(k+½)·κ se vuelven positivos → exp(large) → NaN.
        self.kappa_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_state),
            nn.Sigmoid()
        )
        
    def forward(self, delta: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            delta: (B, L, dt_rank)
            x: (B, L, d_model)
        Returns:
            A_bar: (B, L, dS) — complex scalars per dimension
            B_bar: (B, L, dS) — complex scalars per dimension
        """
        device = delta.device
        batch_size, seq_len, _ = delta.shape
        
        # Hyperbolic curvature
        kappa = self.kappa_transform(x)  # (B, L, dS)
        
        # Autovalor completo: λ = re + i * ω, escalado por κ
        eig_imag = torch.tanh(self.eig_imag_raw.to(device)) * math.pi  # bound to [-π, π]
        eig = self.eig_real.to(device) + 1j * eig_imag
        eig_scaled = eig * kappa  # (B, L, dS) complejo
        
        # Δ_t de la entrada: media sobre todos los canales dt_rank
        dt = delta.mean(dim=-1, keepdim=True)  # (B, L, 1)
        
        # Ā_t = exp(Δ_t · λ)  →  Ā_t[k] = exp(Δ_t * (λ_k + i*ω_k)) * κ_k
        A_bar = torch.exp(dt * eig_scaled)  # (B, L, dS)
        
        # B̄_t = simple trapezoidal discretization
        B_bar = dt * (1.0 + dt * eig_scaled / 2.0)  # (B, L, dS)
        
        return A_bar, B_bar


class Mamba3Block(nn.Module):
    """Bloque Mamba-3 individual con SSM complejo y MIMO"""
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        
        # Input projection
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=False)
        
        # MIMO convolution
        self.mimo_conv = MIMOConv1d(config)
        
        # SiLU activation
        self.act = nn.SiLU()
        
        # Diagonal++ discretization (O(dS) per step)
        if not config.use_diagonal_ssm:
            import warnings
            warnings.warn("use_diagonal_ssm=False is deprecated. Using Diagonal++.")
        self.discretizer = DiagonalSSMDiscretization(config)
        self.x_to_state = nn.Linear(config.d_inner, config.d_state, bias=False)
        self.state_mixer = nn.Linear(config.d_state, config.d_inner, bias=False)
        
        # delta_t projection
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)
        
        # RMS normalization
        self.norm = RMSNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def _apply_ssm_diagonal(self, x: torch.Tensor, delta: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """
        Diagonal++ SSM: scan ELEMENT-WISE usando autovalores de HiPPO.
        
        Recurrencia: h_t[k] = Ā_t[k] * h_{t-1}[k] + B̄_t[k] * x_t[k]
        
        Instead of matmuls (O(dS²)), uses scalar multiplication (O(dS)).
        El state mixer recupera interacciones cross-dimension al final.
        
        Theoretical speedup: ~dS× over full scan (16× for dS=16).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Diagonal discretization (element-wise)
        A_bar, B_bar = self.discretizer(delta, x_original)  # (B, L, dS)
        x_state = self.x_to_state(x)                         # (B, L, dS)
        
        # c_t = B̄_t * x_t (element-wise)
        c = B_bar * x_state  # (B, L, dS)
        
        # --- Scan diagonal: h_t = Ā_t ⊙ h_{t-1} + c_t ---
        # Elimina el bmm O(dS²) del scan original, reemplazado por mul O(dS).
        # Sin matmuls, sin complex value dynamics costosos.
        h = torch.zeros(batch_size, self.d_state, device=device, dtype=A_bar.dtype)
        h_states = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + c[:, t]  # element-wise: O(dS) por paso
            h_states.append(h)
        h = torch.stack(h_states, dim=1)
        
        # State mixer: projects d_state → d_inner (all channels with temporal signal)
        h = self.state_mixer(h.real if h.is_complex() else h)
        
        return h.real if h.is_complex() else h

    def _apply_ssm(self, x: torch.Tensor, delta: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """
        SSM dispatcher. Priority: GPU (Triton) > Diagonal++ (CPU).
        
        BGCEngine → Mamba3MIMO → Mamba3Block → _apply_ssm()
          ├─ GPU (Triton/TileLang)  — if CUDA available
          └─ Diagonal++             — CPU-optimized O(dS) per step
        """
        # GPU kernels (if CUDA available)
        if x.is_cuda:
            try:
                from ..kernels.tilelang_h100 import best_ssm_backend
                backend = best_ssm_backend()
                if backend in ('triton', 'tilelang'):
                    from ..kernels.triton_ssm import triton_ssm_scan
                    A_bar, B_bar = self.discretizer(delta, x_original)
                    if isinstance(A_bar, torch.Tensor) and A_bar.dim() == 4:
                        A_bar = A_bar[:, :, :self.d_state, :self.d_state]
                        B_bar = B_bar[:, :, :self.d_state, 0]
                    x_state = x[:, :, :self.d_state]
                    c = B_bar * x_state
                    h0 = torch.zeros(x.size(0), self.d_state, device=x.device, dtype=A_bar.dtype)
                    h = triton_ssm_scan(A_bar, c, h0)
                    if h.size(-1) < self.d_inner:
                        h = F.pad(h, (0, self.d_inner - self.d_state))
                    return h.real if h.is_complex() else h
            except ImportError:
                pass
            except Exception as e:
                import warnings
                warnings.warn(f"GPU kernel failed ({e}), falling back to Diagonal++")
        
        # Diagonal++ (CPU-optimized, O(dS) per step)
        return self._apply_ssm_diagonal(x, delta, x_original)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba-3 block.
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
        """
        # Input normalization
        x_norm = self.norm(x)
        
        # Dual projection (x for SSM, z for gating)
        xz = self.in_proj(x_norm)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Apply MIMO convolution
        x_conv = self.mimo_conv(x_ssm)
        x_conv = self.act(x_conv)
        
        # Generar delta_t
        delta = F.softplus(self.dt_proj(
            x_conv[:, :, :self.config.dt_rank]
        ))
        
        # Aplicar SSM (pasar x_norm como referencia original)
        y = self._apply_ssm(x_conv, delta, x_norm)
        
        # Gating
        y = y * self.act(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        # Residual connection
        return output + x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class Mamba3MIMO(nn.Module):
    """
    Modelo Mamba-3 MIMO completo
    Complejidad O(L) en tiempo de secuencia
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            Mamba3Block(config) for _ in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_model)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """Forward pass. Accepts token indices (long) or pre-computed embeddings (float).
        Args:
            input_ids: (B, L) token indices or (B, L, D) embeddings
            return_hidden: If True, return hidden states instead of logits
        Returns:
            logits (B, L, V) or hidden states (B, L, D)
        """
        # FIX: Handle both token indices (Long) and embeddings (Float)
        if input_ids.dtype in [torch.long, torch.int]:
            # Input is token indices
            x = self.embedding(input_ids)
        else:
            # Input ya son embeddings/features
            x = input_ids
        
        # Aplicar capas Mamba-3
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm_f(x)
        
        if return_hidden:
            return x
        
        # Cabeza de lenguaje
        logits = self.lm_head(x)
        return logits
    
    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Obtiene estados ocultos para JEPA"""
        return self.forward(input_ids, return_hidden=True)
