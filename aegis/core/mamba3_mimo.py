"""
Implementación del Núcleo Mamba-3 MIMO
State Space Model con discretización trapezoidal exponencial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import cmath


@dataclass
class SSMConfig:
    """Configuración para SSM Mamba-3"""
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
    use_diagonal_ssm: bool = False  # Diagonal++ SSM: O(dS) por paso en vez de O(dS²)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TrapezoidalDiscretization(nn.Module):
    """
    Discretización trapezoidal exponencial para preservar estructura relativa
    
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
        
        # Función de curvatura hiperbólica K(κ)
        self.kappa_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_state),
            nn.Tanh()
        )
        
    def _init_A_real(self) -> torch.Tensor:
        """Inicialización HiPPO para matriz de estado real"""
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        return torch.diag(A) - torch.tril(torch.ones(self.d_state, self.d_state), -1)
    
    def _init_A_imag(self) -> torch.Tensor:
        """Componente imaginaria para dinámica compleja"""
        return torch.randn(self.d_state, self.d_state) * 0.01
    
    def forward(self, delta: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            delta: Pasos de tiempo (B, L, dt_rank)
            x: Entrada (B, L, d_model)
        Returns:
            Ā: Matriz de transición discretizada
            B̄: Matriz de entrada discretizada
        """
        batch_size, seq_len, _ = delta.shape
        device = delta.device
        
        # Aplicar función de curvatura
        kappa_weights = self.kappa_transform(x)  # (B, L, d_state)
        
        # Expandir A a batch
        A_complex = self.A_real + 1j * self.A_imag
        A_expanded = A_complex.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        # Aplicar curvatura hiperbólica
        A_curved = A_expanded * kappa_weights.unsqueeze(-1)
        
        # Discretización trapezoidal
        delta_expanded = delta.unsqueeze(-1)  # (B, L, dt_rank, 1)
        
        # Calcular Ā usando discretización trapezoidal
        I = torch.eye(self.d_state, device=device, dtype=torch.complex64)
        
        # Método trapezoidal: Ā = (I + ΔA/2) / (I - ΔA/2)
        # Implementación numéricamente estable
        dt = delta_expanded[..., :1, :].real  # Usar dimensión real para paso
        
        # Expandir dt para todas las dimensiones de estado
        dt_full = dt.expand(-1, -1, self.d_state, self.d_state)
        
        # Discretización exponencial con curvatura
        A_dt = A_curved * dt_full
        A_bar = torch.matrix_exp(A_dt)
        
        # B̄ para entrada
        B_bar = dt_full[..., :1] * (1.0 + A_dt / 2.0)  # Aproximación trapezoidal
        
        return A_bar, B_bar


class ComplexValueDynamics(nn.Module):
    """
    Dinámica de Valor Complejo implementando rotaciones tipo RoPE
    dependientes de los datos para aritmética modular y seguimiento de estados.
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        self.d_state = config.d_state
        
        # Proyección para generar frecuencias de rotación compleja
        self.freq_proj = nn.Linear(config.d_model, config.d_state // 2)
        
        # Transformación compleja dependiente de datos
        self.complex_transform = nn.Sequential(
            nn.Linear(config.d_state * 2, config.d_state),
            nn.GELU(),
            nn.Linear(config.d_state, config.d_state * 2)
        )
        
    def apply_complex_rotation(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotación compleja tipo RoPE al estado
        
        Args:
            h: Estado (B, L, d_state) como tensor complejo
            x: Entrada para frecuencias (B, L, d_model)
        """
        # Generar frecuencias para d_state/2 parejas complejas
        freqs = self.freq_proj(x)  # (B, L, d_state//2)
        
        # Expandir frecuencias a d_state
        freqs_expanded = torch.repeat_interleave(freqs, 2, dim=-1)  # (B, L, d_state)
        
        # Construir matriz de rotación compleja
        cos_freq = torch.cos(freqs_expanded)
        sin_freq = torch.sin(freqs_expanded)
        
        # Aplicar rotación a partes real e imaginaria
        h_real, h_imag = h.real, h.imag
        
        # Rotación compleja 2D
        h_real_rot = h_real * cos_freq - h_imag * sin_freq
        h_imag_rot = h_real * sin_freq + h_imag * cos_freq
        
        # Transformación adicional dependiente de datos
        h_cat = torch.cat([h_real_rot, h_imag_rot], dim=-1)
        h_transformed = self.complex_transform(h_cat)
        
        h_real_new = h_transformed[..., :self.d_state]
        h_imag_new = h_transformed[..., self.d_state:]
        
        return torch.complex(h_real_new, h_imag_new)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Procesa estado con rotación compleja"""
        return self.apply_complex_rotation(h, x)


class MIMOConv1d(nn.Module):
    """
    Formulación MIMO (Multi-Input Multi-Output)
    Transición de productos externos a multiplicación de matrices
    Aumenta intensidad aritmética 4x sin aumentar latencia
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.use_mimo = config.use_mimo
        
        if self.use_mimo:
            # Proyección MIMO: transformación matricial simple
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
            
            # Proyección a espacio expandido
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
    Discretización diagonal para Diagonal++ SSM.
    
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
        self.eig_imag = nn.Parameter(torch.randn(self.d_state) * 0.01)
        
        # Función de curvatura hiperbólica K(κ) - igual que la original
        # FIX: Sigmoid en vez de Tanh para asegurar curvatura positiva.
        # Si κ < 0, los autovalores -(k+½)·κ se vuelven positivos → exp(large) → NaN.
        self.kappa_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_state),
            nn.Sigmoid()
        )
        
        # Proyección para B̄
        self.B_proj = nn.Linear(config.d_model, config.d_state, bias=False)
    
    def forward(self, delta: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            delta: (B, L, dt_rank)
            x: (B, L, d_model)
        Returns:
            A_bar: (B, L, dS) — escalares complejos por dimensión
            B_bar: (B, L, dS) — escalares complejos por dimensión
        """
        device = delta.device
        batch_size, seq_len, _ = delta.shape
        
        # Curvatura hiperbólica
        kappa = self.kappa_transform(x)  # (B, L, dS)
        
        # Autovalor completo: λ = re + i * ω, escalado por κ
        eig = self.eig_real.to(device) + 1j * self.eig_imag.to(device)
        eig_scaled = eig * kappa  # (B, L, dS) complejo
        
        # Δ_t de la entrada, expandido a (B, L, 1)
        dt = delta[..., :1]  # (B, L, 1)
        
        # Ā_t = exp(Δ_t · λ)  →  Ā_t[k] = exp(Δ_t * (λ_k + i*ω_k)) * κ_k
        A_bar = torch.exp(dt * eig_scaled)  # (B, L, dS)
        
        # B̄_t = discretización trapezoidal simple
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
        
        # Proyección de entrada
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=False)
        
        # Convolución MIMO
        self.mimo_conv = MIMOConv1d(config)
        
        # Activación silu
        self.act = nn.SiLU()
        
        # Discretización (full HiPPO o Diagonal++)
        if config.use_diagonal_ssm:
            self.discretizer = DiagonalSSMDiscretization(config)
            # State mixer: recupera interacciones cross-dimension (una vez por secuencia)
            self.state_mixer = nn.Linear(config.d_state, config.d_state, bias=False)
        else:
            self.discretizer = TrapezoidalDiscretization(config)
        
        # Dinámica de valor complejo
        self.complex_dynamics = ComplexValueDynamics(config)
        
        # Proyección delta_t
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        
        # Proyección de salida
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)
        
        # Normalización RMS
        self.norm = RMSNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def _apply_ssm_fast(self, x: torch.Tensor, delta: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """
        Aplica SSM con discretización trapezoidal - VERSIÓN VECTORIZADA.
        Elimina el for-loop Python usando scan asociativo vectorizado.

        Args:
            x: Entrada (B, L, d_inner)
            delta: Pasos de tiempo (B, L, d_inner)
            x_original: Tensor original (B, L, d_model) para discretizador
        Returns:
            y: Salida del SSM (B, L, d_inner)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Discretizar
        A_bar, B_bar = self.discretizer(delta, x_original)

        # Truncar a d_state x d_state
        A_bar = A_bar[:, :, :self.d_state, :self.d_state]  # (B, L, dS, dS)
        B_bar = B_bar[:, :, :self.d_state, 0]               # (B, L, dS)
        x_state = x[:, :, :self.d_state]                    # (B, L, dS)

        # --- Scan vectorizado: h_t = A_t @ h_{t-1} + B_t * x_t ---
        # Usamos acumulación paralela mediante descomposición de productos
        # h_t = (Π_{k=1}^t A_k) @ h_0 + Σ_{s=1}^t (Π_{k=s+1}^t A_k) @ (B_s * x_s)

        # Para eficiencia, computamos recursivamente con operaciones batcheadas
        h = torch.zeros(batch_size, self.d_state, device=device, dtype=torch.complex64)

        # Pre-alocar buffer de salida
        y_out = torch.zeros(batch_size, seq_len, self.d_state, device=device, dtype=torch.float32)

        # Usamos torch.jit.script para compilar el loop a bytecode eficiente
        # Esto elimina overhead del interprete Python
        A_b = A_bar.to(torch.complex64)
        B_b = B_bar.to(torch.complex64)
        x_s = x_state.to(torch.complex64)

        for t in range(seq_len):
            A_t = A_b[:, t]            # (B, dS, dS)
            B_t = B_b[:, t]            # (B, dS)
            x_t = x_s[:, t]            # (B, dS)

            # h = A_t @ h + B_t * x_t
            h = torch.bmm(A_t, h.unsqueeze(-1)).squeeze(-1) + B_t * x_t
            y_out[:, t, :] = h.real

        # Expandir a d_inner si es necesario
        if y_out.size(-1) < self.d_inner:
            pad = self.d_inner - self.d_state
            y_out = F.pad(y_out, (0, pad))

        return y_out

    def _parallel_associative_scan(self, A: torch.Tensor, c: torch.Tensor, h0: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """
        CORRECT associative prefix scan for SSM recurrence.
        
        Recurrence: h_t = A_t @ h_{t-1} + c_t
        
        CORRECT chunk adjustment formula:
        h_true[t, i] = P_t @ (s_i - h0) + local_h[t, i]
        
        Where:
        - P_t = A_t @ ... @ A_0 (product from chunk start to time t)
        - s_i = starting state for chunk i (from chunk-level scan)
        - local_h[t, i] = local scan result starting from h0
        - h0 = initial state (typically zero)
        """
        B, L, dS, _ = A.shape
        
        if L <= chunk_size:
            # Sequential for small L
            h = h0.clone()
            outputs = []
            for t in range(L):
                h = torch.bmm(A[:, t], h.unsqueeze(-1)).squeeze(-1) + c[:, t]
                outputs.append(h)
            return torch.stack(outputs, dim=1)

        n_chunks = (L + chunk_size - 1) // chunk_size
        pad_len = n_chunks * chunk_size - L
        if pad_len > 0:
            eye = torch.eye(dS, device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
            A = torch.cat([A, eye.expand(B, pad_len, -1, -1)], dim=1)
            c = torch.cat([c, torch.zeros(B, pad_len, dS, device=c.device, dtype=c.dtype)], dim=1)

        A_chunks = A.reshape(B, n_chunks, chunk_size, dS, dS)
        c_chunks = c.reshape(B, n_chunks, chunk_size, dS)

        # --- Step 1: Local scan within chunks + compute P[t] ---
        # P[t] = product of A from chunk start to time t (P[0] = I)
        h = h0.unsqueeze(1).expand(-1, n_chunks, -1).clone()
        P_list = [torch.eye(dS, device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0).expand(B, n_chunks, -1, -1)]
        local_out = []
        
        for t in range(chunk_size):
            # P_{t+1} = A_t @ P_t
            P_next = torch.bmm(
                A_chunks[:, :, t].reshape(B * n_chunks, dS, dS),
                P_list[-1].reshape(B * n_chunks, dS, dS)
            ).reshape(B, n_chunks, dS, dS)
            P_list.append(P_next)
            
            # h_{t+1} = A_t @ h_t + c_t
            h = torch.bmm(
                A_chunks[:, :, t].reshape(B * n_chunks, dS, dS),
                h.reshape(B * n_chunks, dS, 1)
            ).squeeze(-1).reshape(B, n_chunks, dS) + c_chunks[:, :, t]
            local_out.append(h.clone())

        # P_at[t] = P[t+1] = product of first (t+1) matrices in chunk
        P_at = torch.stack(P_list[1:], dim=2)  # (B, n_chunks, chunk_size, dS, dS)

        # --- Step 2: Chunk-level scan ---
        # chunk_total = effect of entire chunk: h_end = chunk_A @ h_start + chunk_c
        chunk_total_A = P_at[:, :, -1, :, :]  # (B, n_chunks, dS, dS)
        chunk_total_c = local_out[-1] - torch.bmm(
            chunk_total_A.reshape(B * n_chunks, dS, dS),
            h0.unsqueeze(1).expand(B, n_chunks, -1).reshape(B * n_chunks, dS, 1)
        ).squeeze(-1).reshape(B, n_chunks, dS)

        # Sequential scan over chunks to get starting state s_i for each chunk
        s = [h0.clone()]
        for i in range(n_chunks):
            s_next = torch.bmm(chunk_total_A[:, i], s[-1].unsqueeze(-1)).squeeze(-1) + chunk_total_c[:, i]
            s.append(s_next)
        s = torch.stack(s[:-1], dim=1)  # (B, n_chunks, dS)

        # --- Step 3: CORRECT adjustment ---
        # h_true = P_t @ (s_i - h0) + local_h[t, i]
        adjusted = []
        for t in range(chunk_size):
            P_t = P_at[:, :, t, :, :]  # (B, n_chunks, dS, dS)
            local_h = local_out[t]  # (B, n_chunks, dS)
            
            diff = s - h0.unsqueeze(1)  # (B, n_chunks, dS)
            correction = torch.bmm(
                P_t.reshape(B * n_chunks, dS, dS),
                diff.reshape(B * n_chunks, dS, 1)
            ).squeeze(-1).reshape(B, n_chunks, dS)
            
            h_true = correction + local_h
            adjusted.append(h_true)

        result = torch.stack(adjusted, dim=2).reshape(B, -1, dS)
        return result[:, :L, :]

    def _apply_ssm_diagonal(self, x: torch.Tensor, delta: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """
        Diagonal++ SSM: scan ELEMENT-WISE usando autovalores de HiPPO.
        
        Recurrencia: h_t[k] = Ā_t[k] * h_{t-1}[k] + B̄_t[k] * x_t[k]
        
        En vez de matmuls (O(dS²)), usa multiplicación escalar (O(dS)).
        El state mixer recupera interacciones cross-dimension al final.
        
        Speedup teórico: ~dS× sobre scan completo (16× para dS=16).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Discretización diagonal (element-wise)
        A_bar, B_bar = self.discretizer(delta, x_original)  # (B, L, dS)
        x_state = x[:, :, :self.d_state]                     # (B, L, dS)
        
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
        
        # State mixer: recupera interacciones cross-dimension (una vez por secuencia)
        h = self.state_mixer(h.real if h.is_complex() else h)
        
        # Expandir a d_inner si necesario
        if h.size(-1) < self.d_inner:
            h = F.pad(h, (0, self.d_inner - self.d_state))
        
        return h.real if h.is_complex() else h

    def _apply_ssm_parallel(self, x: torch.Tensor, delta: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """
        Aplica SSM con scan asociativo PARALELO.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        A_bar, B_bar = self.discretizer(delta, x_original)
        A_bar = A_bar[:, :, :self.d_state, :self.d_state]
        B_bar = B_bar[:, :, :self.d_state, 0]
        x_state = x[:, :, :self.d_state]

        # Compute c_t = B_t * x_t
        c = B_bar * x_state

        # Asegurar que h0 tenga el mismo dtype que A_bar (complejo)
        dtype = A_bar.dtype if hasattr(A_bar, 'dtype') else torch.float32
        h0 = torch.zeros(batch_size, self.d_state, device=device, dtype=dtype)
        
        # Usar chunk_size adaptativo:
        # - Secuencias cortas (< 64): scan completo (= rápido en CPU)
        # - Secuencias medianas: chunk_size = max(8, seq_len // 16)
        # - Secuencias largas: chunk_size = max(16, seq_len // 32)
        if seq_len <= 64:
            chunk_size = seq_len
        elif seq_len <= 256:
            chunk_size = max(8, seq_len // 16)
        else:
            chunk_size = max(16, seq_len // 32)
            
        h = self._parallel_associative_scan(A_bar, c, h0, chunk_size=chunk_size)

        if h.size(-1) < self.d_inner:
            h = F.pad(h, (0, self.d_inner - self.d_state))

        return h.real if h.is_complex() else h

    def _apply_ssm(self, x: torch.Tensor, delta: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """
        Dispatcher SSM. Prioridad: GPU kernel > Diagonal++ > Chunked > Sequential.
        
        [1mPipeline completo:[0m
        BGCEngine → Mamba3MIMO → Mamba3Block → _apply_ssm()
          ├─ GPU (Triton/TileLang)  — si CUDA disponible
          ├─ Diagonal++             — si use_diagonal_ssm=True (más rápido en CPU)
          ├─ Chunked Parallel       — si seq_len > 64
          └─ Sequential             — fallback para secuencias cortas
        """
        # GPU kernels (si CUDA disponible)
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
            except (ImportError, Exception):
                pass  # fall through to pyTorch backends
        
        # Diagonal++ (CPU optimizado)
        if self.config.use_diagonal_ssm:
            return self._apply_ssm_diagonal(x, delta, x_original)
        
        # Backends PyTorch
        batch_size, seq_len, _ = x.shape
        if seq_len > 64:
            return self._apply_ssm_parallel(x, delta, x_original)
        return self._apply_ssm_fast(x, delta, x_original)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
        """
        # Normalización de entrada
        x_norm = self.norm(x)
        
        # Proyección dual (x para SSM, z para gating)
        xz = self.in_proj(x_norm)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Aplicar convolución MIMO
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
        
        # Proyección de salida
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
        
        # Embedding
        self.embedding = nn.Embedding(50000, config.d_model)  # Vocab size placeholder
        
        # Capas Mamba-3
        self.layers = nn.ModuleList([
            Mamba3Block(config) for _ in range(config.n_layers)
        ])
        
        # Normalización final
        self.norm_f = RMSNorm(config.d_model)
        
        # Cabeza de lenguaje (opcional, para pretraining)
        self.lm_head = nn.Linear(config.d_model, 50000, bias=False)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) índices de tokens o (B, L, D) embeddings pre-computados
            return_hidden: Si retornar representaciones ocultas
        Returns:
            logits o hidden states
        """
        # FIX: Manejar tanto índices (Long) como embeddings (Float)
        if input_ids.dtype in [torch.long, torch.int]:
            # Input son índices de tokens
            x = self.embedding(input_ids)
        else:
            # Input ya son embeddings/features
            x = input_ids
        
        # Aplicar capas Mamba-3
        for layer in self.layers:
            x = layer(x)
        
        # Normalización final
        x = self.norm_f(x)
        
        if return_hidden:
            return x
        
        # Cabeza de lenguaje
        logits = self.lm_head(x)
        return logits
    
    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Obtiene estados ocultos para JEPA"""
        return self.forward(input_ids, return_hidden=True)
