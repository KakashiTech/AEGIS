"""
AEGIS — DiagonalPlusPlus SSM Kernel.

Core: Diagonal++ SSM with explicit, hierarchical timescales (κ).
O(dS) per step. CPU-first. No quadratic attention.

For research extensions, see experiments/:
  - fourierflow/   → Attention→SSM structural compiler
  - timescales/    → κ visualization & neural timescales
  - vjepa/         → VJEPA/HJEPA self-supervised learning
  - lorentz/       → Hyperbolic geometry layers
  - cognition/     → AbstractCoT, VSA, LatentMAS
  - causality/     → Causal Foundation Model
  - cyber/         → AEGISCyber PDE-based detection
"""

__version__ = "0.5.0"

from .core.mamba3_mimo import Mamba3MIMO, SSMConfig, MoEStateMixer, DiagonalSSMDiscretization
from .kernels import triton_ssm_scan, is_triton_available, ssm_scan_reference, benchmark_ssm_scan
from .training.trace import StepTrace, RuntimeTraceLogger

__all__ = [
    "Mamba3MIMO", "SSMConfig", "MoEStateMixer", "DiagonalSSMDiscretization",
    "triton_ssm_scan", "is_triton_available", "ssm_scan_reference", "benchmark_ssm_scan",
    "StepTrace", "RuntimeTraceLogger",
]
