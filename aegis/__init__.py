"""
AEGIS — Continuous-Time Foundation Engine.

Not a language model. Not a transformer.
A learned simulator of underlying flow dynamics.

Components:
  - Diagonal++ SSM: element-wise recurrence, O(ds) per step
  - Lorentz geometry: hyperbolic space for causal hierarchies
  - VJEPA: latent predictive learning without reconstruction
  - Abstract-CoT: reasoning without verbal tokens
  - AEGIS Cyber: physical flow-based intrusion detection
"""

__version__ = "0.1.0"

from .core.mamba3_mimo import Mamba3MIMO, SSMConfig
from .geometry.lorentz_layers import LorentzLinear, LorentzProjection
from .learning.vjepa import VJEPA, VJEPAConfig
from .cognition.abstract_cot import AbstractCoT, VSAModule
from .engine.bgce_engine import BGCEngine, BGCEConfig

__all__ = [
    "Mamba3MIMO", "SSMConfig",
    "LorentzLinear", "LorentzProjection",
    "VJEPA", "VJEPAConfig",
    "AbstractCoT", "VSAModule",
    "BGCEngine", "BGCEConfig",
]
