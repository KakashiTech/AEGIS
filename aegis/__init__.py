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

__version__ = "0.4.0"

from .core.mamba3_mimo import Mamba3MIMO, SSMConfig, MoEStateMixer
from .geometry.lorentz_layers import LorentzLinear, LorentzProjection, LorentzManifold, PoincareProjection
from .learning.vjepa import VJEPA, VJEPAConfig, TargetEncoder, Predictor, EBM_energy
from .learning.hjepa import HJEPA, HJEPAConfig
from .cognition.abstract_cot import AbstractCoT, VSAModule, HyperdimensionalEncoder, CircularConvolution
from .cognition.latent_mas import LatentMAS, LatentMASConfig
from .cognition.odar_expert import ODARExpertSystem, ODARConfig
from .engine.bgce_engine import BGCEngine, BGCEConfig, TrainingPipeline, InferenceEngine
from .causality.cfm import CausalFoundationModel, CFMConfig
from .security.aegis_cyber import AEGISCyberDefense, AEGISCyberConfig

__all__ = [
    "Mamba3MIMO", "SSMConfig", "MoEStateMixer",
    "LorentzLinear", "LorentzProjection", "LorentzManifold", "PoincareProjection",
    "VJEPA", "VJEPAConfig", "TargetEncoder", "Predictor", "EBM_energy",
    "HJEPA", "HJEPAConfig",
    "AbstractCoT", "VSAModule", "HyperdimensionalEncoder", "CircularConvolution",
    "LatentMAS", "LatentMASConfig",
    "ODARExpertSystem", "ODARConfig",
    "BGCEngine", "BGCEConfig", "TrainingPipeline", "InferenceEngine",
    "CausalFoundationModel", "CFMConfig",
    "AEGISCyberDefense", "AEGISCyberConfig",
]
