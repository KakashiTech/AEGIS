import warnings
warnings.warn("aegis.causality is deprecated. Import from experiments.causality.causality instead.", DeprecationWarning, stacklevel=2)
from experiments.causality.causality.cfm import CausalFoundationModel, CFMConfig, PartialCausalGraph, ATEEstimator, CausalAttention
__all__ = ["CausalFoundationModel", "CFMConfig", "PartialCausalGraph", "ATEEstimator", "CausalAttention"]
