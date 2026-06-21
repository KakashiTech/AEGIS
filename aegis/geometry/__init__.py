import warnings
warnings.warn("aegis.geometry is deprecated. Import from experiments.lorentz.geometry instead.", DeprecationWarning, stacklevel=2)
from experiments.lorentz.geometry.lorentz_layers import LorentzLinear, LorentzProjection, PoincareProjection, LorentzManifold
__all__ = ["LorentzLinear", "LorentzProjection", "PoincareProjection", "LorentzManifold"]
