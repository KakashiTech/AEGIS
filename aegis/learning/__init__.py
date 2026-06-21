import warnings
warnings.warn("aegis.learning is deprecated. Import from experiments.vjepa.learning instead.", DeprecationWarning, stacklevel=2)
from experiments.vjepa.learning.vjepa import VJEPA, VJEPAConfig, TargetEncoder, Predictor, EBM_energy
from experiments.vjepa.learning.hjepa import HJEPA, HJEPAConfig, HierarchicalLevel, CausalTimePrior, MentalRolloutSimulator
__all__ = ["VJEPA", "VJEPAConfig", "TargetEncoder", "Predictor", "EBM_energy", "HJEPA", "HJEPAConfig", "HierarchicalLevel", "CausalTimePrior", "MentalRolloutSimulator"]
