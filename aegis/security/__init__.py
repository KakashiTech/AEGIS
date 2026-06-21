import warnings
warnings.warn("aegis.security is deprecated. Import from experiments.cyber.security instead.", DeprecationWarning, stacklevel=2)
from experiments.cyber.security.aegis_cyber import AEGISCyberDefense, AEGISCyberConfig
__all__ = ["AEGISCyberDefense", "AEGISCyberConfig"]
