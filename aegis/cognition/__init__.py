import warnings
warnings.warn("aegis.cognition is deprecated. Import from experiments.cognition.cognition instead.", DeprecationWarning, stacklevel=2)
from experiments.cognition.cognition.abstract_cot import AbstractCoT, VSAModule, HyperdimensionalEncoder, CircularConvolution
from experiments.cognition.cognition.latent_mas import LatentMAS, LatentMASConfig, LatentAgent, LatentCommunicationChannel, AgentMemoryBank
from experiments.cognition.cognition.odar_expert import ODARExpertSystem, ODARConfig, ODARRouter, SystemOneExpert, SystemTwoExpert, VariationalEntropyEstimator
__all__ = ["AbstractCoT", "VSAModule", "HyperdimensionalEncoder", "CircularConvolution", "LatentMAS", "LatentMASConfig", "LatentAgent", "LatentCommunicationChannel", "AgentMemoryBank", "ODARExpertSystem", "ODARConfig", "ODARRouter", "SystemOneExpert", "SystemTwoExpert", "VariationalEntropyEstimator"]
