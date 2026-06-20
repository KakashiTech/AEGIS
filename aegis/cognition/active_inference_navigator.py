"""
Amortized Active Inference for Scientific Discovery
System that generates hypotheses and runs "mental experiments"
Hypothesis Space Navigator to minimize epistemic surprise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
import random


@dataclass
class ActiveInferenceConfig:
    """Active Inference Configuration"""
    d_model: int = 768
    n_hypothesis_space: int = 1000  # Hypothesis space size
    epistemic_uncertainty_threshold: float = 0.3
    
    # Hypothesis generation
    n_candidate_hypotheses: int = 10
    hypothesis_dim: int = 256
    
    # Mental experiments
    n_mental_rollouts: int = 20
    rollout_horizon: int = 10
    
    # Amortization
    amortization_hidden_dim: int = 512
    surprise_tolerance: float = 0.1


class HypothesisSpace(nn.Module):
    """
    Latent Hypothesis Space
    Each hypothesis is a point in latent space representing a theory
    """
    
    def __init__(self, config: ActiveInferenceConfig):
        super().__init__()
        self.config = config
        
        # Hypothesis embeddings
        self.hypothesis_embeddings = nn.Parameter(
            torch.randn(config.n_hypothesis_space, config.hypothesis_dim)
        )
        
        # Hypothesis prior (how likely each is a priori)
        self.hypothesis_priors = nn.Parameter(
            torch.ones(config.n_hypothesis_space) / config.n_hypothesis_space
        )
        
        # Mapeo a espacio del modelo
        self.to_model_space = nn.Sequential(
            nn.Linear(config.hypothesis_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU()
        )
        
        # Distance metric in hypothesis space
        self.distance_metric = nn.Parameter(torch.eye(config.hypothesis_dim))
    
    def encode_hypothesis(self, hypothesis_id: int) -> torch.Tensor:
        """Get hypothesis embedding"""
        h_emb = self.hypothesis_embeddings[hypothesis_id]
        return self.to_model_space(h_emb)
    
    def compute_epistemic_distance(self, h1: int, h2: int) -> float:
        """Epistemic distance between two hypotheses"""
        diff = self.hypothesis_embeddings[h1] - self.hypothesis_embeddings[h2]
        distance = torch.sqrt(diff @ self.distance_metric @ diff)
        return distance.item()
    
    def sample_hypothesis(self, temperature: float = 1.0) -> int:
        """Sample hypotheses from prior"""
        probs = F.softmax(self.hypothesis_priors / temperature, dim=0)
        return torch.multinomial(probs, 1).item()
    
    def get_high_uncertainty_regions(self, n_regions: int = 5) -> List[int]:
        """Identify regions of high epistemic uncertainty"""
        # Calculate entropy of each hypothesis relative to neighbors
        uncertainties = []
        
        for i in range(self.config.n_hypothesis_space):
            # Distances to all others
            distances = []
            for j in range(self.config.n_hypothesis_space):
                if i != j:
                    dist = self.compute_epistemic_distance(i, j)
                    distances.append(dist)
            
            # Variance of distances as local uncertainty measure
            uncertainty = torch.tensor(distances).std().item()
            uncertainties.append((i, uncertainty))
        
        # Sort by uncertainty and return top-n
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in uncertainties[:n_regions]]


class MentalExperimentSimulator(nn.Module):
    """
    "Mental Experiment" Simulator
    Predicts virtual experiment outcomes without physical execution
    """
    
    def __init__(self, config: ActiveInferenceConfig, world_model: nn.Module):
        super().__init__()
        self.config = config
        self.world_model = world_model  # World model (H-JEPA or similar)
        
        # Generador de resultados experimentales
        self.outcome_predictor = nn.Sequential(
            nn.Linear(config.d_model + config.hypothesis_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
        # Variance estimator (prediction uncertainty)
        self.variance_estimator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Softplus()
        )
    
    def simulate_experiment(self, 
                          current_state: torch.Tensor,
                          hypothesis_embedding: torch.Tensor,
                          intervention: Optional[torch.Tensor] = None) -> Dict:
        """
        Simulate a mental experiment
        
        Args:
            current_state: Current system state
            hypothesis_embedding: Hypothesis to test
            intervention: Experimental intervention (optional)
        
        Returns:
            dict with predicted outcome, uncertainty, expected surprise
        """
        batch_size = current_state.size(0)
        
        # Combine state with hypothesis
        combined = torch.cat([current_state, hypothesis_embedding.expand(batch_size, -1)], dim=-1)
        
        # Predict outcome
        predicted_outcome = self.outcome_predictor(combined)
        
        # Estimate uncertainty
        predicted_variance = self.variance_estimator(predicted_outcome)
        
        # Simular rollout futuro
        future_outcomes = []
        state = predicted_outcome
        
        for step in range(self.config.rollout_horizon):
            with torch.no_grad():
                next_state = self.world_model(state.unsqueeze(1) if state.dim() == 2 else state)
                if next_state.dim() == 3:
                    next_state = next_state[:, -1, :]
                future_outcomes.append(next_state)
                state = next_state
        
        # Compute expected surprise (prediction divergence)
        if len(future_outcomes) > 1:
            future_stack = torch.stack(future_outcomes, dim=1)
            surprise = future_stack.std(dim=1).mean().item()
        else:
            surprise = 0.0
        
        return {
            'predicted_outcome': predicted_outcome,
            'predicted_variance': predicted_variance,
            'future_rollouts': future_outcomes,
            'expected_surprise': surprise,
            'rollout_horizon': self.config.rollout_horizon
        }
    
    def batch_simulate(self,
                      current_state: torch.Tensor,
                      hypotheses: List[int],
                      hypothesis_space: HypothesisSpace) -> List[Dict]:
        """Simulate multiple hypotheses in parallel"""
        results = []
        
        for h_id in hypotheses:
            h_emb = hypothesis_space.encode_hypothesis(h_id)
            result = self.simulate_experiment(current_state, h_emb)
            result['hypothesis_id'] = h_id
            results.append(result)
        
        return results


class EpistemicSurpriseMinimizer(nn.Module):
    """
    Epistemic Surprise Minimizer
    Guides exploration toward uncertainty-reducing regions
    """
    
    def __init__(self, config: ActiveInferenceConfig):
        super().__init__()
        self.config = config
        
        # Network to predict uncertainty reduction
        self.uncertainty_reduction_pred = nn.Sequential(
            nn.Linear(config.d_model + config.hypothesis_dim, config.amortization_hidden_dim),
            nn.GELU(),
            nn.Linear(config.amortization_hidden_dim, 1)
        )
        
        # Surprise history
        self.surprise_history = []
        self.epistemic_uncertainty_history = []
    
    def compute_expected_information_gain(self,
                                         current_state: torch.Tensor,
                                         hypothesis_emb: torch.Tensor) -> float:
        """
        Compute expected information gain (uncertainty reduction)
        IG = H(prior) - H(posterior esperado)
        """
        # Simplified: use network prediction
        combined = torch.cat([current_state, hypothesis_emb], dim=-1)
        
        # High value = greater expected uncertainty reduction
        info_gain = torch.sigmoid(self.uncertainty_reduction_pred(combined))
        
        return info_gain.mean().item()
    
    def select_next_experiment(self,
                              candidate_hypotheses: List[int],
                              current_state: torch.Tensor,
                              hypothesis_space: HypothesisSpace,
                              mental_simulator: MentalExperimentSimulator) -> int:
        """
        Select next experiment to maximize learning
        Uses amortization for fast decisions
        """
        expected_gains = []
        
        for h_id in candidate_hypotheses:
            h_emb = hypothesis_space.encode_hypothesis(h_id)
            
            # Expected information gain
            info_gain = self.compute_expected_information_gain(current_state, h_emb)
            
            # Expected surprise of mental experiment
            sim_result = mental_simulator.simulate_experiment(current_state, h_emb)
            surprise = sim_result['expected_surprise']
            
            # Balance information gain vs tolerable surprise
            score = info_gain - self.config.surprise_tolerance * surprise
            expected_gains.append((h_id, score))
        
        # Select hypothesis with highest score
        expected_gains.sort(key=lambda x: x[1], reverse=True)
        return expected_gains[0][0]
    
    def update_uncertainty(self, actual_surprise: float):
        """Update epistemic uncertainty history"""
        self.surprise_history.append(actual_surprise)
        
        # Calculate current epistemic uncertainty (entropy of recent surprises)
        if len(self.surprise_history) >= 10:
            recent_surprises = torch.tensor(self.surprise_history[-10:])
            # Normalizar
            recent_surprises = recent_surprises / (recent_surprises.sum() + 1e-10)
            epistemic_uncertainty = -(recent_surprises * torch.log(recent_surprises + 1e-10)).sum()
            self.epistemic_uncertainty_history.append(epistemic_uncertainty.item())
    
    def should_continue_exploration(self) -> bool:
        """Decide whether to continue exploration based on uncertainty threshold"""
        if not self.epistemic_uncertainty_history:
            return True
        
        current_uncertainty = self.epistemic_uncertainty_history[-1]
        return current_uncertainty > self.config.epistemic_uncertainty_threshold


class HypothesisNavigator(nn.Module):
    """
    Hypothesis Space Navigator
    Complete system guiding scientific discovery
    """
    
    def __init__(self, config: ActiveInferenceConfig, world_model: nn.Module):
        super().__init__()
        self.config = config
        
        # Components
        self.hypothesis_space = HypothesisSpace(config)
        self.mental_simulator = MentalExperimentSimulator(config, world_model)
        self.surprise_minimizer = EpistemicSurpriseMinimizer(config)
        
        # Candidate hypothesis generator
        self.hypothesis_generator = nn.Sequential(
            nn.Linear(config.d_model, config.amortization_hidden_dim),
            nn.GELU(),
            nn.Linear(config.amortization_hidden_dim, config.n_candidate_hypotheses * config.hypothesis_dim)
        )
        
        # Statistics
        self.experiments_conducted = 0
        self.hypotheses_tested = []
        self.discoveries = []
    
    def generate_candidate_hypotheses(self, current_state: torch.Tensor) -> List[int]:
        """
        Generate candidate hypotheses based on current state
        """
        # Use network to generate candidates
        candidates_logits = self.hypothesis_generator(current_state.mean(dim=1))
        candidates_logits = candidates_logits.view(-1, self.config.n_candidate_hypotheses, self.config.hypothesis_dim)
        
        # Select hypotheses from space closest to generated candidates
        candidate_ids = []
        for i in range(self.config.n_candidate_hypotheses):
            candidate_emb = candidates_logits[0, i]
            
            # Find closest hypothesis in space
            distances = torch.cdist(
                candidate_emb.unsqueeze(0),
                self.hypothesis_space.hypothesis_embeddings
            ).squeeze(0)
            
            closest_idx = distances.argmin().item()
            candidate_ids.append(closest_idx)
        
        return list(set(candidate_ids))  # Remover duplicados
    
    def scientific_discovery_loop(self, initial_state: torch.Tensor, 
                                 max_iterations: int = 100) -> Dict:
        """
        Full scientific discovery loop
        
        1. Generate candidate hypotheses
        2. Select optimal experiment
        3. Simular mentalmente
        4. Actualizar conocimiento
        5. Repetir hasta convergencia
        """
        current_state = initial_state
        iteration = 0
        
        print("Starting scientific discovery loop...")
        
        while iteration < max_iterations and self.surprise_minimizer.should_continue_exploration():
            # 1. Generar candidatas
            candidates = self.generate_candidate_hypotheses(current_state)
            
            # 2. Seleccionar mejor experimento
            selected_hypothesis = self.surprise_minimizer.select_next_experiment(
                candidates, current_state, self.hypothesis_space, self.mental_simulator
            )
            
            # 3. Simular mentalmente
            h_emb = self.hypothesis_space.encode_hypothesis(selected_hypothesis)
            sim_result = self.mental_simulator.simulate_experiment(current_state, h_emb)
            
            # 4. Simulate real outcome (in real implementation, this would be a physical experiment)
            # Here we simulate with noise
            actual_outcome = sim_result['predicted_outcome'] + torch.randn_like(sim_result['predicted_outcome']) * 0.1
            
            # 5. Calcular sorpresa real
            prediction_error = F.mse_loss(sim_result['predicted_outcome'], actual_outcome)
            self.surprise_minimizer.update_uncertainty(prediction_error.item())
            
            # 6. Actualizar estado
            current_state = actual_outcome.unsqueeze(1) if actual_outcome.dim() == 2 else actual_outcome
            
            # 7. Registrar
            self.experiments_conducted += 1
            self.hypotheses_tested.append({
                'iteration': iteration,
                'hypothesis_id': selected_hypothesis,
                'predicted_surprise': sim_result['expected_surprise'],
                'actual_error': prediction_error.item()
            })
            
            # Check if "discovery" (low surprise, correct prediction)
            if prediction_error.item() < 0.05:
                self.discoveries.append({
                    'iteration': iteration,
                    'hypothesis_id': selected_hypothesis,
                    'confidence': 1 - prediction_error.item()
                })
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: {len(self.discoveries)} discoveries, "
                      f"epistemic uncertainty: {self.surprise_minimizer.epistemic_uncertainty_history[-1]:.4f}")
        
        return {
            'iterations': iteration,
            'experiments': self.experiments_conducted,
            'discoveries': len(self.discoveries),
            'hypotheses_tested': self.hypotheses_tested,
            'final_uncertainty': self.surprise_minimizer.epistemic_uncertainty_history[-1] if self.surprise_minimizer.epistemic_uncertainty_history else None
        }
    
    def get_discoveries_summary(self) -> Dict:
        """Summary of scientific discoveries"""
        return {
            'total_discoveries': len(self.discoveries),
            'total_experiments': self.experiments_conducted,
            'efficiency': len(self.discoveries) / max(self.experiments_conducted, 1),
            'high_confidence_discoveries': sum(1 for d in self.discoveries if d['confidence'] > 0.9),
            'discovery_rate': len(self.discoveries) / max(len(self.hypotheses_tested), 1)
        }


class AmortizedActiveInference(nn.Module):
    """
    Complete Amortized Active Inference system
    Integrates all components for efficient scientific discovery
    """
    
    def __init__(self, config: ActiveInferenceConfig, world_model: nn.Module):
        super().__init__()
        self.config = config
        self.navigator = HypothesisNavigator(config, world_model)
        
        # Amortization: network that learns to predict hypothesis quality
        self.hypothesis_quality_predictor = nn.Sequential(
            nn.Linear(config.hypothesis_dim, config.amortization_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.amortization_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def amortized_hypothesis_quality(self, hypothesis_id: int) -> float:
        """
        Fast (amortized) prediction of hypothesis quality
        Avoids simulating the entire experiment
        """
        h_emb = self.navigator.hypothesis_space.hypothesis_embeddings[hypothesis_id]
        quality = self.hypothesis_quality_predictor(h_emb)
        return quality.item()
    
    def forward(self, initial_state: torch.Tensor, mode: str = 'discovery') -> Dict:
        """
        Main forward
        
        Args:
            initial_state: Initial system state
            mode: 'discovery' (exploration) or 'exploitation' (use of knowledge)
        """
        if mode == 'discovery':
            # Discovery mode: explore hypothesis space
            return self.navigator.scientific_discovery_loop(initial_state)
        else:
            # Exploitation mode: use acquired knowledge
            # Select best known hypothesis
            if self.navigator.discoveries:
                best_discovery = max(self.navigator.discoveries, key=lambda x: x['confidence'])
                return {
                    'best_hypothesis': best_discovery['hypothesis_id'],
                    'confidence': best_discovery['confidence'],
                    'mode': 'exploitation'
                }
            else:
                return {'message': 'No discoveries yet, run discovery mode first'}
    
    def get_stats(self) -> Dict:
        """System statistics"""
        return {
            **self.navigator.get_discoveries_summary(),
            'amortization_active': True,
            'epistemic_uncertainty_threshold': self.config.epistemic_uncertainty_threshold
        }
