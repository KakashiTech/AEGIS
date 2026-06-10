"""
Inferencia Activa Amortizada para Descubrimiento Científico
Sistema que genera hipótesis y realiza "experimentos mentales"
Navegador de Espacio de Hipótesis para minimizar sorpresa epistémica
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
    """Configuración Inferencia Activa"""
    d_model: int = 768
    n_hypothesis_space: int = 1000  # Tamaño del espacio de hipótesis
    epistemic_uncertainty_threshold: float = 0.3
    
    # Generación de hipótesis
    n_candidate_hypotheses: int = 10
    hypothesis_dim: int = 256
    
    # Experimentos mentales
    n_mental_rollouts: int = 20
    rollout_horizon: int = 10
    
    # Amortización
    amortization_hidden_dim: int = 512
    surprise_tolerance: float = 0.1


class HypothesisSpace(nn.Module):
    """
    Espacio de Hipótesis latente
    Cada hipótesis es un punto en el espacio latente que representa una teoría
    """
    
    def __init__(self, config: ActiveInferenceConfig):
        super().__init__()
        self.config = config
        
        # Embeddings de hipótesis
        self.hypothesis_embeddings = nn.Parameter(
            torch.randn(config.n_hypothesis_space, config.hypothesis_dim)
        )
        
        # Prior de hipótesis (qué tan probable es cada una a priori)
        self.hypothesis_priors = nn.Parameter(
            torch.ones(config.n_hypothesis_space) / config.n_hypothesis_space
        )
        
        # Mapeo a espacio del modelo
        self.to_model_space = nn.Sequential(
            nn.Linear(config.hypothesis_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU()
        )
        
        # Métrica de distancia en espacio de hipótesis
        self.distance_metric = nn.Parameter(torch.eye(config.hypothesis_dim))
    
    def encode_hypothesis(self, hypothesis_id: int) -> torch.Tensor:
        """Obtener embedding de hipótesis"""
        h_emb = self.hypothesis_embeddings[hypothesis_id]
        return self.to_model_space(h_emb)
    
    def compute_epistemic_distance(self, h1: int, h2: int) -> float:
        """Distancia epistémica entre dos hipótesis"""
        diff = self.hypothesis_embeddings[h1] - self.hypothesis_embeddings[h2]
        distance = torch.sqrt(diff @ self.distance_metric @ diff)
        return distance.item()
    
    def sample_hypothesis(self, temperature: float = 1.0) -> int:
        """Muestrear hipótesis según priores"""
        probs = F.softmax(self.hypothesis_priors / temperature, dim=0)
        return torch.multinomial(probs, 1).item()
    
    def get_high_uncertainty_regions(self, n_regions: int = 5) -> List[int]:
        """Identificar regiones de alta incertidumbre epistémica"""
        # Calcular entropía de cada hipótesis con respecto a vecinos
        uncertainties = []
        
        for i in range(self.config.n_hypothesis_space):
            # Distancias a todas las demás
            distances = []
            for j in range(self.config.n_hypothesis_space):
                if i != j:
                    dist = self.compute_epistemic_distance(i, j)
                    distances.append(dist)
            
            # Varianza de distancias como medida de incertidumbre local
            uncertainty = torch.tensor(distances).std().item()
            uncertainties.append((i, uncertainty))
        
        # Ordenar por incertidumbre y retornar top-n
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in uncertainties[:n_regions]]


class MentalExperimentSimulator(nn.Module):
    """
    Simulador de "Experimentos Mentales"
    Predice resultados de experimentos virtuales sin ejecutarlos físicamente
    """
    
    def __init__(self, config: ActiveInferenceConfig, world_model: nn.Module):
        super().__init__()
        self.config = config
        self.world_model = world_model  # Modelo del mundo (H-JEPA o similar)
        
        # Generador de resultados experimentales
        self.outcome_predictor = nn.Sequential(
            nn.Linear(config.d_model + config.hypothesis_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
        # Estimador de varianza (incertidumbre en predicción)
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
        Simular un experimento mental
        
        Args:
            current_state: Estado actual del sistema
            hypothesis_embedding: Hipótesis a testear
            intervention: Intervención experimental (opcional)
        
        Returns:
            dict con outcome predicho, incertidumbre, sorpresa esperada
        """
        batch_size = current_state.size(0)
        
        # Combinar estado con hipótesis
        combined = torch.cat([current_state, hypothesis_embedding.expand(batch_size, -1)], dim=-1)
        
        # Predecir outcome
        predicted_outcome = self.outcome_predictor(combined)
        
        # Estimar incertidumbre
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
        
        # Calcular sorpresa esperada (divergencia de predicciones)
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
        """Simular múltiples hipótesis en paralelo"""
        results = []
        
        for h_id in hypotheses:
            h_emb = hypothesis_space.encode_hypothesis(h_id)
            result = self.simulate_experiment(current_state, h_emb)
            result['hypothesis_id'] = h_id
            results.append(result)
        
        return results


class EpistemicSurpriseMinimizer(nn.Module):
    """
    Minimizador de Sorpresa Epistémica
    Dirige exploración hacia regiones que maximizan reducción de incertidumbre
    """
    
    def __init__(self, config: ActiveInferenceConfig):
        super().__init__()
        self.config = config
        
        # Red para predecir reducción de incertidumbre
        self.uncertainty_reduction_pred = nn.Sequential(
            nn.Linear(config.d_model + config.hypothesis_dim, config.amortization_hidden_dim),
            nn.GELU(),
            nn.Linear(config.amortization_hidden_dim, 1)
        )
        
        # Historial de sorpresas
        self.surprise_history = []
        self.epistemic_uncertainty_history = []
    
    def compute_expected_information_gain(self,
                                         current_state: torch.Tensor,
                                         hypothesis_emb: torch.Tensor) -> float:
        """
        Calcular ganancia de información esperada (reducción de incertidumbre)
        IG = H(prior) - H(posterior esperado)
        """
        # Simplificación: usar predicción de red
        combined = torch.cat([current_state, hypothesis_emb], dim=-1)
        
        # Valor alto = mayor reducción esperada de incertidumbre
        info_gain = torch.sigmoid(self.uncertainty_reduction_pred(combined))
        
        return info_gain.mean().item()
    
    def select_next_experiment(self,
                              candidate_hypotheses: List[int],
                              current_state: torch.Tensor,
                              hypothesis_space: HypothesisSpace,
                              mental_simulator: MentalExperimentSimulator) -> int:
        """
        Seleccionar siguiente experimento para maximizar aprendizaje
        Usa amortización para decidir rápidamente
        """
        expected_gains = []
        
        for h_id in candidate_hypotheses:
            h_emb = hypothesis_space.encode_hypothesis(h_id)
            
            # Ganancia de información esperada
            info_gain = self.compute_expected_information_gain(current_state, h_emb)
            
            # Sorpresa esperada del experimento mental
            sim_result = mental_simulator.simulate_experiment(current_state, h_emb)
            surprise = sim_result['expected_surprise']
            
            # Balancear ganancia de información vs sorpresa tolerable
            score = info_gain - self.config.surprise_tolerance * surprise
            expected_gains.append((h_id, score))
        
        # Seleccionar hipótesis con mayor score
        expected_gains.sort(key=lambda x: x[1], reverse=True)
        return expected_gains[0][0]
    
    def update_uncertainty(self, actual_surprise: float):
        """Actualizar historial de incertidumbre epistémica"""
        self.surprise_history.append(actual_surprise)
        
        # Calcular incertidumbre epistémica actual (entropía de sorpresas recientes)
        if len(self.surprise_history) >= 10:
            recent_surprises = torch.tensor(self.surprise_history[-10:])
            # Normalizar
            recent_surprises = recent_surprises / (recent_surprises.sum() + 1e-10)
            epistemic_uncertainty = -(recent_surprises * torch.log(recent_surprises + 1e-10)).sum()
            self.epistemic_uncertainty_history.append(epistemic_uncertainty.item())
    
    def should_continue_exploration(self) -> bool:
        """Decidir si continuar exploración basado en umbral de incertidumbre"""
        if not self.epistemic_uncertainty_history:
            return True
        
        current_uncertainty = self.epistemic_uncertainty_history[-1]
        return current_uncertainty > self.config.epistemic_uncertainty_threshold


class HypothesisNavigator(nn.Module):
    """
    Navegador de Espacio de Hipótesis
    Sistema completo que guía el descubrimiento científico
    """
    
    def __init__(self, config: ActiveInferenceConfig, world_model: nn.Module):
        super().__init__()
        self.config = config
        
        # Componentes
        self.hypothesis_space = HypothesisSpace(config)
        self.mental_simulator = MentalExperimentSimulator(config, world_model)
        self.surprise_minimizer = EpistemicSurpriseMinimizer(config)
        
        # Generador de hipótesis candidatas
        self.hypothesis_generator = nn.Sequential(
            nn.Linear(config.d_model, config.amortization_hidden_dim),
            nn.GELU(),
            nn.Linear(config.amortization_hidden_dim, config.n_candidate_hypotheses * config.hypothesis_dim)
        )
        
        # Estadísticas
        self.experiments_conducted = 0
        self.hypotheses_tested = []
        self.discoveries = []
    
    def generate_candidate_hypotheses(self, current_state: torch.Tensor) -> List[int]:
        """
        Generar hipótesis candidatas basadas en estado actual
        """
        # Usar red para generar candidatas
        candidates_logits = self.hypothesis_generator(current_state.mean(dim=1))
        candidates_logits = candidates_logits.view(-1, self.config.n_candidate_hypotheses, self.config.hypothesis_dim)
        
        # Seleccionar hipótesis del espacio más cercanas a los candidatos generados
        candidate_ids = []
        for i in range(self.config.n_candidate_hypotheses):
            candidate_emb = candidates_logits[0, i]
            
            # Encontrar hipótesis más cercana en espacio
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
        Loop completo de descubrimiento científico
        
        1. Generar hipótesis candidatas
        2. Seleccionar experimento óptimo
        3. Simular mentalmente
        4. Actualizar conocimiento
        5. Repetir hasta convergencia
        """
        current_state = initial_state
        iteration = 0
        
        print("Iniciando loop de descubrimiento científico...")
        
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
            
            # 4. Simular outcome real (en implementación real, esto sería un experimento físico)
            # Aquí simulamos con ruido
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
            
            # Verificar si es "descubrimiento" (sorpresa baja, predicción acertada)
            if prediction_error.item() < 0.05:
                self.discoveries.append({
                    'iteration': iteration,
                    'hypothesis_id': selected_hypothesis,
                    'confidence': 1 - prediction_error.item()
                })
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"  Iteración {iteration}: {len(self.discoveries)} descubrimientos, "
                      f"incertidumbre epistémica: {self.surprise_minimizer.epistemic_uncertainty_history[-1]:.4f}")
        
        return {
            'iterations': iteration,
            'experiments': self.experiments_conducted,
            'discoveries': len(self.discoveries),
            'hypotheses_tested': self.hypotheses_tested,
            'final_uncertainty': self.surprise_minimizer.epistemic_uncertainty_history[-1] if self.surprise_minimizer.epistemic_uncertainty_history else None
        }
    
    def get_discoveries_summary(self) -> Dict:
        """Resumen de descubrimientos científicos"""
        return {
            'total_discoveries': len(self.discoveries),
            'total_experiments': self.experiments_conducted,
            'efficiency': len(self.discoveries) / max(self.experiments_conducted, 1),
            'high_confidence_discoveries': sum(1 for d in self.discoveries if d['confidence'] > 0.9),
            'discovery_rate': len(self.discoveries) / max(len(self.hypotheses_tested), 1)
        }


class AmortizedActiveInference(nn.Module):
    """
    Sistema de Inferencia Activa Amortizada completo
    Integra todos los componentes para descubrimiento científico eficiente
    """
    
    def __init__(self, config: ActiveInferenceConfig, world_model: nn.Module):
        super().__init__()
        self.config = config
        self.navigator = HypothesisNavigator(config, world_model)
        
        # Amortización: red que aprende a predecir qué tan buena es una hipótesis
        self.hypothesis_quality_predictor = nn.Sequential(
            nn.Linear(config.hypothesis_dim, config.amortization_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.amortization_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def amortized_hypothesis_quality(self, hypothesis_id: int) -> float:
        """
        Predicción rápida (amortizada) de calidad de hipótesis
        Evita simular todo el experimento
        """
        h_emb = self.navigator.hypothesis_space.hypothesis_embeddings[hypothesis_id]
        quality = self.hypothesis_quality_predictor(h_emb)
        return quality.item()
    
    def forward(self, initial_state: torch.Tensor, mode: str = 'discovery') -> Dict:
        """
        Forward principal
        
        Args:
            initial_state: Estado inicial del sistema
            mode: 'discovery' (exploración) o 'exploitation' (uso de conocimiento)
        """
        if mode == 'discovery':
            # Modo descubrimiento: explorar espacio de hipótesis
            return self.navigator.scientific_discovery_loop(initial_state)
        else:
            # Modo explotación: usar conocimiento adquirido
            # Seleccionar mejor hipótesis conocida
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
        """Estadísticas del sistema"""
        return {
            **self.navigator.get_discoveries_summary(),
            'amortization_active': True,
            'epistemic_uncertainty_threshold': self.config.epistemic_uncertainty_threshold
        }
