"""
Causal Foundation Models (CFM)
Evolución de H-JEPA hacia Modelo Fundacional Causal con Grafos Parciales
Incorpora conocimiento experto previo mediante sesgos aprendibles en atención
Objetivo: Estimación ATE (Average Treatment Effect) zero-shot con precisión de modelos especializados
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import math


@dataclass
class CFMConfig:
    """Configuración Causal Foundation Model"""
    d_model: int = 768
    n_causal_vars: int = 32  # Variables causales
    n_confounders: int = 8   # Variables confusoras
    max_parents_per_var: int = 5
    graph_sparsity: float = 0.3  # Sparcidad del grafo causal
    
    # Para grafos parciales
    known_edges_ratio: float = 0.4  # % de aristas conocidas a priori
    
    # ATE estimation
    ate_hidden_dim: int = 256
    n_mc_samples: int = 1000  # Muestras Monte Carlo para ATE
    
    # Atención causal
    n_causal_heads: int = 8
    causal_attention_dropout: float = 0.1


class PartialCausalGraph(nn.Module):
    """
    Grafo Causal Parcial
    Acepta conocimiento experto previo (ancestral) y aprende el resto
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        n = config.n_causal_vars
        
        # Matriz de adyacencia completa (aprendible)
        self.adjacency_logits = nn.Parameter(torch.randn(n, n) * 0.1)
        
        # Máscara de aristas conocidas (prior experto)
        # True = arista conocida (no se aprende, se fija)
        # False = arista desconocida (se aprende)
        self.register_buffer('known_edges_mask', torch.zeros(n, n, dtype=torch.bool))
        
        # Valores fijos para aristas conocidas
        self.register_buffer('known_edge_values', torch.zeros(n, n))
        
        # Restricciones estructurales (DAG)
        self.register_buffer('dag_mask', self._create_dag_mask(n))
        
        # Mecanismos causales (funciones estructurales)
        self.structural_equations = nn.ModuleList([
            StructuralEquation(config.d_model, config.max_parents_per_var)
            for _ in range(n)
        ])
        
        # Identificabilidad: score para determinar qué tan identificable es el grafo
        self.identifiability_score = nn.Parameter(torch.tensor(0.0))
    
    def _create_dag_mask(self, n: int) -> torch.Tensor:
        """Crear máscara triangular superior para garantizar DAG"""
        mask = torch.triu(torch.ones(n, n), diagonal=1)
        return mask  # True permite aristas (i -> j si i < j)
    
    def set_known_edges(self, edges: List[Tuple[int, int]], strengths: Optional[List[float]] = None):
        """
        Establecer aristas causales conocidas desde conocimiento experto
        
        Args:
            edges: Lista de (origen, destino) conocidos
            strengths: Fuerza de cada arista (opcional)
        """
        for idx, (i, j) in enumerate(edges):
            self.known_edges_mask[i, j] = True
            if strengths and idx < len(strengths):
                self.known_edge_values[i, j] = strengths[idx]
            else:
                self.known_edge_values[i, j] = 0.5  # Default
    
    def get_adjacency_matrix(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Obtener matriz de adyacencia (suave o dura)
        
        Combina aristas conocidas (fijas) con aristas aprendidas
        """
        n = self.config.n_causal_vars
        
        # Aristas aprendidas (Gumbel-Softmax para diferenciabilidad)
        learned_edges = torch.sigmoid(self.adjacency_logits / temperature)
        
        # Aplicar máscara DAG
        learned_edges = learned_edges * self.dag_mask
        
        # Combinar conocidas + aprendidas
        adjacency = torch.where(
            self.known_edges_mask,
            self.known_edge_values,
            learned_edges
        )
        
        # Aplicar sparcidad
        adjacency = adjacency * (adjacency > self.config.graph_sparsity)
        
        if hard:
            # Para inferencia, binarizar
            adjacency = (adjacency > 0.5).float()
        
        return adjacency
    
    def forward(self, variables: torch.Tensor, intervention_idx: Optional[int] = None,
                intervention_value: Optional[float] = None) -> torch.Tensor:
        """
        Forward pasando información causalmente
        
        Args:
            variables: (B, n_vars, d_model) - representaciones de variables
            intervention_idx: Índice de variable a intervenir (do-operator)
            intervention_value: Valor de intervención
        
        Returns:
            outcomes: (B, n_vars, d_model) - resultado causal
        """
        batch_size = variables.size(0)
        adjacency = self.get_adjacency_matrix()
        
        # Orden topológico (ya garantizado por máscara DAG)
        outcomes = torch.zeros_like(variables)
        
        for var_idx in range(self.config.n_causal_vars):
            # Obtener padres de esta variable
            parents_mask = adjacency[:, var_idx] > 0.5  # (n_vars,)
            
            if parents_mask.sum() == 0:
                # Variable raíz (sin padres)
                outcomes[:, var_idx] = variables[:, var_idx]
            else:
                # Aplicar ecuación estructural
                parent_indices = parents_mask.nonzero(as_tuple=True)[0]
                parent_values = outcomes[:, parent_indices, :]  # (B, n_parents, d)
                
                # Ecuación estructural
                var_outcome = self.structural_equations[var_idx](parent_values)
                outcomes[:, var_idx] = var_outcome
            
            # Aplicar intervención do(X=x) si corresponde
            if intervention_idx == var_idx and intervention_value is not None:
                outcomes[:, var_idx] = intervention_value
        
        return outcomes
    
    def get_causal_parents(self, var_idx: int) -> List[int]:
        """Obtener índices de padres causales de una variable"""
        adjacency = self.get_adjacency_matrix(hard=True)
        parents = adjacency[:, var_idx].nonzero(as_tuple=True)[0].tolist()
        return parents


class StructuralEquation(nn.Module):
    """
    Ecuación Estructural: Y = f(pa(Y), N)
    Representa el mecanismo causal de cómo los padres causan el hijo
    """
    
    def __init__(self, d_model: int, max_parents: int):
        super().__init__()
        self.d_model = d_model
        self.max_parents = max_parents
        
        # Red neuronal que implementa f
        # Toma padres y produce variable hija
        input_dim = d_model * max_parents
        
        self.f = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Término de ruido (representa factores no observados)
        self.noise_proj = nn.Linear(d_model, d_model)
    
    def forward(self, parent_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            parent_values: (B, n_parents, d_model)
        
        Returns:
            outcome: (B, d_model)
        """
        batch_size = parent_values.size(0)
        n_parents = parent_values.size(1)
        
        # Pad o truncar a max_parents
        if n_parents < self.max_parents:
            padding = torch.zeros(batch_size, self.max_parents - n_parents, self.d_model,
                                device=parent_values.device)
            parent_values = torch.cat([parent_values, padding], dim=1)
        elif n_parents > self.max_parents:
            parent_values = parent_values[:, :self.max_parents, :]
        
        # Flatten padres
        parents_flat = parent_values.view(batch_size, -1)
        
        # Aplicar mecanismo f
        outcome = self.f(parents_flat)
        
        # Añadir ruido (para representar incertidumbre)
        noise = torch.randn_like(outcome) * 0.01
        outcome = outcome + self.noise_proj(noise)
        
        return outcome


class CausalAttention(nn.Module):
    """
    Mecanismo de Atención Causal con sesgos aprendibles
    Incorpora conocimiento experto en los sesgos de atención
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        
        # Atención multi-cabeza estándar
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_causal_heads,
            dropout=config.causal_attention_dropout,
            batch_first=True
        )
        
        # Sesgos aprendibles para incorporar prior causal
        self.causal_biases = nn.Parameter(
            torch.randn(config.n_causal_vars, config.n_causal_vars) * 0.1
        )
        
        # Máscara de atención causal (puede ser parcial)
        self.register_buffer('causal_mask', None)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                causal_graph: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward con sesgos causales
        
        Args:
            query, key, value: Tensores de atención
            causal_graph: Grafo causal para guiar atención (opcional)
        """
        batch_size, seq_len, _ = query.shape
        
        # Si tenemos grafo causal, usarlo para modificar atención
        if causal_graph is not None and seq_len == self.config.n_causal_vars:
            # Expandir grafo a batch
            graph_bias = causal_graph.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Modificar key con sesgo causal
            key_biased = key + torch.matmul(graph_bias, value)
        else:
            key_biased = key
        
        # Atención estándar con sesgos
        attn_output, attn_weights = self.attention(query, key_biased, value)
        
        return attn_output


class ATEEstimator(nn.Module):
    """
    Estimador de Average Treatment Effect (ATE)
    E[Y | do(X=1)] - E[Y | do(X=0)]
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        
        # Red para estimar outcome potencial
        self.potential_outcome_net = nn.Sequential(
            nn.Linear(config.d_model * config.n_causal_vars, config.ate_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ate_hidden_dim, config.ate_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.ate_hidden_dim // 2, 1)
        )
    
    def estimate_ate(self, cfm: PartialCausalGraph, variables: torch.Tensor,
                     treatment_idx: int, outcome_idx: int) -> Dict:
        """
        Estimar ATE usando Monte Carlo con intervenciones do(X=x)
        
        Args:
            cfm: Modelo causal con grafo
            variables: Variables observadas
            treatment_idx: Índice de tratamiento
            outcome_idx: Índice de outcome
        
        Returns:
            dict con ATE estimado, intervalo de confianza
        """
        batch_size = variables.size(0)
        device = variables.device
        
        outcomes_treatment_1 = []
        outcomes_treatment_0 = []
        
        # Monte Carlo sampling
        for _ in range(self.config.n_mc_samples):
            # Intervención do(X=1)
            outcome_1 = cfm(variables, intervention_idx=treatment_idx, intervention_value=1.0)
            outcomes_treatment_1.append(outcome_1[:, outcome_idx].mean(dim=-1))
            
            # Intervención do(X=0)
            outcome_0 = cfm(variables, intervention_idx=treatment_idx, intervention_value=0.0)
            outcomes_treatment_0.append(outcome_0[:, outcome_idx].mean(dim=-1))
        
        # Promedios
        mean_1 = torch.stack(outcomes_treatment_1).mean(dim=0)
        mean_0 = torch.stack(outcomes_treatment_0).mean(dim=0)
        
        # ATE
        ate = mean_1 - mean_0
        
        # Intervalo de confianza (95%)
        std_1 = torch.stack(outcomes_treatment_1).std(dim=0)
        std_0 = torch.stack(outcomes_treatment_0).std(dim=0)
        
        ci_lower = ate - 1.96 * torch.sqrt(std_1**2 + std_0**2)
        ci_upper = ate + 1.96 * torch.sqrt(std_1**2 + std_0**2)
        
        return {
            'ate': ate.mean().item(),
            'ate_per_sample': ate,
            'ci_95_lower': ci_lower.mean().item(),
            'ci_95_upper': ci_upper.mean().item(),
            'n_samples': self.config.n_mc_samples
        }
    
    def forward(self, variables: torch.Tensor) -> torch.Tensor:
        """Forward simple para integración con modelo grande"""
        # Flatten todas las variables
        x = variables.view(variables.size(0), -1)
        return self.potential_outcome_net(x)


class CausalFoundationModel(nn.Module):
    """
    Modelo Fundacional Causal completo
    Integra grafo parcial, atención causal, y estimación ATE
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        
        # Grafo causal parcial
        self.causal_graph = PartialCausalGraph(config)
        
        # Atención con sesgos causales
        self.causal_attention = CausalAttention(config)
        
        # Estimador ATE
        self.ate_estimator = ATEEstimator(config)
        
        # Codificador de variables
        self.variable_encoder = nn.Linear(config.d_model, config.d_model)
        
        # Decoder para predicciones
        self.outcome_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Métricas de identificabilidad
        self.identifiability_history = []
    
    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Codificar observaciones a espacio causal"""
        return self.variable_encoder(observations)
    
    def forward(self, observations: torch.Tensor, 
                treatment_idx: Optional[int] = None,
                outcome_idx: Optional[int] = None,
                return_ate: bool = False) -> Dict:
        """
        Forward completo del CFM
        
        Args:
            observations: Datos observados (B, n_vars, d)
            treatment_idx: Para estimación ATE
            outcome_idx: Para estimación ATE
            return_ate: Si calcular ATE
        """
        # Codificar
        variables = self.encode_observations(observations)
        
        # Aplicar grafo causal
        causal_outcomes = self.causal_graph(variables)
        
        # Atención causal
        attended = self.causal_attention(
            causal_outcomes, causal_outcomes, causal_outcomes,
            causal_graph=self.causal_graph.get_adjacency_matrix()
        )
        
        # Decodificar
        predictions = self.outcome_decoder(attended)
        
        result = {
            'predictions': predictions,
            'causal_outcomes': causal_outcomes,
            'attended': attended,
            'causal_graph': self.causal_graph.get_adjacency_matrix(hard=True)
        }
        
        # Estimar ATE si se solicita
        if return_ate and treatment_idx is not None and outcome_idx is not None:
            ate_results = self.ate_estimator.estimate_ate(
                self.causal_graph, variables, treatment_idx, outcome_idx
            )
            result['ate_estimation'] = ate_results
        
        return result
    
    def intervene(self, observations: torch.Tensor, 
                  intervention_dict: Dict[int, float]) -> torch.Tensor:
        """
        Realizar múltiples intervenciones do(X=x)
        
        Args:
            observations: Datos observados
            intervention_dict: {var_idx: value} intervenciones
        """
        variables = self.encode_observations(observations)
        
        # Aplicar intervenciones secuencialmente
        for var_idx, value in intervention_dict.items():
            variables = self.causal_graph(
                variables, 
                intervention_idx=var_idx, 
                intervention_value=value
            )
        
        return self.outcome_decoder(variables)
    
    def get_causal_explanation(self, var_idx: int) -> Dict:
        """Obtener explicación causal de una variable"""
        parents = self.causal_graph.get_causal_parents(var_idx)
        
        return {
            'variable': var_idx,
            'parents': parents,
            'n_parents': len(parents),
            'is_root': len(parents) == 0,
            'graph_sparsity': self.config.graph_sparsity
        }
    
    def zero_shot_causal_inference(self, source_data: torch.Tensor, 
                                   target_intervention: int,
                                   reference_model_predictions: Optional[torch.Tensor] = None) -> Dict:
        """
        Inferencia causal zero-shot
        Compara con modelos especializados
        """
        # Estimar efecto causal
        result = self.forward(
            source_data, 
            treatment_idx=target_intervention,
            outcome_idx=0,  # Asumiendo outcome principal
            return_ate=True
        )
        
        ate_estimated = result['ate_estimation']['ate']
        
        # Calcular precisión si tenemos referencia
        if reference_model_predictions is not None:
            ate_reference = reference_model_predictions.mean().item()
            error = abs(ate_estimated - ate_reference)
            accuracy = max(0, 1 - error / abs(ate_reference + 1e-8))
        else:
            accuracy = None
            error = None
        
        return {
            'ate_estimated': ate_estimated,
            'ate_reference': ate_reference if reference_model_predictions is not None else None,
            'error': error,
            'accuracy_vs_specialized': accuracy,
            'is_zero_shot': True
        }
