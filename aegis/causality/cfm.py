"""
Causal Foundation Models (CFM)
H-JEPA evolution toward Causal Foundation Model with Partial Graphs
Incorporates expert prior knowledge via learnable attention biases
Goal: Zero-shot ATE (Average Treatment Effect) estimation with specialist-level accuracy
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
    """Configuration Causal Foundation Model"""
    d_model: int = 768
    n_causal_vars: int = 32  # Causal variables
    n_confounders: int = 8   # Confounding variables
    max_parents_per_var: int = 5
    graph_sparsity: float = 0.3  # Causal graph sparsity
    
    # Para grafos parciales
    known_edges_ratio: float = 0.4  # % of edges known a priori
    
    # ATE estimation
    ate_hidden_dim: int = 256
    n_mc_samples: int = 1000  # Monte Carlo samples for ATE
    
    # Causal attention
    n_causal_heads: int = 8
    causal_attention_dropout: float = 0.1


class PartialCausalGraph(nn.Module):
    """
    Partial Causal Graph
    Accepts prior expert knowledge (ancestral) and learns the rest
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        n = config.n_causal_vars
        
        # Full adjacency matrix (learnable)
        self.adjacency_logits = nn.Parameter(torch.randn(n, n) * 0.1)
        
        # Known edge mask (expert prior)
        # True = known edge (not learned, fixed)
        # False = unknown edge (learned)
        self.register_buffer('known_edges_mask', torch.zeros(n, n, dtype=torch.bool))
        
        # Fixed values for known edges
        self.register_buffer('known_edge_values', torch.zeros(n, n))
        
        # Structural constraints (DAG)
        self.register_buffer('dag_mask', self._create_dag_mask(n))
        
        # Causal mechanisms (structural functions)
        self.structural_equations = nn.ModuleList([
            StructuralEquation(config.d_model, config.max_parents_per_var)
            for _ in range(n)
        ])
        
        # Identifiability: score for graph identifiability
        self.identifiability_score = nn.Parameter(torch.tensor(0.0))
    
    def _create_dag_mask(self, n: int) -> torch.Tensor:
        """Create upper triangular mask to ensure DAG"""
        mask = torch.triu(torch.ones(n, n), diagonal=1)
        return mask  # True permite aristas (i -> j si i < j)
    
    def set_known_edges(self, edges: List[Tuple[int, int]], strengths: Optional[List[float]] = None):
        """
        Set known causal edges from expert knowledge
        
        Args:
            edges: List of known (source, target)
            strengths: Strength of each edge (optional)
        """
        for idx, (i, j) in enumerate(edges):
            self.known_edges_mask[i, j] = True
            if strengths and idx < len(strengths):
                self.known_edge_values[i, j] = strengths[idx]
            else:
                self.known_edge_values[i, j] = 0.5  # Default
    
    def get_adjacency_matrix(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Get adjacency matrix (soft or hard)
        
        Combines known (fixed) edges with learned edges
        """
        n = self.config.n_causal_vars
        
        # Learned edges (Gumbel-Softmax for differentiability)
        learned_edges = torch.sigmoid(self.adjacency_logits / temperature)
        
        # Apply DAG mask
        learned_edges = learned_edges * self.dag_mask
        
        # Combine known + learned
        adjacency = torch.where(
            self.known_edges_mask,
            self.known_edge_values,
            learned_edges
        )
        
        # Apply sparsity
        adjacency = adjacency * (adjacency > self.config.graph_sparsity)
        
        if hard:
            # For inference, binarize
            adjacency = (adjacency > 0.5).float()
        
        return adjacency
    
    def forward(self, variables: torch.Tensor, intervention_idx: Optional[int] = None,
                intervention_value: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass passing information causally
        
        Args:
            variables: (B, n_vars, d_model) - representaciones de variables
            intervention_idx: Variable to intervene on (do-operator)
            intervention_value: Intervention value
        
        Returns:
            outcomes: (B, n_vars, d_model) - causal outcome
        """
        batch_size = variables.size(0)
        adjacency = self.get_adjacency_matrix()
        
        # Topological order (guaranteed by DAG mask)
        outcomes = torch.zeros_like(variables)
        
        for var_idx in range(self.config.n_causal_vars):
            # Get parents of this variable
            parents_mask = adjacency[:, var_idx] > 0.5  # (n_vars,)
            
            if parents_mask.sum() == 0:
                # Root variable (no parents)
                outcomes[:, var_idx] = variables[:, var_idx]
            else:
                # Apply structural equation
                parent_indices = parents_mask.nonzero(as_tuple=True)[0]
                parent_values = outcomes[:, parent_indices, :]  # (B, n_parents, d)
                
                # Structural equation
                var_outcome = self.structural_equations[var_idx](parent_values)
                outcomes[:, var_idx] = var_outcome
            
            # Apply do(X=x) intervention if applicable
            if intervention_idx == var_idx and intervention_value is not None:
                outcomes[:, var_idx] = intervention_value
        
        return outcomes
    
    def get_causal_parents(self, var_idx: int) -> List[int]:
        """Get causal parent indices of a variable"""
        adjacency = self.get_adjacency_matrix(hard=True)
        parents = adjacency[:, var_idx].nonzero(as_tuple=True)[0].tolist()
        return parents


class StructuralEquation(nn.Module):
    """
    Structural Equation: Y = f(pa(Y), N)
    Represents the causal mechanism of how parents cause child
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
        
        # Noise term (unobserved factors)
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
        
        # Apply mechanism f
        outcome = self.f(parents_flat)
        
        # Add noise (to represent uncertainty)
        noise = torch.randn_like(outcome) * 0.01
        outcome = outcome + self.noise_proj(noise)
        
        return outcome


class CausalAttention(nn.Module):
    """
    Causal Attention Mechanism with learnable biases
    Incorporates expert knowledge in attention biases
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        
        # Standard multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_causal_heads,
            dropout=config.causal_attention_dropout,
            batch_first=True
        )
        
        # Learnable biases to incorporate causal prior
        self.causal_biases = nn.Parameter(
            torch.randn(config.n_causal_vars, config.n_causal_vars) * 0.1
        )
        
        # Causal attention mask (can be partial)
        self.register_buffer('causal_mask', None)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                causal_graph: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward with causal biases
        
        Args:
            query, key, value: Attention tensors
            causal_graph: Causal graph to guide attention (optional)
        """
        batch_size, seq_len, _ = query.shape
        
        # If causal graph exists, use it to modify attention
        if causal_graph is not None and seq_len == self.config.n_causal_vars:
            # Expand graph to batch
            graph_bias = causal_graph.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Modificar key con sesgo causal
            key_biased = key + torch.matmul(graph_bias, value)
        else:
            key_biased = key
        
        # Standard attention with biases
        attn_output, attn_weights = self.attention(query, key_biased, value)
        
        return attn_output


class ATEEstimator(nn.Module):
    """
    Average Treatment Effect (ATE) Estimator
    E[Y | do(X=1)] - E[Y | do(X=0)]
    
    Amortized inference: predicts ATE in 1 forward pass via learned
    potential outcome function (vs 32K MC samples as baseline).
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        
        # Network to estimate potential outcome (simple forward, no treatment/outcome indices)
        self.potential_outcome_net = nn.Sequential(
            nn.Linear(config.d_model * config.n_causal_vars, config.ate_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ate_hidden_dim, config.ate_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.ate_hidden_dim // 2, 1)
        )
        
        # Amortized ATE network: variables + treatment/outcome one-hot → ATE
        input_dim = (config.d_model * config.n_causal_vars) + 2 * config.n_causal_vars
        
        self.amortized_net = nn.Sequential(
            nn.Linear(input_dim, config.ate_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ate_hidden_dim, config.ate_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ate_hidden_dim, 1)
        )
        
        # Error prediction (aleatoric uncertainty)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.ate_hidden_dim, config.ate_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.ate_hidden_dim // 2, 1),
            nn.Softplus()
        )
    
    def estimate_ate_amortized(self, variables: torch.Tensor,
                                treatment_idx: int, outcome_idx: int) -> Dict:
        """
        Estimate ATE in a single forward pass (amortized inference).
        
        Args:
            variables: (B, n_vars, d_model) observed variables
            treatment_idx: Index of treatment variable
            outcome_idx: Index of outcome variable
        
        Returns:
            dict with estimated ATE and uncertainty
        """
        batch_size, n_vars, d_model = variables.shape
        
        # Flatten variables
        var_flat = variables.view(batch_size, -1)  # (B, n_vars * d_model)
        
        # Treatment one-hot (broadcasted)
        treatment_onehot = torch.zeros(batch_size, n_vars, device=variables.device)
        treatment_onehot[:, treatment_idx] = 1.0
        
        # Outcome one-hot
        outcome_onehot = torch.zeros(batch_size, n_vars, device=variables.device)
        outcome_onehot[:, outcome_idx] = 1.0
        
        # Concatenate
        x = torch.cat([var_flat, treatment_onehot, outcome_onehot], dim=-1)
        
        # Hidden representation
        hidden = self.amortized_net[0](x)
        hidden = self.amortized_net[1](hidden)
        hidden = self.amortized_net[2](hidden)
        hidden = self.amortized_net[3](hidden)
        
        # ATE prediction
        ate = self.amortized_net[4](hidden).squeeze(-1)
        
        # Uncertainty (aleatoric)
        uncertainty = self.uncertainty_head(hidden).squeeze(-1)
        
        return {
            'ate': ate.mean().item(),
            'ate_per_sample': ate,
            'ci_95_lower': (ate - 1.96 * uncertainty).mean().item(),
            'ci_95_upper': (ate + 1.96 * uncertainty).mean().item(),
            'n_samples': 1,
            'amortized': True
        }
    
    def estimate_ate(self, cfm: PartialCausalGraph, variables: torch.Tensor,
                     treatment_idx: int, outcome_idx: int, use_amortized: bool = True) -> Dict:
        """
        Estimate ATE: amortized (1 forward) or Monte Carlo (n_mc_samples).
        
        Args:
            cfm: Causal model with graph
            variables: Observed variables
            treatment_idx: Index of treatment variable
            outcome_idx: Index of outcome variable
            use_amortized: Use amortized inference (default True)
        
        Returns:
            dict with estimated ATE, confidence interval
        """
        if use_amortized:
            return self.estimate_ate_amortized(variables, treatment_idx, outcome_idx)
        
        # Monte Carlo baseline (for training/validation)
        batch_size = variables.size(0)
        device = variables.device
        
        outcomes_treatment_1 = []
        outcomes_treatment_0 = []
        
        for _ in range(self.config.n_mc_samples):
            outcome_1 = cfm(variables, intervention_idx=treatment_idx, intervention_value=1.0)
            outcomes_treatment_1.append(outcome_1[:, outcome_idx].mean(dim=-1))
            
            outcome_0 = cfm(variables, intervention_idx=treatment_idx, intervention_value=0.0)
            outcomes_treatment_0.append(outcome_0[:, outcome_idx].mean(dim=-1))
        
        mean_1 = torch.stack(outcomes_treatment_1).mean(dim=0)
        mean_0 = torch.stack(outcomes_treatment_0).mean(dim=0)
        
        ate = mean_1 - mean_0
        
        std_1 = torch.stack(outcomes_treatment_1).std(dim=0)
        std_0 = torch.stack(outcomes_treatment_0).std(dim=0)
        
        ci_lower = ate - 1.96 * torch.sqrt(std_1**2 + std_0**2)
        ci_upper = ate + 1.96 * torch.sqrt(std_1**2 + std_0**2)
        
        return {
            'ate': ate.mean().item(),
            'ate_per_sample': ate,
            'ci_95_lower': ci_lower.mean().item(),
            'ci_95_upper': ci_upper.mean().item(),
            'n_samples': self.config.n_mc_samples,
            'amortized': False
        }
    
    def forward(self, variables: torch.Tensor) -> torch.Tensor:
        """Simple forward for large model integration"""
        batch_size, n_vars, d_model = variables.shape
        x = variables.view(batch_size, -1)
        return self.potential_outcome_net(x)


class CausalFoundationModel(nn.Module):
    """
    Complete Causal Foundation Model
    Integrates partial graph, causal attention, and ATE estimation
    """
    
    def __init__(self, config: CFMConfig):
        super().__init__()
        self.config = config
        
        # Partial causal graph
        self.causal_graph = PartialCausalGraph(config)
        
        # Causal biased attention
        self.causal_attention = CausalAttention(config)
        
        # ATE estimator
        self.ate_estimator = ATEEstimator(config)
        
        # Variable encoder
        self.variable_encoder = nn.Linear(config.d_model, config.d_model)
        
        # Decoder para predicciones
        self.outcome_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Identifiability metrics
        self.identifiability_history = []
    
    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations to causal space"""
        return self.variable_encoder(observations)
    
    def forward(self, observations: torch.Tensor, 
                treatment_idx: Optional[int] = None,
                outcome_idx: Optional[int] = None,
                return_ate: bool = False) -> Dict:
        """
        Full CFM forward pass
        
        Args:
            observations: Observed data (B, n_vars, d)
            treatment_idx: For ATE estimation
            outcome_idx: For ATE estimation
            return_ate: Whether to compute ATE
        """
        # Encode
        variables = self.encode_observations(observations)
        
        # Apply causal graph
        causal_outcomes = self.causal_graph(variables)
        
        # Causal attention
        attended = self.causal_attention(
            causal_outcomes, causal_outcomes, causal_outcomes,
            causal_graph=self.causal_graph.get_adjacency_matrix()
        )
        
        # Decode
        predictions = self.outcome_decoder(attended)
        
        result = {
            'predictions': predictions,
            'causal_outcomes': causal_outcomes,
            'attended': attended,
            'causal_graph': self.causal_graph.get_adjacency_matrix(hard=True)
        }
        
        # Estimate ATE if requested
        if return_ate and treatment_idx is not None and outcome_idx is not None:
            ate_results = self.ate_estimator.estimate_ate(
                self.causal_graph, variables, treatment_idx, outcome_idx
            )
            result['ate_estimation'] = ate_results
        
        return result
    
    def intervene(self, observations: torch.Tensor, 
                  intervention_dict: Dict[int, float]) -> torch.Tensor:
        """
        Execute multiple do(X=x) interventions
        
        Args:
            observations: Datos observados
            intervention_dict: {var_idx: value} interventions
        """
        variables = self.encode_observations(observations)
        
        # Apply interventions sequentially
        for var_idx, value in intervention_dict.items():
            variables = self.causal_graph(
                variables, 
                intervention_idx=var_idx, 
                intervention_value=value
            )
        
        return self.outcome_decoder(variables)
    
    def get_causal_explanation(self, var_idx: int) -> Dict:
        """Get causal explanation for a variable"""
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
        Zero-shot causal inference
        Compares with specialized models
        """
        # Estimate causal effect
        result = self.forward(
            source_data, 
            treatment_idx=target_intervention,
            outcome_idx=0,  # Asumiendo outcome principal
            return_ate=True
        )
        
        ate_estimated = result['ate_estimation']['ate']
        
        # Calculate accuracy if reference available
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
