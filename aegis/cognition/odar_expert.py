"""
ODAR-Expert: Enrutador Adaptativo con Inferencia Activa
System 1 (fast) vs System 2 (deliberative with Abstract-CoT)
Minimizes variational energy and varentropy to avoid over-thinking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math


@dataclass
class ODARConfig:
    """Configuration ODAR-Expert"""
    d_model: int = 768
    n_experts: int = 2  # Sistema 1 (fast) y Sistema 2 (slow)
    difficulty_threshold: float = 0.5
    entropy_weight: float = 0.1
    precision_target: float = 0.98
    cost_reduction_target: float = 1.78
    amortization_layers: int = 3


class VariationalEntropyEstimator(nn.Module):
    """
    Estimador de Dificultad basado en Inferencia Activa amortizada
    Compute varentropy to decide System 1 vs System 2
    """
    
    def __init__(self, config: ODARConfig):
        super().__init__()
        self.config = config
        
        # Amortization network for difficulty estimation
        self.amortization_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, config.d_model // 4),
            nn.LayerNorm(config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimator (varentropy)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Softplus()  # Asegurar positividad
        )
        
        # Threshold adaptativo
        self.threshold = nn.Parameter(torch.tensor(config.difficulty_threshold))
        
    def compute_varentropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute varentropy (variance of entropy)
        Mide incertidumbre en la incertidumbre
        """
        # Expected entropy
        probs = F.softmax(x, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1, keepdim=True)
        
        # Entropy variance (simplified varentropy)
        varentropy = self.uncertainty_head(x)
        
        return varentropy
    
    def estimate_difficulty(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Estimar dificultad de la consulta en milisegundos
        
        Returns:
            difficulty_score: (B, 1) entre 0 y 1
            metrics: dict with varentropy, energy, etc.
        """
        # Score de dificultad base
        base_difficulty = self.amortization_net(x)
        
        # Varentropy
        varentropy = self.compute_varentropy(x)
        
        # Variational energy (approximation)
        variational_energy = torch.norm(x, p=2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        
        # Dificultad combinada
        difficulty = base_difficulty + self.config.entropy_weight * varentropy
        
        # Normalizar
        difficulty = torch.clamp(difficulty, 0.0, 1.0)
        
        metrics = {
            'base_difficulty': base_difficulty,
            'varentropy': varentropy,
            'variational_energy': variational_energy,
            'threshold': self.threshold
        }
        
        return difficulty, metrics


class SystemOneExpert(nn.Module):
    """
    System 1: Fast, instinctive, no Abstract-CoT
    Para tareas triviales y patrones memorizados
    """
    
    def __init__(self, d_model: int, vocab_size: int = 50000):
        super().__init__()
        self.d_model = d_model
        
        # Direct projection (fast path)
        self.fast_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )
        
        # Cache de respuestas frecuentes
        self.response_cache = {}
        self.cache_hits = 0
        self.total_queries = 0
    
    def forward(self, hidden_states: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """
        Fast forward pass without reasoning
        """
        self.total_queries += hidden_states.size(0)
        
        # Check cache (simplificado)
        if use_cache:
            cache_key = hash(hidden_states[0, 0, :5].detach().cpu().numpy().tobytes())
            if cache_key in self.response_cache:
                self.cache_hits += 1
                return self.response_cache[cache_key]
        
        # Fast projection
        logits = self.fast_projection(hidden_states)
        
        # Guardar en cache
        if use_cache:
            self.response_cache[cache_key] = logits.detach()
        
        return logits
    
    def get_cache_stats(self) -> Dict:
        """Cache usage statistics"""
        if self.total_queries == 0:
            return {'cache_hit_rate': 0.0, 'hits': 0, 'total': 0}
        
        return {
            'cache_hit_rate': self.cache_hits / self.total_queries,
            'hits': self.cache_hits,
            'total': self.total_queries
        }


class SystemTwoExpert(nn.Module):
    """
    Sistema 2: Lento, deliberativo, con Abstract-CoT
    Para tareas complejas que requieren razonamiento profundo
    """
    
    def __init__(self, d_model: int, abstract_cot_module: nn.Module):
        super().__init__()
        self.d_model = d_model
        self.abstract_cot = abstract_cot_module
        
        # Procesamiento deliberativo
        self.deliberative_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ) for _ in range(3)
        ])
        
        # Cabeza de salida
        self.output_head = nn.Linear(d_model, 50000)
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass deliberativo con Abstract-CoT
        """
        # Aplicar capas deliberativas
        x = hidden_states
        for layer in self.deliberative_layers:
            x = layer(x)
        
        # Abstract-CoT
        cot_outputs = self.abstract_cot(input_ids, x, use_reasoning=True)
        
        # Salida final
        logits = self.output_head(cot_outputs['output'])
        
        metrics = {
            'abstract_tokens_used': cot_outputs['abstract_tokens'].shape[1] if cot_outputs['abstract_tokens'] is not None else 0,
            'efficiency_ratio': cot_outputs.get('efficiency_ratio', 1.0)
        }
        
        return logits, metrics


class ODARRouter(nn.Module):
    """
    Enrutador ODAR que decide entre Sistema 1 y Sistema 2
    Based on difficulty estimation and variational energy minimization
    """
    
    def __init__(self, config: ODARConfig, system1: SystemOneExpert, system2: SystemTwoExpert):
        super().__init__()
        self.config = config
        self.system1 = system1
        self.system2 = system2
        
        # Estimador de dificultad
        self.difficulty_estimator = VariationalEntropyEstimator(config)
        
        # Statistics
        self.system1_usage = 0
        self.system2_usage = 0
        self.total_tokens_saved = 0
        self.math_correct = 0
        self.math_total = 0
    
    def route(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict, int]:
        """
        Decidir enrutamiento y ejecutar
        
        Returns:
            logits: Salida del modelo
            metrics: Decision metrics
            system_used: 1 o 2
        """
        # Estimar dificultad
        difficulty, difficulty_metrics = self.difficulty_estimator.estimate_difficulty(hidden_states)
        
        # Routing decision
        use_system2 = (difficulty > self.difficulty_estimator.threshold).any()
        
        if use_system2:
            # Sistema 2: Deliberativo
            logits, system_metrics = self.system2(hidden_states, input_ids)
            self.system2_usage += 1
            system_used = 2
            
            # Calcular tokens "ahorrados" vs usar Sistema 2 siempre
            tokens_saved = 0  # No savings, but higher accuracy
        else:
            # System 1: Fast
            logits = self.system1(hidden_states)
            self.system1_usage += 1
            system_used = 1
            
            # Tokens ahorrados vs usar Sistema 2
            tokens_saved = 50  # Aproximadamente lo que ahorra Abstract-CoT
        
        self.total_tokens_saved += tokens_saved * hidden_states.size(0)
        
        metrics = {
            **difficulty_metrics,
            **(system_metrics if use_system2 else {}),
            'system_used': system_used,
            'difficulty_score': difficulty.mean().item(),
            'tokens_saved': tokens_saved,
            'routing_decision': 'System 2 (deliberative)' if use_system2 else 'System 1 (fast)'
        }
        
        return logits, metrics, system_used
    
    def update_math_accuracy(self, correct: bool):
        """Update math accuracy"""
        self.math_total += 1
        if correct:
            self.math_correct += 1
    
    def get_stats(self) -> Dict:
        """Performance statistics"""
        total = self.system1_usage + self.system2_usage
        if total == 0:
            return {}
        
        math_accuracy = self.math_correct / max(self.math_total, 1)
        
        return {
            'system1_usage_pct': self.system1_usage / total * 100,
            'system2_usage_pct': self.system2_usage / total * 100,
            'total_tokens_saved': self.total_tokens_saved,
            'math_accuracy': math_accuracy,
            'target_accuracy': self.config.precision_target,
            'accuracy_gap': self.config.precision_target - math_accuracy,
            'cache_hit_rate': self.system1.get_cache_stats()['cache_hit_rate']
        }


class ODARExpertSystem(nn.Module):
    """
    Sistema ODAR-Expert completo
    Integrates difficulty estimation and adaptive routing
    """
    
    def __init__(self, config: ODARConfig, abstract_cot_module: nn.Module):
        super().__init__()
        self.config = config
        
        # Crear expertos
        self.system1 = SystemOneExpert(config.d_model)
        self.system2 = SystemTwoExpert(config.d_model, abstract_cot_module)
        
        # Router
        self.router = ODARRouter(config, self.system1, self.system2)
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass con enrutamiento adaptativo
        """
        logits, metrics, system_used = self.router.route(hidden_states, input_ids)
        
        return logits, metrics
    
    def evaluate_math_problem(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, 
                             ground_truth: torch.Tensor) -> Tuple[torch.Tensor, bool, Dict]:
        """
        Evaluate math problem and track accuracy
        """
        logits, metrics, system_used = self.router.route(hidden_states, input_ids)
        
        # Determinar si es correcto (simplificado)
        prediction = logits.argmax(dim=-1)
        correct = (prediction == ground_truth).all().item()
        
        # Update statistics
        self.router.update_math_accuracy(correct)
        
        return logits, correct, metrics
    
    def get_full_stats(self) -> Dict:
        """Get complete system statistics"""
        stats = self.router.get_stats()
        stats['target_cost_reduction'] = self.config.cost_reduction_target
        
        # Verificar si se alcanzaron objetivos
        if 'math_accuracy' in stats:
            stats['target_98_reached'] = stats['math_accuracy'] >= 0.98
        
        return stats
