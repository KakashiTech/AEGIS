"""
Motor de Continuo Bio-Geométrico (BGCE)
Pipeline E2E: input_ids → Embedding → Mamba3 → (VJEPA) → (Lorentz) → (Abstract-CoT) → LM Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import time
import math
from tqdm import tqdm

from ..core.mamba3_mimo import Mamba3MIMO, SSMConfig
from ..geometry.lorentz_layers import LorentzProjection, LorentzLinear, LorentzManifold
from ..learning.vjepa import VJEPA, VJEPAConfig
from ..cognition.abstract_cot import AbstractCoT, AbstractCoTConfig, VSAModule


@dataclass
class BGCEConfig:
    """Configuración completa del Motor BGCE"""
    
    # Arquitectura
    d_model: int = 256
    n_layers: int = 4
    vocab_size: int = 5000
    max_seq_len: int = 2048
    
    # Mamba-3
    ssm_config: SSMConfig = field(default_factory=lambda: SSMConfig(
        d_model=256,
        d_state=16,
        d_inner=512,
        dt_rank=8,
        use_complex=True,
        use_mimo=True,
        use_diagonal_ssm=False
    ))
    
    # Geometría Lorentz
    use_lorentz: bool = False
    lorentz_curvature: float = 1.0
    
    # VJEPA
    use_vjepa: bool = False
    vjepa_config: VJEPAConfig = field(default_factory=lambda: VJEPAConfig(
        d_model=256,
        d_pred=128,
        predictor_depth=2,
        ema_decay=0.9998,
        mask_ratio=0.75
    ))
    
    # Abstract-CoT
    use_abstract_cot: bool = False
    abstract_cot_config: AbstractCoTConfig = field(default_factory=lambda: AbstractCoTConfig(
        num_abstract_tokens=64,
        d_model=256,
        max_reasoning_steps=8,
        use_vsa=True
    ))
    
    # Entrenamiento
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Optimización
    mixed_precision: bool = False
    compile_model: bool = False
    
    # Sistema
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    num_workers: int = 2
    
    # Inferencia
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95


class LorentzHead(nn.Module):
    """
    Cabeza del modelo con proyector de Variedad de Lorentz
    """
    
    def __init__(self, config: BGCEConfig):
        super().__init__()
        self.config = config
        self.manifold = LorentzManifold(config.lorentz_curvature, config.d_model)
        
        # Proyección a Lorentz
        self.to_lorentz = LorentzProjection(
            euclidean_dim=config.d_model,
            lorentz_dim=config.d_model,
            curvature=config.lorentz_curvature
        )
        
        # Capa lineal en Lorentz
        self.lorentz_linear = LorentzLinear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            curvature=config.lorentz_curvature
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) representaciones euclidianas
        Returns:
            logits: (B, L, vocab_size)
        """
        # Proyectar a Lorentz
        x_lorentz = self.to_lorentz(x)  # (B, L, d_model+1)
        
        # Aplicar capa lineal de Lorentz
        logits = self.lorentz_linear(x_lorentz)
        
        return logits


class ContinualLiquidNeurons(nn.Module):
    """
    Solución analítica de tiempo continuo (CfC - Continuous-time Cellular Automata)
    Neuronas líquidas para procesamiento de flujos
    """
    
    def __init__(self, dim: int, time_constant: float = 1.0):
        super().__init__()
        self.dim = dim
        self.tau = time_constant
        
        # Parámetros de ODE
        self.W = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        
        # Función de activación continua
        self.activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Solución de tiempo continuo
        dx/dt = -x/τ + f(W*x + U*input)
        
        Solución analítica con dt pequeño
        """
        # Término de decaimiento
        decay = -x / self.tau
        
        # Término de entrada
        input_term = self.activation(self.W(x))
        
        # Actualización
        dx = decay + input_term
        x_new = x + dt * dx
        
        return x_new


class BGCEngine(nn.Module):
    """
    Motor de Continuo Bio-Geométrico E2E.
    Pipeline: input_ids → Embedding → Mamba3 → (VJEPA) → (Lorentz) → (Abstract-CoT) → LM Head
    """
    
    def __init__(self, config: BGCEConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Núcleo Mamba-3 MIMO
        self.backbone = Mamba3MIMO(config.ssm_config)
        
        # Proyección a Lorentz
        if config.use_lorentz:
            self.lorentz_proj = LorentzProjection(
                euclidean_dim=config.d_model,
                lorentz_dim=config.d_model,
                curvature=config.lorentz_curvature
            )
            self.lorentz_head = LorentzHead(config)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Neuronas líquidas (procesamiento continuo)
        self.liquid_layer = ContinualLiquidNeurons(config.d_model)
        
        # Módulo VJEPA para pretraining (solo si se usa)
        if config.use_vjepa:
            self.vjepa = VJEPA(self, config.vjepa_config)
        
        # Módulo Abstract-CoT (solo si se usa)
        if config.use_abstract_cot:
            self.abstract_cot = AbstractCoT(config.abstract_cot_config)
        
        # Inicialización
        self._init_weights()
        
        # Estadísticas
        self.register_buffer('total_tokens_processed', torch.tensor(0.0))
        self.register_buffer('total_steps', torch.tensor(0))
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                input_ids: torch.Tensor,
                hidden_states: Optional[torch.Tensor] = None,
                use_reasoning: bool = False,
                return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass del BGCE E2E
        
        Args:
            input_ids: (B, L) token IDs
            hidden_states: (B, L, d_model) si ya está embeddeado (opcional)
            use_reasoning: Usar Abstract-CoT
            return_hidden: Retornar estados ocultos
        Returns:
            dict con logits, hidden_states, etc.
        """
        # Embedding si no vienen hidden_states pre-computados
        if hidden_states is None:
            hidden = self.token_embedding(input_ids)
        else:
            hidden = hidden_states
        
        B, L, D = hidden.shape
        
        # Núcleo Mamba-3
        backbone_out = self.backbone(hidden, return_hidden=True)
        hidden = backbone_out if isinstance(backbone_out, torch.Tensor) else backbone_out['hidden']
        
        # Neuronas líquidas
        hidden = self.liquid_layer(hidden)
        
        # Razonamiento abstracto (opcional)
        reasoning_outputs = {}
        if use_reasoning and self.config.use_abstract_cot and hasattr(self, 'abstract_cot'):
            cot_out = self.abstract_cot(input_ids, hidden, use_reasoning=True)
            hidden = cot_out['output']
            reasoning_outputs['abstract_tokens'] = cot_out.get('abstract_tokens')
            reasoning_outputs['reasoning_states'] = cot_out.get('reasoning_states')
        
        # Proyección final
        if self.config.use_lorentz:
            logits = self.lorentz_head(hidden)
        else:
            logits = self.lm_head(hidden)
        
        outputs = {
            'logits': logits,
            'hidden_states': hidden if return_hidden else None,
            **reasoning_outputs,
        }
        
        return outputs
    
    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Obtener estados ocultos (API compatible con Mamba3MIMO)"""
        outputs = self.forward(input_ids, return_hidden=True)
        return outputs['hidden_states']
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 use_reasoning: bool = False) -> torch.Tensor:
        """
        Generación autoregresiva con top-p sampling
        
        Args:
            input_ids: (B, L) prompt
            max_new_tokens: Máximo de tokens a generar
            temperature: Temperatura de sampling
            top_p: Top-p (nucleus) sampling
            use_reasoning: Usar razonamiento abstracto
        Returns:
            generated: (B, L + max_new_tokens)
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if temperature is None:
            temperature = self.config.temperature
        if top_p is None:
            top_p = self.config.top_p
        
        self.eval()
        generated = input_ids
        
        for _ in range(max_new_tokens):
            outputs = self.forward(
                generated,
                use_reasoning=use_reasoning,
                return_hidden=False
            )
            
            logits = outputs['logits']
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-p filtering
            probs = F.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == 2).all():
                break
        
        return generated


class TrainingPipeline:
    """
    Pipeline de entrenamiento en 3 etapas:
    1. SFT sobre datos de razonamiento
    2. Destilación latente (VJEPA)
    3. RL con decodificación restringida
    """
    
    def __init__(self, model: BGCEngine, config: BGCEConfig):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Optimizador
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Scaler para mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Crear optimizador AdamW"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
    
    def _create_scheduler(self):
        """Crear scheduler con warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return max(0.0, 1.0 - (step - self.config.warmup_steps) / 
                     (self.config.max_steps - self.config.warmup_steps))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def stage1_sft(self, dataloader: DataLoader, num_steps: int = 10000):
        """
        Etapa 1: Supervised Fine-Tuning sobre datos de razonamiento
        """
        self.model.train()
        losses = []
        
        pbar = tqdm(total=num_steps, desc="Stage 1: SFT")
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, use_reasoning=True)
                    logits = outputs['logits']
                    
                    # Pérdida de lenguaje
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
            else:
                outputs = self.model(input_ids, use_reasoning=True)
                logits = outputs['logits']
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            # Backward
            loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            losses.append(loss.item())
            
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            pbar.update(1)
        
        pbar.close()
        return losses
    
    def stage2_latent_distillation(self, dataloader: DataLoader, num_steps: int = 20000):
        """
        Etapa 2: Entrenamiento VJEPA - Destilación latente
        """
        self.model.train()
        losses = []
        
        pbar = tqdm(total=num_steps, desc="Stage 2: Latent Distillation (VJEPA)")
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            
            # VJEPA train step
            metrics = self.model.vjepa.train_step(input_ids, self.optimizer)
            
            losses.append(metrics['loss'])
            
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ema': f'{metrics["ema_loss"]:.4f}'
                })
            
            pbar.update(1)
            self.scheduler.step()
        
        pbar.close()
        return losses
    
    def stage3_rl_tuning(self, dataloader: DataLoader, num_steps: int = 5000):
        """
        Etapa 3: RL con decodificación restringida
        """
        # Simplificación: fine-tuning con reward shaping
        self.model.train()
        losses = []
        
        pbar = tqdm(total=num_steps, desc="Stage 3: RL Tuning")
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            
            # Generar con reasoning
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids[:, :10],  # Contexto corto
                    max_new_tokens=50,
                    use_reasoning=True
                )
            
            # Reward: eficiencia de razonamiento
            outputs = self.model(generated, use_reasoning=True)
            
            # Simular reward (en práctica, usar RL verdadero)
            logits = outputs['logits']
            
            # Pérdida con reward weighting
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                generated[:, 1:].reshape(-1)
            )
            
            loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            losses.append(loss.item())
            
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            pbar.update(1)
        
        pbar.close()
        return losses


class InferenceEngine:
    """
    Motor de inferencia optimizado para baja latencia
    """
    
    def __init__(self, model: BGCEngine, config: BGCEConfig):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Estadísticas de latencia
        self.latency_history = []
    
    @torch.no_grad()
    def inference(self, prompt: str, tokenizer, max_tokens: int = 100) -> Tuple[str, float]:
        """
        Inferencia con medición de latencia
        
        Returns:
            response: Respuesta generada
            latency_ms: Latencia promedio en ms
        """
        self.model.eval()
        
        # Tokenizar
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Medir tiempo
        start_time = time.time()
        
        # Generar
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            use_reasoning=True
        )
        
        end_time = time.time()
        
        # Calcular latencia
        total_time_ms = (end_time - start_time) * 1000
        latency_per_token = total_time_ms / max_tokens
        
        self.latency_history.append(latency_per_token)
        
        # Decodificar
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return response, latency_per_token
    
    def benchmark(self, prompts: List[str], tokenizer) -> Dict[str, float]:
        """
        Benchmark de rendimiento
        """
        latencies = []
        
        for prompt in tqdm(prompts, desc="Benchmarking"):
            _, latency = self.inference(prompt, tokenizer, max_tokens=50)
            latencies.append(latency)
        
        return {
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p50_latency_ms': sorted(latencies)[len(latencies) // 2],
            'p99_latency_ms': sorted(latencies)[int(len(latencies) * 0.99)]
        }
