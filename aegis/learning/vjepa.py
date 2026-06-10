"""
Variational JEPA (VJEPA) - Joint-Embedding Predictive Architecture

Sistema de aprendizaje predictivo que minimiza energía variacional
entre representaciones predichas y reales sin reconstrucción de tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math
import copy


@dataclass
class VJEPAConfig:
    """Configuración para VJEPA"""
    d_model: int = 768
    d_pred: int = 384
    predictor_depth: int = 12
    ema_decay: float = 0.9998
    mask_ratio: float = 0.75
    mask_strategy: str = "block"  # block, random, causal
    loss_type: str = "l1"  # l1, l2, cosine
    use_variance: bool = True
    context_length: int = 512
    target_length: int = 64
    batch_size: int = 64


class EBM_energy(nn.Module):
    """
    Energy-Based Model para calibrar trayectorias latentes
    """
    
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Calcula energía entre predicción y objetivo
        Menor energía = mejor predicción
        """
        z_concat = torch.cat([z_pred, z_target], dim=-1)
        energy = self.energy_net(z_concat)
        return energy.squeeze(-1)
    
    def contrastive_loss(self, 
                         z_pred: torch.Tensor, 
                         z_target: torch.Tensor,
                         negatives: torch.Tensor) -> torch.Tensor:
        """
        Pérdida contrastiva para EBM
        """
        # Energía positiva (predicción correcta)
        pos_energy = self.forward(z_pred, z_target)
        
        # Energías negativas (predicciones incorrectas)
        neg_energies = []
        for neg in negatives:
            neg_energy = self.forward(z_pred, neg)
            neg_energies.append(neg_energy)
        
        neg_energies = torch.stack(neg_energies, dim=1)
        
        # InfoNCE-style loss
        logits = torch.cat([pos_energy.unsqueeze(1), neg_energies], dim=1)
        labels = torch.zeros(len(z_pred), dtype=torch.long, device=z_pred.device)
        
        loss = F.cross_entropy(-logits, labels)
        return loss


class TargetEncoder(nn.Module):
    """
    Codificador objetivo actualizado mediante EMA
    Proporciona objetivos estables para el predictor
    """
    
    def __init__(self, encoder: nn.Module, ema_decay: float = 0.9998):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.ema_decay = ema_decay
        
        # Congelar parámetros del encoder objetivo
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        """Actualizar usando Promedio Móvil Exponencial"""
        for param_t, param_s in zip(self.encoder.parameters(), 
                                     online_encoder.parameters()):
            param_t.data.mul_(self.ema_decay).add_(
                param_s.data, alpha=1 - self.ema_decay
            )
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, return_hidden=True)


class Predictor(nn.Module):
    """
    Predictor que predice representaciones enmascaradas
    desde representaciones de contexto
    """
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Embed de posición para bloques enmascarados
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_pred))
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.target_length, config.d_pred) * 0.02
        )
        
        # Proyección de contexto a dimensión de predictor
        self.context_proj = nn.Linear(config.d_model, config.d_pred)
        
        # Transformer predictor
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=config.d_pred,
            nhead=8,
            dim_feedforward=config.d_pred * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.predictor_transformer = nn.TransformerEncoder(
            predictor_layer,
            num_layers=config.predictor_depth
        )
        
        # Proyección de salida
        self.output_proj = nn.Linear(config.d_pred, config.d_model)
        
        # Predicción de varianza (para VJEPA variacional) - aplicada ANTES de output_proj
        if config.use_variance:
            self.variance_pred = nn.Linear(config.d_pred, config.d_model)
    
    def forward(self, 
                context: torch.Tensor, 
                mask_indices: torch.Tensor,
                return_variance: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            context: Representaciones de contexto (B, L_ctx, d_model)
            mask_indices: Índices de bloques enmascarados (B, n_masks)
            return_variance: Si retornar varianza predicha
        Returns:
            z_pred: Predicciones de representaciones (B, n_masks, d_model)
            variance: Varianza predicha (opcional)
        """
        batch_size, n_masks = mask_indices.shape
        
        # Proyectar contexto
        context_proj = self.context_proj(context)  # (B, L_ctx, d_pred)
        
        # Crear tokens de máscara con posición
        mask_tokens = self.mask_token.expand(batch_size, n_masks, -1)
        mask_tokens = mask_tokens + self.pos_embed[:, :n_masks, :]
        
        # Concatenar contexto y máscaras
        x = torch.cat([context_proj, mask_tokens], dim=1)
        
        # Aplicar predictor
        x = self.predictor_transformer(x)
        
        # Extraer predicciones de máscaras
        z_pred = x[:, -n_masks:, :]  # (B, n_masks, d_pred)
        
        variance = None
        if return_variance and self.config.use_variance:
            variance = self.variance_pred(z_pred)
            variance = F.softplus(variance)  # Asegurar positividad
        
        z_pred = self.output_proj(z_pred)  # (B, n_masks, d_model)
        
        return z_pred, variance


class MaskingStrategy:
    """
    Estrategias de enmascaramiento para JEPA
    """
    
    @staticmethod
    def block_mask(seq_len: int, mask_ratio: float, block_size: int = 4) -> torch.Tensor:
        """
        Enmascaramiento por bloques (eficiente para hardware)
        """
        num_blocks = seq_len // block_size
        num_masked = int(num_blocks * mask_ratio)
        
        # Seleccionar bloques aleatorios
        mask = torch.zeros(seq_len, dtype=torch.bool)
        block_indices = torch.randperm(num_blocks)[:num_masked]
        
        for idx in block_indices:
            start = idx * block_size
            end = min(start + block_size, seq_len)
            mask[start:end] = True
        
        return mask
    
    @staticmethod
    def random_mask(seq_len: int, mask_ratio: float) -> torch.Tensor:
        """Enmascaramiento aleatorio"""
        num_masked = int(seq_len * mask_ratio)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        indices = torch.randperm(seq_len)[:num_masked]
        mask[indices] = True
        return mask
    
    @staticmethod
    def causal_mask(seq_len: int, mask_ratio: float) -> torch.Tensor:
        """Enmascaramiento causal (última parte de la secuencia)"""
        num_masked = int(seq_len * mask_ratio)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[-num_masked:] = True
        return mask


class VJEPA(nn.Module):
    """
    Variational JEPA completo
    
    Minimiza energía variacional entre representaciones:
    L = ||ẑ_masked - z_target||_1
    """
    
    def __init__(self, encoder: nn.Module, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Codificador online (entrenable)
        self.online_encoder = encoder
        
        # Codificador objetivo (EMA)
        self.target_encoder = TargetEncoder(encoder, config.ema_decay)
        
        # Predictor
        self.predictor = Predictor(config)
        
        # Energy-Based Model
        self.ebm = EBM_energy(config.d_model)
        
        # Estrategia de enmascaramiento
        self.masking_strategy = MaskingStrategy()
        
        # Estadísticas
        self.register_buffer('ema_loss', torch.tensor(0.0))
        self.step_count = 0
    
    def create_masks(self, batch_size: int, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crear máscaras para contexto y objetivo
        """
        context_masks = []
        target_masks = []
        
        for _ in range(batch_size):
            if self.config.mask_strategy == "block":
                mask = self.masking_strategy.block_mask(seq_len, self.config.mask_ratio)
            elif self.config.mask_strategy == "random":
                mask = self.masking_strategy.random_mask(seq_len, self.config.mask_ratio)
            elif self.config.mask_strategy == "causal":
                mask = self.masking_strategy.causal_mask(seq_len, self.config.mask_ratio)
            else:
                mask = self.masking_strategy.block_mask(seq_len, self.config.mask_ratio)
            
            context_mask = ~mask  # Contexto = no enmascarado
            target_mask = mask    # Objetivo = enmascarado
            
            context_masks.append(context_mask)
            target_masks.append(target_mask)
        
        return torch.stack(context_masks).to(device), torch.stack(target_masks).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Paso forward de VJEPA
        
        Args:
            x: Entrada (B, L) o (B, L, d)
        Returns:
            dict con pérdida y representaciones
        """
        device = x.device
        
        # Obtener representaciones
        if x.dim() == 2:
            # Si es input_ids, pasar por encoder
            with torch.no_grad():
                z_full = self.target_encoder(x)  # (B, L, d_model)
        else:
            z_full = x
        
        batch_size, seq_len, _ = z_full.shape
        
        # Crear máscaras
        context_mask, target_mask = self.create_masks(batch_size, seq_len, device)
        
        # Contexto (visible)
        context = []
        for i in range(batch_size):
            ctx = z_full[i][context_mask[i]]
            context.append(ctx)
        
        # Padding para batch
        max_ctx_len = max(len(c) for c in context)
        context_padded = torch.zeros(batch_size, max_ctx_len, self.config.d_model, device=device)
        for i, ctx in enumerate(context):
            context_padded[i, :len(ctx)] = ctx
        
        # Índices de targets enmascarados
        target_indices = []
        for i in range(batch_size):
            indices = torch.where(target_mask[i])[0]
            target_indices.append(indices)
        
        # Targets reales (del encoder objetivo)
        z_targets = []
        for i in range(batch_size):
            z_t = z_full[i][target_mask[i]]
            z_targets.append(z_t)
        
        # Padding de targets
        max_tgt_len = min(max(len(t) for t in z_targets), self.config.target_length)
        target_indices_padded = torch.zeros(batch_size, max_tgt_len, dtype=torch.long, device=device)
        z_targets_padded = torch.zeros(batch_size, max_tgt_len, self.config.d_model, device=device)
        
        for i, (indices, z_t) in enumerate(zip(target_indices, z_targets)):
            n = min(len(indices), max_tgt_len)
            target_indices_padded[i, :n] = indices[:n]
            z_targets_padded[i, :n] = z_t[:n]
        
        # Predicción
        z_pred, variance = self.predictor(
            context_padded, 
            target_indices_padded,
            return_variance=self.config.use_variance
        )
        
        return {
            'z_pred': z_pred,
            'z_target': z_targets_padded,
            'variance': variance,
            'context_mask': context_mask,
            'target_mask': target_mask
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calcular pérdida de predicción latente
        L = ||ẑ_masked - z_target||_1 (o L2, coseno)
        """
        z_pred = outputs['z_pred']
        z_target = outputs['z_target']
        variance = outputs.get('variance')
        
        # Máscara para elementos válidos
        mask = (z_target.abs().sum(dim=-1) > 0).float()
        
        # Diferencia
        diff = z_pred - z_target
        
        # Pérdida según tipo
        if self.config.loss_type == "l1":
            loss = torch.abs(diff)
        elif self.config.loss_type == "l2":
            loss = diff ** 2
        elif self.config.loss_type == "cosine":
            loss = 1 - F.cosine_similarity(z_pred, z_target, dim=-1, eps=1e-8).unsqueeze(-1)
        else:
            loss = torch.abs(diff)
        
        # Aplicar máscara y promediar
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * z_pred.size(-1) + 1e-8)
        
        # Término de varianza (para VJEPA variacional)
        if variance is not None:
            # Negative log-likelihood con varianza
            var_loss = 0.5 * torch.log(variance + 1e-8) + 0.5 * (diff ** 2) / (variance + 1e-8)
            var_loss = (var_loss * mask.unsqueeze(-1)).sum() / (mask.sum() * z_pred.size(-1) + 1e-8)
            loss = loss + 0.1 * var_loss
        
        # Actualizar EMA loss para monitoreo
        self.ema_loss = 0.99 * self.ema_loss + 0.01 * loss.detach()
        
        return loss
    
    def update_target_encoder(self):
        """Actualizar codificador objetivo con EMA"""
        self.target_encoder.update(self.online_encoder)
        self.step_count += 1
    
    def train_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Un paso completo de entrenamiento"""
        self.online_encoder.train()
        self.predictor.train()
        
        # Forward
        outputs = self.forward(x)
        loss = self.compute_loss(outputs)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.online_encoder.parameters()) + list(self.predictor.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        # Actualizar encoder objetivo
        self.update_target_encoder()
        
        return {
            'loss': loss.item(),
            'ema_loss': self.ema_loss.item()
        }
