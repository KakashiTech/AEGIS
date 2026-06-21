"""
Variational JEPA (VJEPA) — Joint-Embedding Predictive Architecture

Predictive learning system that minimizes variational energy
between predicted and actual representations without token reconstruction.
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
    """Configuration for VJEPA"""
    d_model: int = 768
    d_pred: int = 384
    predictor_depth: int = 12
    ema_decay: float = 0.9998
    mask_ratio: float = 0.75
    mask_strategy: str = "block"  # block, random, causal, causal_graph
    loss_type: str = "l1"  # l1, l2, cosine
    use_variance: bool = True
    context_length: int = 512
    target_length: int = 64
    batch_size: int = 64
    input_dim: int = 768  # input dimension for continuous data (projected to d_model)
    # Causal graph params
    causal_graph: Optional[Dict] = None  # {effect_idx: [cause_indices]}
    n_causal_features: int = 3  # number of causal features for graph masking
    # Thermodynamic regularizer
    thermo_beta: float = 0.01  # β·||h||² coefficient
    track_efficiency: bool = True


class EBM_energy(nn.Module):
    """
    Energy-Based Model for calibrating latent trajectories
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
        Compute energy between prediction and target.
        Lower energy = better prediction.
        """
        z_concat = torch.cat([z_pred, z_target], dim=-1)
        energy = self.energy_net(z_concat)
        return energy.squeeze(-1)
    
    def contrastive_loss(self, 
                         z_pred: torch.Tensor, 
                         z_target: torch.Tensor,
                         negatives: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss for EBM
        """
        B = z_pred.size(0)
        # Positive energy (correct prediction)
        pos_energy = self.forward(z_pred, z_target)  # (B,)
        
        # Negative energies (incorrect predictions)
        # negatives: (B, N, D) — N negative samples per anchor
        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(1)  # (B, D) -> (B, 1, D)
        N = negatives.size(1)
        z_pred_exp = z_pred.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        neg_energy = self.forward(z_pred_exp.reshape(-1, z_pred.size(-1)),
                                  negatives.reshape(-1, negatives.size(-1)))
        neg_energy = neg_energy.reshape(B, N)
        
        # InfoNCE-style loss
        logits = torch.cat([pos_energy.unsqueeze(1), neg_energy], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=z_pred.device)
        
        loss = F.cross_entropy(-logits, labels)
        return loss


class TargetEncoder(nn.Module):
    """
    Target encoder updated via EMA.
    
    Clones ONLY the core backbone (+ liquid layer if present), NOT the full
    composite model. For BGCEngine this avoids cloning lm_head, lorentz_head,
    VJEPA, AbstractCoT — preventing O(d_model²·n_layers) waste.
    
    Provides stable targets for the predictor.
    """
    
    def __init__(self, encoder: nn.Module, ema_decay: float = 0.9998):
        super().__init__()
        self.ema_decay = ema_decay
        
        # Extract core backbone (avoids cloning heads/VJEPA/etc. for composite models)
        backbone = encoder.backbone if hasattr(encoder, 'backbone') else encoder
        self.core = copy.deepcopy(backbone.module if hasattr(backbone, 'module') else backbone)
        
        # Also clone liquid layer if the composite model has one
        if hasattr(encoder, 'liquid_layer'):
            self.liquid = copy.deepcopy(encoder.liquid_layer)
        else:
            self.liquid = nn.Identity()
        
        # Freeze everything
        for param in self.core.parameters():
            param.requires_grad = False
        for param in self.liquid.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        """Update using Exponential Moving Average over matching parameters"""
        source = online_encoder.backbone if hasattr(online_encoder, 'backbone') else online_encoder
        for param_t, param_s in zip(self.core.parameters(), source.parameters()):
            param_t.data.mul_(self.ema_decay).add_(
                param_s.data, alpha=1 - self.ema_decay
            )
        if hasattr(online_encoder, 'liquid_layer') and not isinstance(self.liquid, nn.Identity):
            for param_t, param_s in zip(self.liquid.parameters(), online_encoder.liquid_layer.parameters()):
                param_t.data.mul_(self.ema_decay).add_(
                    param_s.data, alpha=1 - self.ema_decay
                )
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core(x, return_hidden=True)
        if isinstance(out, dict):
            out = out['hidden_states']
        out = self.liquid(out)
        return out


class Predictor(nn.Module):
    """
    Predictor that predicts masked representations
    from context representations
    """
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Position embedding for masked blocks
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_pred))
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.target_length, config.d_pred) * 0.02
        )
        
        # Context projection to predictor dimension
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
        
        # Output projection
        self.output_proj = nn.Linear(config.d_pred, config.d_model)
        
        # Variance prediction (for variational VJEPA) - applied BEFORE output_proj
        if config.use_variance:
            self.variance_pred = nn.Linear(config.d_pred, config.d_model)
    
    def forward(self, 
                context: torch.Tensor, 
                mask_indices: torch.Tensor,
                return_variance: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            context: Representaciones de contexto (B, L_ctx, d_model)
            mask_indices: Masked block indices (B, n_masks)
            return_variance: Whether to return predicted variance
        Returns:
            z_pred: Representation predictions (B, n_masks, d_model)
            variance: Predicted variance (optional)
        """
        batch_size, n_masks = mask_indices.shape
        
        if n_masks > self.config.target_length:
            import warnings
            warnings.warn(f"n_masks={n_masks} > target_length={self.config.target_length}. "
                          f"Truncating to {self.config.target_length}.")
        
        # Project context
        context_proj = self.context_proj(context)  # (B, L_ctx, d_pred)
        
        # Create mask tokens with position
        mask_tokens = self.mask_token.expand(batch_size, n_masks, -1)
        mask_tokens = mask_tokens + self.pos_embed[:, :n_masks, :]
        
        # Concatenate context and masks
        x = torch.cat([context_proj, mask_tokens], dim=1)
        
        # Apply predictor
        x = self.predictor_transformer(x)
        
        # Extract mask predictions
        z_pred = x[:, -n_masks:, :]  # (B, n_masks, d_pred)
        
        variance = None
        if return_variance and self.config.use_variance:
            variance = self.variance_pred(z_pred)
            variance = F.softplus(variance)  # Ensure positivity
        
        z_pred = self.output_proj(z_pred)  # (B, n_masks, d_model)
        
        return z_pred, variance


class MaskingStrategy:
    """
    Masking strategies for JEPA
    """
    
    @staticmethod
    def block_mask(seq_len: int, mask_ratio: float, block_size: int = 4) -> torch.Tensor:
        """
        Block masking (hardware-efficient)
        """
        num_blocks = seq_len // block_size
        num_masked = int(num_blocks * mask_ratio)
        
        # Select random blocks
        mask = torch.zeros(seq_len, dtype=torch.bool)
        block_indices = torch.randperm(num_blocks)[:num_masked]
        
        for idx in block_indices:
            start = idx * block_size
            end = min(start + block_size, seq_len)
            mask[start:end] = True
        
        return mask
    
    @staticmethod
    def random_mask(seq_len: int, mask_ratio: float) -> torch.Tensor:
        """Random masking"""
        num_masked = int(seq_len * mask_ratio)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        indices = torch.randperm(seq_len)[:num_masked]
        mask[indices] = True
        return mask
    
    @staticmethod
    def causal_mask(seq_len: int, mask_ratio: float) -> torch.Tensor:
        """Causal masking (last part of the sequence)"""
        num_masked = int(seq_len * mask_ratio)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[-num_masked:] = True
        return mask
    
    @staticmethod
    def causal_graph_mask(seq_len: int, mask_ratio: float,
                          causal_graph: dict, n_causal_features: int,
                          batch_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Causal graph-based masking.
        
        - EFFECT features (those with causes): masked
        - CAUSE features (those that cause something): visible
        - INDEPENDENT: random masking
        
        Args:
            causal_graph: {effect_idx: [cause_indices]}
            n_causal_features: number of causal features
            batch_data: tensor (B, L, D) optional for feature scaling
        Returns:
            mask: tensor (seq_len,) bool — True = masked
        """
        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        # Determine which feature indices are effects vs causes
        effect_indices = set(causal_graph.get('causes', {}).keys())
        cause_indices = set()
        for causes in causal_graph.get('causes', {}).values():
            cause_indices.update(causes)
        # Features not in either are "independent"
        
        # For causal VJEPA: mask effects + some independent
        # Keep all causes visible
        d_per_feature = seq_len // n_causal_features if n_causal_features > 0 else seq_len
        
        for feat_idx in range(n_causal_features):
            start = feat_idx * d_per_feature
            end = min((feat_idx + 1) * d_per_feature, seq_len)
            feat_len = end - start
            
            if feat_idx in effect_indices:
                # Mask this feature heavily
                n = max(1, int(feat_len * mask_ratio))
                indices = torch.randperm(feat_len)[:n] + start
                mask[indices] = True
            elif feat_idx in cause_indices:
                # Keep visible (unmasked)
                pass
            else:
                # Independent: partial mask
                n = max(1, int(feat_len * mask_ratio * 0.5))
                indices = torch.randperm(feat_len)[:n] + start
                mask[indices] = True
        
        return mask


class VJEPA(nn.Module):
    """
    Complete Variational JEPA
    
    Minimizes variational energy between representations:
    L = ||z_pred - z_target||_1
    """
    
    def __init__(self, encoder: nn.Module, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Online encoder (trainable)
        self.online_encoder = encoder
        
        # Target encoder (EMA)
        self.target_encoder = TargetEncoder(encoder, config.ema_decay)
        
        # Project continuous input (B, L, input_dim) -> (B, L, d_model)
        self.input_proj = nn.Linear(config.input_dim, config.d_model) if config.input_dim != config.d_model else nn.Identity()
        
        # Predictor
        self.predictor = Predictor(config)
        
        # Energy-Based Model
        self.ebm = EBM_energy(config.d_model)
        
        # Masking strategy
        self.masking_strategy = MaskingStrategy()
        
        # EMA loss tracking
        self.register_buffer('ema_loss', torch.tensor(0.0))
        self.step_count = 0
    
    def create_masks(self, batch_size: int, seq_len: int, device: str,
                     batch_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create context and target masks
        """
        context_masks = []
        target_masks = []
        
        for b in range(batch_size):
            if self.config.mask_strategy == "block":
                mask = self.masking_strategy.block_mask(seq_len, self.config.mask_ratio)
            elif self.config.mask_strategy == "random":
                mask = self.masking_strategy.random_mask(seq_len, self.config.mask_ratio)
            elif self.config.mask_strategy == "causal":
                mask = self.masking_strategy.causal_mask(seq_len, self.config.mask_ratio)
            elif self.config.mask_strategy == "causal_graph":
                batch_i = batch_data[b] if batch_data is not None else None
                mask = self.masking_strategy.causal_graph_mask(
                    seq_len, self.config.mask_ratio,
                    self.config.causal_graph or {},
                     self.config.n_causal_features,
                     batch_i
                )
            else:
                mask = self.masking_strategy.block_mask(seq_len, self.config.mask_ratio)
            
            context_mask = ~mask  # Context = unmasked
            target_mask = mask    # Target = masked
            
            context_masks.append(context_mask)
            target_masks.append(target_mask)
        
        return torch.stack(context_masks).to(device), torch.stack(target_masks).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        VJEPA forward pass
        
        Args:
            x: Input (B, L) or (B, L, d)
        Returns:
            dict with loss and representations
        """
        device = x.device
        
        # Project continuous input if needed
        if x.dim() == 3:
            x = self.input_proj(x)
        
        # Get target encoder representations (frozen EMA)
        if x.dim() == 2:
            with torch.no_grad():
                z_full = self.target_encoder(x)  # (B, L, d_model)
        else:
            with torch.no_grad():
                z_full = self.target_encoder(x)
        
        batch_size, seq_len, _ = z_full.shape
        
        # Create masks (pass batch_data for causal_graph)
        batch_data = x if x.dim() == 3 else None
        context_mask, target_mask = self.create_masks(batch_size, seq_len, device, batch_data)
        
        # Context (visible)
        context = []
        for i in range(batch_size):
            ctx = z_full[i][context_mask[i]]
            context.append(ctx)
        
        # Batch padding
        max_ctx_len = max(len(c) for c in context)
        context_padded = torch.zeros(batch_size, max_ctx_len, self.config.d_model, device=device)
        for i, ctx in enumerate(context):
            context_padded[i, :len(ctx)] = ctx
        
        # Masked target indices
        target_indices = []
        for i in range(batch_size):
            indices = torch.where(target_mask[i])[0]
            target_indices.append(indices)
        
        # Ground truth targets (from target encoder)
        z_targets = []
        for i in range(batch_size):
            z_t = z_full[i][target_mask[i]]
            z_targets.append(z_t)
        
        # Target padding
        max_tgt_len = min(max(len(t) for t in z_targets), self.config.target_length)
        if max(len(t) for t in z_targets) > self.config.target_length:
            import warnings
            warnings.warn(f"Max target len ({max(len(t) for t in z_targets)}) > "
                          f"target_length ({self.config.target_length}). Truncating.")
        target_indices_padded = torch.zeros(batch_size, max_tgt_len, dtype=torch.long, device=device)
        z_targets_padded = torch.zeros(batch_size, max_tgt_len, self.config.d_model, device=device)
        
        for i, (indices, z_t) in enumerate(zip(target_indices, z_targets)):
            n = min(len(indices), max_tgt_len)
            target_indices_padded[i, :n] = indices[:n]
            z_targets_padded[i, :n] = z_t[:n]
        
        # Prediction
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
            'target_mask': target_mask,
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute latent prediction loss
        L = ||z_pred - z_target||_1 + β·||h||²
        """
        z_pred = outputs['z_pred']
        z_target = outputs['z_target']
        variance = outputs.get('variance')
        
        # Mask for valid elements
        mask = (z_target.abs().sum(dim=-1) > 0).float()
        
        # Difference
        diff = z_pred - z_target
        
        # Loss by type
        if self.config.loss_type == "l1":
            loss = torch.abs(diff)
        elif self.config.loss_type == "l2":
            loss = diff ** 2
        elif self.config.loss_type == "cosine":
            loss = 1 - F.cosine_similarity(z_pred, z_target, dim=-1, eps=1e-8).unsqueeze(-1)
        else:
            loss = torch.abs(diff)
        
        # Apply mask and average
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * z_pred.size(-1) + 1e-8)
        
        # Variance term (for variational VJEPA)
        if variance is not None:
            var_loss = 0.5 * torch.log(variance + 1e-8) + 0.5 * (diff ** 2) / (variance + 1e-8)
            var_loss = (var_loss * mask.unsqueeze(-1)).sum() / (mask.sum() * z_pred.size(-1) + 1e-8)
            loss = loss + 0.1 * var_loss
        
        # ─── Thermodynamic Regularizer β·||z_pred||² ─────────────────
        # Penalizes prediction energy (latent representations).
        # Lower energy → more stable and generalizable representations.
        if self.config.thermo_beta > 0:
            z_norm = z_pred.pow(2).mean()
            thermo_loss = self.config.thermo_beta * z_norm
            loss = loss + thermo_loss
        
        # Update EMA loss for monitoring
        self.ema_loss = 0.99 * self.ema_loss + 0.01 * loss.detach()
        
        return loss
    
    def update_target_encoder(self):
        """Update target encoder with EMA"""
        self.target_encoder.update(self.online_encoder)
        self.step_count += 1
    
    def train_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """One complete training step"""
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
        
        # Update target encoder
        self.update_target_encoder()
        
        return {
            'loss': loss.item(),
            'ema_loss': self.ema_loss.item(),
        }
