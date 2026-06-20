"""
Learning verification metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def compute_param_delta(model: nn.Module, snapshots: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute mean absolute delta per parameter."""
    deltas = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in snapshots:
            delta = (param.detach() - snapshots[name]).abs().mean().item()
            deltas[name] = delta
    return deltas


def verify_learning_signal(
    model: nn.Module,
    loss: torch.Tensor,
    grad_norm_threshold: float = 1e-10,
    param_delta_threshold: float = 1e-10,
) -> Tuple[bool, Dict[str, any]]:
    """
    Check if a step produced real learning signal.

    Returns:
        (is_learning, diagnostics)
    """
    diagnostics = {
        "loss_finite": False,
        "loss_nonzero": False,
        "has_gradients": False,
        "grad_norm": 0.0,
        "has_param_updates": False,
        "param_delta": 0.0,
        "nan_detected": False,
        "inf_detected": False,
    }

    # Verificar loss
    diagnostics["loss_finite"] = torch.isfinite(loss).item()
    diagnostics["loss_nonzero"] = (loss.abs() > 0.0).item()

    # Verificar gradientes
    total_norm = 0.0
    has_any_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_any_grad = True
            total_norm += param.grad.norm().item() ** 2
            if torch.isnan(param.grad).any():
                diagnostics["nan_detected"] = True
            if torch.isinf(param.grad).any():
                diagnostics["inf_detected"] = True
        if torch.isnan(param).any():
            diagnostics["nan_detected"] = True
        if torch.isinf(param).any():
            diagnostics["inf_detected"] = True

    diagnostics["has_gradients"] = has_any_grad
    diagnostics["grad_norm"] = total_norm ** 0.5 if total_norm > 0 else 0.0

    is_learning = (
        diagnostics["loss_finite"]
        and diagnostics["loss_nonzero"]
        and diagnostics["has_gradients"]
        and diagnostics["grad_norm"] > grad_norm_threshold
        and not diagnostics["nan_detected"]
        and not diagnostics["inf_detected"]
    )

    return is_learning, diagnostics
