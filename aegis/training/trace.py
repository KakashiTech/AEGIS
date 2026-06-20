"""
TrainingTrace - Sistema de trazabilidad runtime obligatorio.
Every step must produce real execution evidence.
"""

import json
import time
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class StepTrace:
    """Trace de un solo step de entrenamiento."""
    step: int
    timestamp: float

    # Loss breakdown
    loss_total: float = 0.0
    loss_components: Dict[str, float] = field(default_factory=dict)

    # Gradient signals
    grad_norm_global: float = 0.0
    grad_norm_per_module: Dict[str, float] = field(default_factory=dict)

    # Parameter changes (param_delta)
    param_delta_per_module: Dict[str, float] = field(default_factory=dict)
    param_mean_per_module: Dict[str, float] = field(default_factory=dict)

    # Execution counters
    forward_calls: int = 0
    backward_calls: int = 0
    optimizer_step_executed: bool = False

    # Memory
    memory_mb: float = 0.0
    memory_reserved_mb: float = 0.0

    # Numerical stability
    nan_detected: bool = False
    inf_detected: bool = False

    # Latency
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    total_step_ms: float = 0.0

    # Learning verdict
    has_gradients: bool = False
    has_param_updates: bool = False
    loss_is_finite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_learning(self) -> bool:
        """Verdict: este step produjo aprendizaje real?"""
        return (
            self.loss_is_finite
            and self.has_gradients
            and self.grad_norm_global > 0.0
            and self.has_param_updates
            and not self.nan_detected
            and not self.inf_detected
        )


class RuntimeTraceLogger:
    """
    Logger obligatorio de trazas runtime.
    If a module does not log here, it is considered DEAD CODE.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        filename: str = "runtime_trace.jsonl",
        modules_to_track: Optional[List[str]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.modules_to_track = modules_to_track or []

        self.step_count = 0
        self.learning_steps = 0
        self.dead_steps = 0
        self.traces: List[StepTrace] = []

        # Parameter snapshots for delta computation
        self._param_snapshots: Dict[str, torch.Tensor] = {}

        # Escribir header
        self._append_line({"event": "trace_session_start", "timestamp": time.time()})

    def _append_line(self, obj: Dict[str, Any]) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    def snapshot_params(self, model: nn.Module) -> None:
        """Save parameter state BEFORE step."""
        self._param_snapshots.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._param_snapshots[name] = param.detach().clone()

    def compute_param_delta(self, model: nn.Module) -> Dict[str, float]:
        """Compute parameter delta AFTER step."""
        delta = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._param_snapshots:
                diff = (param.detach() - self._param_snapshots[name]).abs().mean().item()
                delta[name] = diff
        return delta

    def compute_grad_norm(self, model: nn.Module) -> tuple[float, Dict[str, float]]:
        """Compute global and per-module gradient norm."""
        global_norm = 0.0
        per_module: Dict[str, float] = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                g_norm = param.grad.norm().item()
                global_norm += g_norm ** 2
                # Group by module (first name segment)
                module_name = name.split(".")[0]
                per_module[module_name] = per_module.get(module_name, 0.0) + g_norm ** 2

        global_norm = global_norm ** 0.5
        per_module = {k: v ** 0.5 for k, v in per_module.items()}
        return global_norm, per_module

    def log_step(
        self,
        step: int,
        loss_total: float,
        loss_components: Dict[str, float],
        model: nn.Module,
        forward_calls: int = 1,
        backward_calls: int = 1,
        optimizer_step_executed: bool = False,
        forward_ms: float = 0.0,
        backward_ms: float = 0.0,
        optimizer_ms: float = 0.0,
    ) -> StepTrace:
        """
        Loguear un step completo con evidencia real.
        """
        trace = StepTrace(step=step, timestamp=time.time())

        # Loss
        trace.loss_total = float(loss_total)
        trace.loss_components = {k: float(v) for k, v in loss_components.items()}
        trace.loss_is_finite = (
            not (loss_total != loss_total)  # not NaN
            and not (abs(loss_total) == float("inf"))
            and loss_total != 0.0  # zero-loss path bloqueado
        )

        # Numerical stability
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    trace.nan_detected = True
                if torch.isinf(param.grad).any():
                    trace.inf_detected = True
            if torch.isnan(param).any():
                trace.nan_detected = True
            if torch.isinf(param).any():
                trace.inf_detected = True

        # Gradients
        trace.grad_norm_global, trace.grad_norm_per_module = self.compute_grad_norm(model)
        trace.has_gradients = trace.grad_norm_global > 0.0

        # Parameter deltas
        trace.param_delta_per_module = self.compute_param_delta(model)
        trace.param_mean_per_module = {
            name.split(".")[0]: delta
            for name, delta in trace.param_delta_per_module.items()
        }
        # Group by module
        module_deltas: Dict[str, List[float]] = {}
        for name, delta in trace.param_delta_per_module.items():
            mod = name.split(".")[0]
            module_deltas.setdefault(mod, []).append(delta)
        trace.param_delta_per_module = {
            mod: sum(vals) / len(vals) for mod, vals in module_deltas.items()
        }
        trace.has_param_updates = any(d > 0.0 for d in trace.param_delta_per_module.values())

        # Execution counters
        trace.forward_calls = forward_calls
        trace.backward_calls = backward_calls
        trace.optimizer_step_executed = optimizer_step_executed

        # Memory
        if torch.cuda.is_available():
            trace.memory_mb = torch.cuda.memory_allocated() / 1024 ** 2
            trace.memory_reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
        else:
            trace.memory_mb = 0.0
            trace.memory_reserved_mb = 0.0

        # Latency
        trace.forward_ms = forward_ms
        trace.backward_ms = backward_ms
        trace.optimizer_ms = optimizer_ms
        trace.total_step_ms = forward_ms + backward_ms + optimizer_ms

        # Contadores
        self.step_count += 1
        if trace.is_learning:
            self.learning_steps += 1
        else:
            self.dead_steps += 1

        self.traces.append(trace)
        self._append_line(trace.to_dict())

        return trace

    def get_summary(self) -> Dict[str, Any]:
        """Training session summary."""
        if not self.traces:
            return {"status": "NO_TRACES", "learning_rate": 0.0}

        learning_rate = self.learning_steps / max(self.step_count, 1)

        return {
            "status": "PARTIAL_FUNCTIONAL" if learning_rate < 0.95 else "FUNCTIONAL",
            "total_steps": self.step_count,
            "learning_steps": self.learning_steps,
            "dead_steps": self.dead_steps,
            "learning_rate": learning_rate,
            "avg_grad_norm": sum(t.grad_norm_global for t in self.traces) / len(self.traces),
            "avg_param_delta": sum(
                sum(t.param_delta_per_module.values()) / max(len(t.param_delta_per_module), 1)
                for t in self.traces
            ) / len(self.traces),
            "nan_events": sum(1 for t in self.traces if t.nan_detected),
            "inf_events": sum(1 for t in self.traces if t.inf_detected),
            "avg_step_ms": sum(t.total_step_ms for t in self.traces) / len(self.traces),
        }

    def print_summary(self) -> None:
        summary = self.get_summary()
        print("=" * 60)
        print("RUNTIME TRACE SUMMARY")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("=" * 60)
