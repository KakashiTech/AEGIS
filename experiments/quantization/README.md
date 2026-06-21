# SSM Quantization

Prototype INT8 quantization for the Diagonal++ SSM kernel in Mamba3MIMO.

## Motivation

The Diagonal++ SSM is uniquely suited for quantization because:

1. **Element-wise recurrence** — The state update `h_t = Λ · h_{t-1} + (I - Λ) · B̄ x_t` is element-wise, meaning no matrix multiply in the recurrence path. This avoids the main source of quantization error in traditional RNNs/LSTMs.

2. **No softmax** — Attention-based models require FP16/FP32 for softmax numerical stability. The SSM has no softmax, so INT8 inference is viable everywhere.

3. **Bounded κ parameters** — The discretization parameters κ = exp(-Δ · exp(log λ)) are naturally bounded in (0, 1), making them quantization-friendly.

## Quantization Strategy

We use asymmetric uniform quantization (affine quantization):

```
x_q = clamp(round(x / scale + zero_point), qmin, qmax)
x̂ = (x_q - zero_point) * scale
```

Where:
- `scale = (max - min) / (2^n - 1)`
- `zero_point = round(qmin - min / scale)`

### What we quantize

- **SSM weights** (log λ, B, C) — quantized to INT8 and stored
- **Projection matrices** — quantized via the wrapper
- **State h** — left in FP32 (recurrence needs precision)

### What we DON'T quantize

- The recurrence computation itself — kept in FP32 since it's element-wise and not the bottleneck
- The discretization step (exp/softplus) — kept in FP32 for numerical stability

## Files

| File | Description |
|------|-------------|
| `quant_ssm.py` | Quantized SSM implementation and benchmark |
| `__init__.py` | Package init |

## Usage

```python
from experiments.quantization.quant_ssm import quantize_model, QuantizedSSMStep

# Quantize a trained model
wrapper, size, ratio = quantize_model(model, n_bits=8)

# Run quantized inference
output = wrapper.inference_quantized(x)
```

## Results

Run the benchmark to see compression ratios for your model:

```bash
python -m experiments.quantization.quant_ssm
```

Expected compression for a 256-d model with 4 layers:

| Precision | Size | Ratio |
|-----------|------|-------|
| FP32      | ~6 MB | 1.0x |
| INT8      | ~1.5 MB | 4.0x |

## Future Work

1. **Actual INT8 matmul** — Replace FP32 dequantize+matmul with `torch.int8` matmul on H100/Ada GPUs
2. **K-quant** — Per-channel quantization for projection weights
3. **Fusion** — Fuse quantization with the SSM kernel for end-to-end INT8
4. **Weight-only vs activation quantization** — Currently weight-only; activation quantization needs careful calibration
5. **4-bit / NF4** — Extend to 4-bit quantization for further compression
