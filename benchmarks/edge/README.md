# Edge Benchmark Suite — Diagonal++ SSM

Measures CPU inference performance of Mamba3MIMO across model sizes, sequence lengths, and batch sizes.

## Usage

```bash
cd /path/to/HOBBIT
PYTHONPATH=. python benchmarks/edge/bench_edge.py
```

## Model Configs

| Profile | d_model | d_state | d_inner | Layers |
|---------|---------|---------|---------|--------|
| nano    | 64      | 8       | 128     | 2      |
| tiny    | 128     | 16      | 256     | 4      |
| small   | 256     | 32      | 512     | 6      |

## Metrics

- **tok/s**: Tokens per second (higher is better)
- **P50/P95/P99 (ms)**: Latency percentiles
- **Memory**: Peak memory usage (MB)

## Output

Results saved as JSON to `results/bench_<timestamp>.json`.
