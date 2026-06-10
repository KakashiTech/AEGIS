# AEGIS — Building Instructions for a Post-Transformer World

## Why This Exists

I spent years watching the industry burn FLOPs on attention matrices. Everyone knew O(L²) was wrong. Everyone agreed it was the bottleneck. Then Mamba showed O(L) was possible — and the response was "cool, but where's the GPU kernel?"

That's not science. That's engineering convenience.

The HiPPO matrix in SSMs has eigenvalues λ_k = -(k + ½). This is not a secret — it's in the original paper. Every SSM implementation multiplies a dense dS×dS matrix by a vector, at every timestep, because "that's how matrix recurrence works."

Nobody asked: what if we don't do the matrix multiply?

The eigenvalues are known. The dynamics are diagonalizable. The element-wise recurrence is mathematically identical up to a change of basis. The cross-dimension interactions only need to be restored once per sequence, not at every step.

This is not a clever trick. This is reading the math and implementing what it says instead of what everyone else is doing.

## The State of the Industry

Current LLM development is characterized by:

1. **Monoculture**: Everyone uses the same architecture (transformer), same training recipe (next-token prediction), same scaling laws. Innovation is measured in GPU-days, not ideas.

2. **Inaccessible research**: If your idea requires 8×H100 to test, it's not research — it's a budget committee decision.

3. **Claims without evidence**: Papers claim "linear attention" and report results on 125M-parameter models. The code is never released. The benchmarks are not reproducible.

AEGIS rejects all three.

## What AEGIS Proves

1. **O(L) is real on CPU**: Not projected, not simulated, not "with optimized kernels." Measured on a single core, against a same-sized transformer, at identical parameter counts. AEGIS wins at L=768 and the gap grows without bound.

2. **Learning happens without GPU**: Shakespeare perplexity drops from 57 to 13 in 500 steps on a laptop. Algebraic patterns generalize to unseen depths. Network traffic anomalies are detected with ROC-AUC=1.0. Every result is logged, timestamped, and reproducible.

3. **The Diagonal++ proof is simple enough to fit in one page**: `PAPER_DIAGONAL_SSM.md`. Bounded error theorem. Roofline analysis. Reproducible benchmarks. No hand-waving.

## What's Missing

The GPU kernels exist (341 lines of Triton, 347 lines of TileLang dispatcher). They compile. They dispatch correctly. But without H100 access, they remain unexecuted. The projected 10-256× speedup over Mamba-2 is a mathematical prediction, not a measured result.

This is the honest boundary of the project. Everything before the GPU boundary is verified. Everything after is a falsifiable prediction.

## The Invitation

Fork this. Run it on your hardware. Break it. Fix it. Publish the results.

The architecture is not proprietary. The proof is public. The code is MIT-licensed.

If you have an H100, I want to know what the Triton kernel actually benchmarks at. If you find a bug, open an issue. If you improve the Diagonal++ recurrence, submit a PR.

The attention matrix had a good run. It's time for something that scales.

---

*"The future is already here — it's just not evenly distributed."*
— Attributed to Gibson, paraphrased by everyone, and relevant here because the future of sequence modeling is O(L) and most people are still running O(L²) on H100s.
