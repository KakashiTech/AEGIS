import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import time

def attention_matrix(Q, K, causal=True):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if causal:
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    return F.softmax(scores, dim=-1)

def approximate_with_ssm(A, num_dims=16, steps=2000, lr=0.005):
    """Find best Diagonal++ SSM approximation of causal attention matrix A.
    
    A is (L, L) lower-triangular. We learn a diagonal SSM:
      A_hat[i,j] = sum_k C[i,k] * λ_k^{i-j} * B[j,k]  (for i ≥ j)
    
    Returns A_hat, λ, B, C and final metrics.
    """
    L = A.size(-1)
    device = A.device
    
    log_lambda = nn.Parameter(torch.zeros(num_dims, device=device))
    B = nn.Parameter(torch.randn(L, num_dims, device=device) * 0.01)
    C = nn.Parameter(torch.randn(L, num_dims, device=device) * 0.01)
    
    optimizer = torch.optim.Adam([log_lambda, B, C], lr=lr)
    
    i_idx = torch.arange(L, device=device).view(-1, 1)
    j_idx = torch.arange(L, device=device).view(1, -1)
    diff = i_idx - j_idx
    diff_valid = torch.where(diff >= 0, diff.float(), 0.0)
    
    for step in range(steps):
        lam = torch.sigmoid(log_lambda)
        log_lam = torch.log(lam + 1e-30)
        decay = torch.exp(diff_valid.unsqueeze(-1) * log_lam.view(1, 1, -1))
        decay = torch.where(diff.view(L, L, 1) >= 0, decay, torch.zeros_like(decay))
        A_hat = torch.einsum('ik,ijk,jk->ij', C, decay, B)
        loss = F.mse_loss(A_hat, A)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"  step {step}: MSE = {loss.item():.8f}")
    
    A_hat = A_hat.detach()
    lam = torch.sigmoid(log_lambda).detach()
    B_det = B.detach()
    C_det = C.detach()
    mse = F.mse_loss(A_hat, A).item()
    cos = F.cosine_similarity(A_hat.flatten(), A.flatten(), dim=0).item()
    return A_hat, lam, B_det, C_det, {"mse": mse, "cosine_similarity": cos}

def run_single_head(L, d_head, d_states, steps, device="cpu"):
    """Run FourierFlow on a single attention head."""
    torch.manual_seed(42)
    Q = torch.randn(1, L, d_head, device=device)
    K = torch.randn(1, L, d_head, device=device)
    A = attention_matrix(Q, K)[0]
    
    print(f"Attention matrix: {A.shape}, sparsity={(A==0).sum().item()/A.numel()*100:.1f}%")
    
    results = []
    for d in d_states:
        t0 = time.time()
        A_hat, lam, B, C, metrics = approximate_with_ssm(A, num_dims=d, steps=steps)
        dt = time.time() - t0
        print(f"L={L} d={d}: MSE={metrics['mse']:.8f} cos={metrics['cosine_similarity']:.6f} "
              f"λ=[{lam.min():.4f},{lam.max():.4f}] time={dt:.0f}s")
        results.append({"d": d, "mse": metrics["mse"], "cos": metrics["cosine_similarity"]})
    return results

def run_multi_head(L, h, d_head, d_state, steps, device="cpu"):
    """Run FourierFlow on each head of a multi-head attention."""
    torch.manual_seed(42)
    Q = torch.randn(1, h, L, d_head, device=device)
    K = torch.randn(1, h, L, d_head, device=device)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
    mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    A_mh = F.softmax(scores, dim=-1)
    
    print(f"Multi-head attention: {A_mh.shape} ({h} heads, L={L})")
    
    results = []
    t0 = time.time()
    for head_idx in range(h):
        A_h = A_mh[0, head_idx]
        _, _, _, _, metrics = approximate_with_ssm(A_h, num_dims=d_state, steps=steps)
        results.append(metrics)
        print(f"  head {head_idx}: MSE={metrics['mse']:.8f} cos={metrics['cosine_similarity']:.6f}")
    
    msess = [r["mse"] for r in results]
    coss = [r["cosine_similarity"] for r in results]
    print(f"Multi-head L={L} h={h} d={d_state}:")
    print(f"  MSE: min={min(msess):.8f} mean={sum(msess)/h:.8f} max={max(msess):.8f}")
    print(f"  Cos: min={min(coss):.6f} mean={sum(coss)/h:.6f} max={max(coss):.6f}")
    print(f"  Total time: {time.time()-t0:.0f}s")
    return results

def main():
    parser = argparse.ArgumentParser(description="FourierFlow: Attention-to-SSM Compiler")
    parser.add_argument("--L", type=int, default=32, help="Sequence length")
    parser.add_argument("--d_head", type=int, default=64, help="Head dimension")
    parser.add_argument("--d_state", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="SSM state dimensions to test")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps per run")
    parser.add_argument("--heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()
    
    if args.heads > 1 and len(args.d_state) > 1:
        print("Multi-head mode: using only first d_state value")
        args.d_state = [args.d_state[0]]
    
    if args.heads > 1:
        run_multi_head(args.L, args.heads, args.d_head,
                       args.d_state[0], args.steps, args.device)
    else:
        run_single_head(args.L, args.d_head, args.d_state,
                        args.steps, args.device)

if __name__ == "__main__":
    main()
