import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention_matrix(Q, K, causal=True):
    """Compute full causal attention matrix from Q, K."""
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if causal:
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    return F.softmax(scores, dim=-1)

def approximate_with_ssm(A, num_dims=16):
    """Find best SSM approximation of attention matrix A.
    
    A is (L, L) lower-triangular. We want:
    SSM_ij = C_i * (λ^{i-j} * B_j)  (for i ≥ j)
    
    But since Diagonal++ has per-dimension λ_k, we approximate:
    A ≈ sum_k [C_k ⊗ B_k] * (diag(λ_k) in time)
    
    This is essentially: learn λ_k, B_k, C_k to minimize ||A - A_hat||_F
    """
    L = A.size(-1)
    
    # Learnable SSM parameters
    log_lambda = nn.Parameter(torch.zeros(num_dims))  # log(1-λ) for stability
    B = nn.Parameter(torch.randn(L, num_dims) * 0.01)
    C = nn.Parameter(torch.randn(L, num_dims) * 0.01)
    
    optimizer = torch.optim.Adam([log_lambda, B, C], lr=0.01)
    
    for step in range(2000):
        lam = torch.sigmoid(log_lambda)  # λ in (0, 1)
        
        # Build SSM matrix: A_hat[i,j] = sum_k C[i,k] * lam[k]^{i-j} * B[j,k]
        # This is a structured matrix we can compute efficiently
        i_idx = torch.arange(L, device=A.device).view(-1, 1, 1)  # (L, 1, 1)
        j_idx = torch.arange(L, device=A.device).view(1, -1, 1)  # (1, L, 1)
        
        decay = lam.view(1, 1, -1) ** (i_idx - j_idx)  # (L, L, d), zero for i<j
        decay = torch.where(i_idx >= j_idx, decay, torch.zeros_like(decay))
        
        # A_hat[i,j] = sum_k C[i,k] * decay[i,j,k] * B[j,k]
        A_hat = torch.einsum('ik,ijk,jk->ij', C, decay, B)
        
        loss = F.mse_loss(A_hat, A)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"Step {step}: MSE = {loss.item():.6f}")
    
    return A_hat.detach(), torch.sigmoid(log_lambda).detach(), B.detach(), C.detach()


def compute_ssm_approximation_error(A, A_hat):
    """Compute approximation quality metrics."""
    mse = F.mse_loss(A_hat, A).item()
    cos_sim = F.cosine_similarity(A_hat.flatten(), A.flatten(), dim=0).item()
    return {"mse": mse, "cosine_similarity": cos_sim}

def demo():
    """Demonstrate attention→SSM mapping with synthetic Q, K matrices."""
    torch.manual_seed(42)
    L, d_head = 32, 64
    
    Q = torch.randn(1, L, d_head)
    K = torch.randn(1, L, d_head)
    
    A = attention_matrix(Q, K)[0]  # (L, L)
    
    print(f"Attention matrix shape: {A.shape}")
    print(f"Attention matrix sparsity (upper triangle): {(A == 0).sum().item() / A.numel() * 100:.1f}%")
    
    # Try different numbers of SSM dimensions
    for d in [4, 8, 16, 32]:
        A_hat, lam, B, C = approximate_with_ssm(A, num_dims=d)
        metrics = compute_ssm_approximation_error(A, A_hat)
        print(f"d={d}: MSE={metrics['mse']:.6f}, cos_sim={metrics['cosine_similarity']:.4f}")
        print(f"  Learned λ range: [{lam.min().item():.4f}, {lam.max().item():.4f}]")
    
    return A, A_hat


if __name__ == "__main__":
    A, A_hat = demo()
    print("\nFourierFlow: Attention→SSM mapping prototype complete.")
    print(f"Final MSE: {F.mse_loss(A_hat, A).item():.6f}")
