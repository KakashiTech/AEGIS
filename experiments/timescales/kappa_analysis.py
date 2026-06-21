import torch
import math

def extract_kappa(model):
    """Extract κ parameters from a Mamba3MIMO model.
    
    Returns: dict layer_idx -> {'kappa': tensor(d_state,), 'lambda': tensor(d_state,), 'half_life': tensor(d_state,)}
    """
    kappa_data = {}
    for name, param in model.named_parameters():
        if 'kappa_scale' in name and 'kappa_base' not in name:
            parts = name.split('.')
            layer_idx = None
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
            if layer_idx is None:
                continue
            
            kappa = param.detach().abs()
            d_state = kappa.shape[0]
            hippo_lambda = -(torch.arange(d_state, dtype=torch.float32) + 0.5)
            half_life = -math.log(2) / (kappa * hippo_lambda + 1e-8)
            
            kappa_data[layer_idx] = {
                'kappa': kappa,
                'lambda': hippo_lambda,
                'half_life': half_life,
            }
    
    return kappa_data


def plot_kappa_heatmap(kappa_data, save_path='kappa_heatmap.png'):
    """Plot κ values as a heatmap: layers × dimensions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return None
    
    n_layers = len(kappa_data)
    if n_layers == 0:
        print("No κ data to plot")
        return None
    
    d_state = kappa_data[list(kappa_data.keys())[0]]['kappa'].shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, n_layers * 0.6)))
    
    sorted_layers = sorted(kappa_data.keys())
    
    kappa_matrix = torch.stack([kappa_data[i]['kappa'] for i in sorted_layers])
    im1 = axes[0].imshow(kappa_matrix.numpy(), aspect='auto', cmap='viridis')
    axes[0].set_xlabel('SSM Dimension')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('\u03ba (timescale curvature)')
    plt.colorbar(im1, ax=axes[0])
    
    hl_matrix = torch.stack([kappa_data[i]['half_life'] for i in sorted_layers])
    hl_log = torch.log10(hl_matrix + 1e-8)
    im2 = axes[1].imshow(hl_log.numpy(), aspect='auto', cmap='plasma')
    axes[1].set_xlabel('SSM Dimension')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('log\u2081\u2080(Half-life in steps)')
    plt.colorbar(im2, ax=axes[1])
    
    valid = (hl_matrix >= 1.0) & (hl_matrix <= 1000.0)
    im3 = axes[2].imshow(valid.numpy(), aspect='auto', cmap='RdYlGn')
    axes[2].set_xlabel('SSM Dimension')
    axes[2].set_ylabel('Layer')
    axes[2].set_title('Valid timescale (1-1000 steps)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved \u03ba heatmap to {save_path}")
    plt.close()
    return fig


def analyze_timescale_distribution(kappa_data):
    """Statistics about the timescale distribution across layers."""
    stats = {}
    for layer_idx, data in sorted(kappa_data.items()):
        hl = data['half_life']
        stats[layer_idx] = {
            'kappa_mean': data['kappa'].mean().item(),
            'kappa_std': data['kappa'].std().item(),
            'kappa_min': data['kappa'].min().item(),
            'kappa_max': data['kappa'].max().item(),
            'hl_median': hl.median().item(),
            'hl_geometric_mean': hl.log().mean().exp().item(),
            'hl_range': (hl.min().item(), hl.max().item()),
            'n_valid_timescales': ((hl >= 1.0) & (hl <= 1000.0)).sum().item(),
            'n_too_fast': (hl < 1.0).sum().item(),
            'n_too_slow': (hl > 1000.0).sum().item(),
        }
    return stats


def demo():
    """Run timescale analysis on a randomly initialized model."""
    from aegis.core.mamba3_mimo import SSMConfig, Mamba3MIMO
    
    print("Creating Diagonal++ SSM model...")
    config = SSMConfig(d_model=256, d_state=64, d_inner=512, n_layers=6)
    model = Mamba3MIMO(config)
    
    kappa_data = extract_kappa(model)
    stats = analyze_timescale_distribution(kappa_data)
    
    print(f"\n{'='*60}")
    print(f"NEURAL TIMESCALE ANALYSIS")
    print(f"{'='*60}")
    print(f"Layers: {len(kappa_data)}")
    print(f"Dimensions per layer: {kappa_data[list(kappa_data.keys())[0]]['kappa'].shape[0]}")
    print()
    
    for layer_idx, s in sorted(stats.items()):
        print(f"Layer {layer_idx}:")
        print(f"  \u03ba: mean={s['kappa_mean']:.4f} \u00b1 {s['kappa_std']:.4f}, range=[{s['kappa_min']:.4f}, {s['kappa_max']:.4f}]")
        print(f"  Half-life: median={s['hl_median']:.1f} steps, geo_mean={s['hl_geometric_mean']:.1f} steps")
        print(f"  Timescale validity: {s['n_valid_timescales']} valid, {s['n_too_fast']} too fast, {s['n_too_slow']} too slow")
        print()
    
    fig = plot_kappa_heatmap(kappa_data, save_path='kappa_heatmap.png')
    
    print(f"\n{'='*60}")
    total_keep = sum((hl >= 2.0).sum().item() for d in kappa_data.values() for hl in [d['half_life']])
    total_dims = sum(d['kappa'].shape[0] for d in kappa_data.values())
    print(f"Dimensions with half-life >= 2 steps: {total_keep}/{total_dims} ({total_keep/total_dims*100:.1f}%)")
    
    return kappa_data, stats


if __name__ == "__main__":
    demo()
    print("\nNeural timescales analysis complete.")
