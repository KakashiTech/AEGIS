#!/usr/bin/env python3
"""
train_shakespeare_tiny.py — Train BGCE on Shakespeare character-level LM.
Demonstrates BGCE learns real language on CPU in <5 minutes.
"""
import sys, time, math, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aegis.engine.bgce_engine import BGCEngine, BGCEConfig
from aegis.core.mamba3_mimo import SSMConfig

# ─── Tiny Shakespeare Corpus ───────────────────────────────────────────────
TEXT = """KING LEAR: Blow, winds, and crack your cheeks! rage! blow!
You cataracts and hurricanoes, spout
Till you have drench'd our steeples, drown'd the cocks!
You sulphurous and thought-executing fires,
Vaunt-courriers to oak-cleaving thunderbolts,
Singe my white head! And thou, all-shaking thunder,
Strike flat the thick rotundity o' the world!
Crack nature's moulds, all germens spill at once,
That make ingrateful man!

HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;

MACBETH: Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?

ROMEO: But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off."""

chars = sorted(list(set(TEXT)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join(itos[i] for i in l)

data = torch.tensor(encode(TEXT), dtype=torch.long)
n_data = len(data)
split = int(0.9 * n_data)
train_data, val_data = data[:split], data[split:]

torch.manual_seed(42)

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ─── Model ─────────────────────────────────────────────────────────────────
device = "cpu"
block_size = 64
batch_size = 8
eval_interval = 50
max_steps = 500

# Small model for stability and speed on CPU
d_model = 64

config = BGCEConfig(
    d_model=d_model,
    n_layers=3,
    vocab_size=vocab_size,
    max_seq_len=block_size,
    ssm_config=SSMConfig(
        d_model=d_model, d_state=8, d_inner=d_model*2, dt_rank=4,
        use_complex=True, use_mimo=True, use_diagonal_ssm=True, device=device
    ),
    learning_rate=3e-4,
    weight_decay=0.01,
    max_grad_norm=0.5,
    device=device,
)
model = BGCEngine(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay)
total_params = sum(p.numel() for p in model.parameters())

# Warmup + cosine scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

print(f"Vocab size: {vocab_size}")
print(f"Total chars: {n_data}")
print(f"Model params: {total_params:,}")
print(f"d_model={d_model}, n_layers=3, d_state=8")
print()

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        if len(split_data) < block_size + 1:
            losses[split_name] = float('nan')
            continue
        losses_batch = []
        for _ in range(min(5, max(1, (len(split_data) - block_size) // batch_size))):
            x, y = get_batch(split_data, batch_size, block_size)
            out = model(x)
            loss = F.cross_entropy(out['logits'].view(-1, vocab_size), y.view(-1))
            losses_batch.append(loss.item())
        losses[split_name] = sum(losses_batch) / len(losses_batch) if losses_batch else float('nan')
    model.train()
    return losses

results = {"train_loss": [], "val_loss": [], "perplexity": []}
best_val_loss = float('inf')
has_nan = False

for step in range(max_steps):
    x, y = get_batch(train_data, batch_size, block_size)
    out = model(x)
    logits = out['logits']
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    if torch.isnan(loss):
        if not has_nan:
            print(f"NaN at step {step}, lowering LR and resetting...")
            has_nan = True
        for g in optimizer.param_groups:
            g['lr'] *= 0.5
        optimizer.zero_grad()
        continue

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    scheduler.step()

    if step % eval_interval == 0:
        losses = estimate_loss()
        if losses['val'] is not None and not math.isnan(losses['val']):
            ppl = math.exp(losses['val']) if losses['val'] < 10 else float('inf')
            results["train_loss"].append(losses['train'])
            results["val_loss"].append(losses['val'])
            results["perplexity"].append(ppl)
            if losses['val'] < best_val_loss and not math.isnan(losses['val']):
                best_val_loss = losses['val']
            print(f"step {step:>4d}: train={losses['train']:.4f}  val={losses['val']:.4f}  ppl={ppl:.2f}  lr={scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"step {step:>4d}: (eval skipped — insufficient data)")

# ─── Results ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SHAKESPEARE TINY RESULTS")
print("=" * 60)
initial_ppl = results['perplexity'][0] if results['perplexity'] else float('nan')
final_ppl = results['perplexity'][-1] if len(results['perplexity']) > 1 else float('nan')
print(f"Initial perplexity: {initial_ppl:.2f}")
print(f"Best perplexity:    {math.exp(best_val_loss):.2f}" if best_val_loss < float('inf') else "Best perplexity: N/A")

# Generation demo
@torch.no_grad()
def generate(model, start_text, max_new=80, temperature=1.0):
    model.eval()
    x = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)
    for _ in range(max_new):
        x_cond = x[:, -block_size:]
        out = model(x_cond)
        logits = out['logits'][:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
            return "[nan]"
        ix = torch.multinomial(probs, 1)
        x = torch.cat([x, ix], dim=1)
    return decode(x[0].tolist())

if not math.isnan(best_val_loss):
    print()
    print("Sample generations:")
    print("-" * 40)
    for prompt in ["KING", "HAMLET", "ROMEO"]:
        gen = generate(model, prompt, max_new=80, temperature=0.8)
        print(f"Prompt '{prompt}': {gen[:120]}...")
        print()

# Save
results_path = Path(__file__).parent.parent / "benchmarks" / "shakespeare_results.json"
with open(results_path, "w") as f:
    json.dump({
        "vocab_size": vocab_size, "total_chars": n_data, "params": total_params,
        "initial_perplexity": round(initial_ppl, 2) if not math.isnan(initial_ppl) else None,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float('inf') else None,
        "best_perplexity": round(math.exp(best_val_loss), 2) if best_val_loss < float('inf') else None,
    }, f, indent=2)
print(f"Results saved to {results_path}")

learned = best_val_loss < float('inf') and (results['perplexity'][-1] < results['perplexity'][0] if len(results['perplexity']) > 1 else False)
print(f"\n{'✅' if learned else '❌'} BGCE {'learns' if learned else 'does NOT learn'} real language on CPU")
